# -*- coding: utf-8 -*-
"""游戏王 OCG 规则助手（检索增强 + 精排 + 页码引用）。

相对上一版修掉的问题：
  1. reranker 自动使用 GPU（实测 CPU 每问 50 秒以上，GPU 降到毫秒级）。
  2. 答案附带引用页码 —— 索引里本来就有 page_label，旧版从没返回给用户，
     导致用户无法核对，模型说错也看不出来。
  3. 提示词要求用 [n] 标注依据，并对"资料中没有"的情况明确拒答。
  4. 输出流式打印，并把检索/生成耗时分开显示，瓶颈一眼可见。

用法：
  python query_rag.py
  python query_rag.py --no-rerank      # 关闭精排，对比速度与效果
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DB_DIR = str(ROOT / "chroma_db")
COLLECTION = "ocg-rule"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "qwen2.5:7b"

RECALL_K = 20
FINAL_K = 3

PROMPT = """你是游戏王 OCG 规则助手。请依据【参考资料】回答问题，并遵守：
- 每条结论后面用 [编号] 标注依据来自哪段资料。
- 资料里没有的信息，直接说明"资料中没有相关信息"，不要凭印象补充。
- 不要引用【参考资料】之外的页码或规则。

【参考资料】
{context}

【问题】{question}

【回答】"""


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Reranker:
    """两阶段检索的精排阶段：把粗召回的候选按与问题的相关性重排。"""

    def __init__(self):
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        self.model.eval()                      # 关掉 dropout，保证同一问题结果稳定
        if self.device == "cuda":
            self.model = self.model.half()     # 半精度：显存和速度都更划算
        self.model = self.model.to(self.device)

    def rank(self, question: str, candidates):
        if not candidates:
            return []
        pairs = [[question, doc.page_content] for doc in candidates]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                    return_tensors="pt", max_length=512).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
        order = sorted(zip(candidates, scores.tolist()), key=lambda item: -item[1])
        return order


def retrieve(store, question, reranker=None):
    """粗召回 RECALL_K 条；有 reranker 则精排后取 FINAL_K 条。"""
    started = time.perf_counter()
    candidates = store.similarity_search(question, k=RECALL_K if reranker else FINAL_K)
    if reranker:
        ranked = reranker.rank(question, candidates)[:FINAL_K]
    else:
        ranked = [(doc, None) for doc in candidates[:FINAL_K]]
    return ranked, (time.perf_counter() - started) * 1000


def page_label_of(doc) -> str:
    """页码标签；补充笔记没有 PDF 页码，不要显示成 p.?。"""
    label = doc.metadata.get("page_label")
    if label:
        return "p.%s" % label
    if doc.metadata.get("role") == "rule-note":
        return "补充笔记"
    return "未标注页码"


def build_context(ranked):
    lines = []
    for order, (doc, _score) in enumerate(ranked, start=1):
        lines.append("[%d] 规则书 %s\n%s" % (order, page_label_of(doc), doc.page_content))
    return "\n\n".join(lines)


def answer(store, llm, question, reranker=None):
    ranked, retrieval_ms = retrieve(store, question, reranker)
    if not ranked:
        print("没有检索到相关资料。")
        return retrieval_ms, 0.0, []
    prompt = PROMPT.format(context=build_context(ranked), question=question)
    started = time.perf_counter()
    for piece in llm.stream(prompt):           # 流式输出，避免长时间"思考中..."
        sys.stdout.write(piece)
        sys.stdout.flush()
    return retrieval_ms, (time.perf_counter() - started) * 1000, ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true", help="关闭精排，用于对比")
    args = parser.parse_args()

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings,
                   collection_name=COLLECTION)
    reranker = None if args.no_rerank else Reranker()
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)

    device = reranker.device if reranker else "未启用"
    print("OCG 规则助手已启动（精排设备：%s，输入 q 退出）" % device)

    while True:
        try:
            question = input("\n你问：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() == "q":
            break
        print("AI答：", end="")
        retrieval_ms, generation_ms, ranked = answer(store, llm, question, reranker)
        print("\n\n依据：")
        for order, (doc, score) in enumerate(ranked, start=1):
            score_text = "%.3f" % score if score is not None else "—"
            print("  [%d] 规则书 %s  相关度 %s" % (order, page_label_of(doc), score_text))
        print("耗时：检索 %.0f ms / 生成 %.0f ms" % (retrieval_ms, generation_ms))


if __name__ == "__main__":
    main()
