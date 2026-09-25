# -*- coding: utf-8 -*-
"""检索核心：应用（query_rag.py）与评测（eval/run_eval.py）共用的单一实现。

为什么需要这个模块
------------------
面向用户的问答和面向评分的评测**必须让模型看到完全相同的证据**，否则评测就不再代表
线上行为 —— 你测的是一个系统，上线的是另一个。此前两边各写了一份检索与提示词逻辑，
属于典型的"两处真相"，任何一边改动都会静默地让评测数字失真。

本模块是这两条路径的唯一来源：
  * 索引与模型常量
  * 向量检索 / 交叉编码器精排 / 中文 BM25 与融合
  * 证据编号与页码引用的格式
  * 两种任务的提示词模板（开放式问答 / 单选题）

检索配置（`MODES`）
-------------------
  closed_book    不检索，直接问模型 —— 衡量模型自身的规则常识，是所有对比的基线
  vector         向量 top3
  vector_deep    向量 top8（验证"片段给少了"这一假设）
  vector_rerank  向量粗召回 20 → 交叉编码器精排 → top3（两阶段检索）
  hybrid         BM25 与向量各取 20，交替融合 → top3

行为约束：本模块的检索结果与提示词格式与重构前逐字节一致，
`eval/results/` 下已发布的 990 条记录仍然可由当前代码复现。
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DB_DIR = str(ROOT / "chroma_db")
COLLECTION = "ocg-rule"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "qwen2.5:7b"

RECALL_K = 20   # 粗召回条数
FINAL_K = 3     # 默认送进提示词的条数
DEEP_K = 8      # 加大上下文条数

MODES = ("closed_book", "vector", "vector_deep", "vector_rerank", "hybrid")
MODE_LABEL = {
    "closed_book": "闭卷（无检索·基线）",
    "vector": "纯向量 top3",
    "vector_deep": "纯向量 top8",
    "vector_rerank": "向量20 + rerank3",
    "hybrid": "BM25+向量融合 top3",
}

# 两种任务的提示词。注意：开放式问答要求标注引用并允许拒答；
# 单选题要求只输出一个字母，以便确定性评分（不需要 LLM 当裁判）。
PROMPT_QA = """你是游戏王 OCG 规则助手。请依据【参考资料】回答问题，并遵守：
- 每条结论后面用 [编号] 标注依据来自哪段资料。
- 资料里没有的信息，直接说明"资料中没有相关信息"，不要凭印象补充。
- 不要引用【参考资料】之外的页码或规则。

【参考资料】
{context}

【问题】{question}

【回答】"""

PROMPT_MC_WITH_CONTEXT = """你是游戏王 OCG 规则裁判。请仅依据下面的【参考资料】判断这道单项选择题的答案。
资料可能不完整，但你必须在 A/B/C/D/E 中选出一个最可能的答案。

【参考资料】
{context}

【题目】
{stem}
{options}

只输出一个选项字母（A/B/C/D/E），不要输出任何解释。"""

PROMPT_MC_CLOSED_BOOK = """你是游戏王 OCG 规则裁判。请判断这道单项选择题的答案。

【题目】
{stem}
{options}

只输出一个选项字母（A/B/C/D/E），不要输出任何解释。"""


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings,
                  collection_name=COLLECTION)


class Reranker:
    """交叉编码器精排：逐条给（问题，候选片段）打相关性分。"""

    def __init__(self):
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        self.model.eval()                      # 关掉 dropout，同一问题结果稳定
        if self.device == "cuda":
            self.model = self.model.half()     # 半精度：显存与速度都更划算
        self.model = self.model.to(self.device)

    def rank(self, question: str, candidates):
        """返回 [(document, score)]，按分数降序。"""
        if not candidates:
            return []
        pairs = [[question, doc.page_content] for doc in candidates]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                    return_tensors="pt", max_length=512).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
        return sorted(zip(candidates, scores.tolist()), key=lambda item: -item[1])


class Bm25Index:
    """中文 BM25。

    用字符二元组代替分词：不引入 jieba 之类的额外依赖，且对中文召回足够好、
    结果完全可复现。评分公式与参数（k1=1.5, b=0.75）沿用重构前的实现。
    """

    def __init__(self, documents, metadatas, k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.metadatas = [dict(item or {}) for item in metadatas]
        self.k1, self.b = k1, b
        self._grams = [self._tokenize(text) for text in self.documents]
        self._lengths = [len(item) for item in self._grams]
        self._average_length = sum(self._lengths) / max(len(self._lengths), 1)
        self._document_frequency = Counter()
        for item in self._grams:
            self._document_frequency.update(set(item))

    @staticmethod
    def _tokenize(text: str):
        text = re.sub(r"\s+", "", text)
        return [text[index:index + 2] for index in range(max(len(text) - 1, 1))]

    def _score(self, query: str, index: int) -> float:
        counts = Counter(self._grams[index])
        total = len(self._grams)
        value = 0.0
        for gram in set(self._tokenize(query)):
            frequency = counts.get(gram, 0)
            if not frequency:
                continue
            idf = math.log(1 + (total - self._document_frequency[gram] + 0.5)
                           / (self._document_frequency[gram] + 0.5))
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * self._lengths[index] / self._average_length)
            value += idf * frequency * (self.k1 + 1) / denominator
        return value

    def search(self, query: str, top_k: int):
        ranked = sorted(range(len(self.documents)), key=lambda i: -self._score(query, i))[:top_k]
        return [Document(page_content=self.documents[i], metadata=self.metadatas[i])
                for i in ranked]


@dataclass(frozen=True)
class Retrieved:
    """一次检索的结果：送进提示词的文档、耗时、以及实际条数。"""
    mode: str
    documents: list
    elapsed_ms: float


class Retriever:
    """按配置检索。需要精排/BM25 时才加载对应组件，避免无谓的启动开销。"""

    def __init__(self, store=None, need_reranker: bool = False, need_bm25: bool = False,
                 verbose: bool = False):
        self.store = store if store is not None else build_store()
        self.reranker = Reranker() if need_reranker else None
        self.bm25 = None
        if need_bm25:
            corpus = self.store.get(include=["documents", "metadatas"])
            self.bm25 = Bm25Index(corpus["documents"], corpus["metadatas"])
            if verbose:
                print("BM25 语料：%d 条" % len(corpus["documents"]))

    @property
    def device(self) -> str:
        return self.reranker.device if self.reranker else "未启用"

    def retrieve(self, question: str, mode: str) -> Retrieved:
        started = time.perf_counter()
        if mode == "closed_book":
            return Retrieved(mode, [], 0.0)

        if mode == "vector":
            documents = self.store.similarity_search(question, k=FINAL_K)
        elif mode == "vector_deep":
            documents = self.store.similarity_search(question, k=DEEP_K)
        elif mode == "vector_rerank":
            candidates = self.store.similarity_search(question, k=RECALL_K)
            documents = [doc for doc, _ in self.reranker.rank(question, candidates)[:FINAL_K]]
        elif mode == "hybrid":
            documents = self._hybrid(question)
        else:
            raise ValueError("未知检索配置：%s" % mode)

        return Retrieved(mode, documents, (time.perf_counter() - started) * 1000)

    def _hybrid(self, question: str):
        """BM25 与向量结果交替取，按 chunk_id 去重，直到凑够 FINAL_K。

        交替（round-robin）而不是加权求和：两路分数不在同一量纲上，直接相加需要标定，
        交替融合无参数、可复现。
        """
        dense = self.store.similarity_search(question, k=RECALL_K)
        lexical = self.bm25.search(question, RECALL_K)
        merged, seen = [], set()
        for position in range(RECALL_K):
            for queue in (lexical, dense):
                if position >= len(queue):
                    continue
                item = queue[position]
                key = item.metadata.get("chunk_id") or item.page_content[:64]
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
                if len(merged) >= FINAL_K:
                    return merged
        return merged


# ------------------------------------------------------------------ 证据格式

def page_label_of(document) -> str:
    """页码标签。补充笔记没有 PDF 页码，不要显示成 p.?。"""
    label = document.metadata.get("page_label")
    if label:
        return "p.%s" % label
    if document.metadata.get("role") == "rule-note":
        return "补充笔记"
    return "未标注页码"


def build_context(documents) -> str:
    """把检索结果编号成提示词里的【参考资料】。

    编号是引用溯源的基础：模型被要求用 [n] 标注依据，用户可据此回查原页。
    """
    parts = []
    for order, document in enumerate(documents, start=1):
        parts.append("[%d] (规则书 %s)\n%s" % (order, page_label_of(document),
                                              document.page_content))
    return "\n\n".join(parts)


def format_options(options) -> str:
    return "\n".join("%s. %s" % (key, options[key]) for key in sorted(options))


def build_qa_prompt(question: str, documents) -> str:
    return PROMPT_QA.format(context=build_context(documents), question=question)


def build_mc_prompt(mode: str, context: str, stem: str, options) -> str:
    """单选题提示词。模板按检索配置选择（而非按上下文是否为空），与重构前一致。"""
    template = PROMPT_MC_CLOSED_BOOK if mode == "closed_book" else PROMPT_MC_WITH_CONTEXT
    return template.format(context=context, stem=stem, options=format_options(options))


def extract_choice(text: str):
    """从模型输出里取选项字母；取不到返回 None（计入解析失败而非算错）。"""
    match = re.search(r"[A-E]", (text or "").upper())
    return match.group(0) if match else None
