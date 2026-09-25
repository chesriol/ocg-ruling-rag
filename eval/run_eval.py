# -*- coding: utf-8 -*-
"""三配置对比评测：闭卷 / 纯向量 / 向量+rerank / BM25+向量 混合。

设计与取舍：
  * 只评测 goldens.json 里 gradeable=true 的单选题（官方题优先），不用 LLM 当裁判，
    准确率是确定性的数字。
  * 结果按题增量写入 eval/results/<config>.jsonl，可中断续跑（长跑必备）。
  * 第 6 章考卷已在建库阶段从索引里剔除，所以这里不存在"检索到答案原文"的泄漏。

用法：
  python eval/run_eval.py --limit 10                # 冒烟测试
  python eval/run_eval.py                           # 全量（可续跑）
  python eval/run_eval.py --report                  # 只出报告表
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
GOLDENS_PATH = EVAL_DIR / "goldens.json"
RESULTS_DIR = EVAL_DIR / "results"
REPORT_PATH = EVAL_DIR / "report.md"

DB_DIR = str(ROOT / "chroma_db")
COLLECTION = "ocg-rule"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "qwen2.5:7b"

RECALL_K = 20   # 粗召回条数
FINAL_K = 3     # 默认送进提示词的条数
DEEP_K = 8      # 加大上下文条数，用于验证"片段给少了"这一假设
ALL_CONFIGS = ["closed_book", "vector", "vector_deep", "vector_rerank", "hybrid"]

PROMPT_WITH_CONTEXT = """你是游戏王 OCG 规则裁判。请仅依据下面的【参考资料】判断这道单项选择题的答案。
资料可能不完整，但你必须在 A/B/C/D/E 中选出一个最可能的答案。

【参考资料】
{context}

【题目】
{stem}
{options}

只输出一个选项字母（A/B/C/D/E），不要输出任何解释。"""

PROMPT_CLOSED_BOOK = """你是游戏王 OCG 规则裁判。请判断这道单项选择题的答案。

【题目】
{stem}
{options}

只输出一个选项字母（A/B/C/D/E），不要输出任何解释。"""


# ---------------------------------------------------------------- 检索组件

def format_options(question: dict) -> str:
    return "\n".join("%s. %s" % (key, question["options"][key])
                     for key in sorted(question["options"]))


def build_bm25(documents):
    """字符二元组 BM25：中文无需分词依赖，够用且可复现。"""
    def grams(text: str):
        text = re.sub(r"\s+", "", text)
        return [text[i:i + 2] for i in range(max(len(text) - 1, 1))]

    doc_grams = [grams(text) for text in documents]
    doc_len = [len(item) for item in doc_grams]
    average_len = sum(doc_len) / max(len(doc_len), 1)
    document_frequency = Counter()
    for item in doc_grams:
        document_frequency.update(set(item))
    total = len(doc_grams)

    def score(query: str, index: int, k1: float = 1.5, b: float = 0.75):
        counts = Counter(doc_grams[index])
        value = 0.0
        for gram in set(grams(query)):
            frequency = counts.get(gram, 0)
            if not frequency:
                continue
            idf = math.log(1 + (total - document_frequency[gram] + 0.5)
                           / (document_frequency[gram] + 0.5))
            denominator = frequency + k1 * (1 - b + b * doc_len[index] / average_len)
            value += idf * frequency * (k1 + 1) / denominator
        return value

    return score


def vector_search(store, question, top_k):
    return store.similarity_search(question, k=top_k)


def lexical_search(documents, metadatas, score, question, top_k):
    ranked = sorted(range(len(documents)), key=lambda i: -score(question, i))[:top_k]
    # 统一返回 Document，避免与向量检索的返回类型不一致（混合配置要合并两个队列）
    return [Document(page_content=documents[i], metadata=dict(metadatas[i] or {}))
            for i in ranked]


def bm25_pick(question, documents, metadatas, score, top_k=RECALL_K):
    return lexical_search(documents, metadatas, score, question, top_k)


# ---------------------------------------------------------------- 各配置的上下文

def context_for(config, question_text, store, bm25_state, reranker):
    """返回 (context 文本, 检索耗时毫秒, 命中的 chunk 数)。"""
    started = time.perf_counter()
    if config == "closed_book":
        return "", 0.0, 0

    if config == "vector":
        docs = vector_search(store, question_text, FINAL_K)
        chunks = docs
    elif config == "vector_deep":
        chunks = vector_search(store, question_text, DEEP_K)
    elif config == "vector_rerank":
        candidates = vector_search(store, question_text, RECALL_K)
        chunks = reranker(question_text, candidates, FINAL_K)
    elif config == "hybrid":
        documents, metadatas, score = bm25_state
        dense = vector_search(store, question_text, RECALL_K)
        lexical = bm25_pick(question_text, documents, metadatas, score, RECALL_K)
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
                    break
            if len(merged) >= FINAL_K:
                break
        chunks = merged
    else:
        raise ValueError("未知配置：%s" % config)

    parts = []
    for order, doc in enumerate(chunks, start=1):
        label = doc.metadata.get("page_label")
        if not label:
            label = "补充笔记" if doc.metadata.get("role") == "rule-note" else "未标注页码"
        else:
            label = "p.%s" % label
        parts.append("[%d] (规则书 %s)\n%s" % (order, label, doc.page_content))
    return "\n\n".join(parts), (time.perf_counter() - started) * 1000, len(chunks)


# ---------------------------------------------------------------- 主流程

def load_reranker():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
    model.eval()
    if device == "cuda":
        model = model.half()
    model = model.to(device)

    def rerank(question, candidates, final_k):
        if not candidates:
            return []
        pairs = [[question, doc.page_content] for doc in candidates]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True,
                               return_tensors="pt", max_length=512).to(device)
            scores = model(**inputs, return_dict=True).logits.view(-1).float()
        order = sorted(zip(candidates, scores.tolist()), key=lambda item: -item[1])
        return [doc for doc, _ in order[:final_k]]

    rerank.device = device
    return rerank


def load_store():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings,
                   collection_name=COLLECTION)
    return store


def load_goldens(limit=None, official_only=True):
    payload = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))
    questions = [q for q in payload["questions"] if q["gradeable"]]
    if official_only:
        questions = [q for q in questions if q["official"]]
    if limit:
        questions = questions[:limit]
    return questions


def load_done(path: Path):
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    continue
    return done


def extract_letter(text: str):
    match = re.search(r"[A-E]", text.upper())
    return match.group(0) if match else None


def run(configs, limit, official_only, quiet=False):
    from langchain_ollama import OllamaLLM

    questions = load_goldens(limit=limit, official_only=official_only)
    store = load_store()
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)

    needs_reranker = "vector_rerank" in configs
    reranker = load_reranker() if needs_reranker else None
    if needs_reranker and not quiet:
        print("rerank 设备：%s" % reranker.device)

    bm25_state = None
    if "hybrid" in configs:
        raw = store.get(include=["documents", "metadatas"])
        bm25_state = (raw["documents"], raw["metadatas"], build_bm25(raw["documents"]))
        if not quiet:
            print("BM25 语料：%d 条" % len(raw["documents"]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for config in configs:
        result_path = RESULTS_DIR / ("%s.jsonl" % config)
        done = load_done(result_path)
        todo = [q for q in questions if q["id"] not in done]
        if not quiet:
            print("\n=== 配置 %s：待跑 %d 题（已完成 %d）===" % (config, len(todo), len(done)))
        with result_path.open("a", encoding="utf-8") as handle:
            for order, question in enumerate(todo, start=1):
                question_text = question["stem"]
                context, retrieval_ms, chunk_count = context_for(
                    config, question_text, store, bm25_state, reranker)
                template = PROMPT_CLOSED_BOOK if config == "closed_book" else PROMPT_WITH_CONTEXT
                prompt = template.format(context=context, stem=question_text,
                                         options=format_options(question))
                started = time.perf_counter()
                try:
                    raw_answer = llm.invoke(prompt)
                    error = None
                except Exception as exc:  # 单题失败不拖垮整轮
                    raw_answer, error = "", "%s: %s" % (type(exc).__name__, exc)
                generation_ms = (time.perf_counter() - started) * 1000

                predicted = extract_letter(raw_answer or "")
                record = {
                    "id": question["id"],
                    "config": config,
                    "set_key": question["set_key"],
                    "official": question["official"],
                    "gold": question["answer"],
                    "predicted": predicted,
                    "correct": predicted == question["answer"],
                    "retrieval_ms": round(retrieval_ms, 1),
                    "generation_ms": round(generation_ms, 1),
                    "context_chunks": chunk_count,
                    "raw_answer": (raw_answer or "")[:200],
                    "error": error,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if not quiet and order % 10 == 0:
                    print("  %d/%d" % (order, len(todo)))
    if not quiet:
        print()


def percentile(values, ratio):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(int(len(ordered) * ratio), len(ordered) - 1)
    return ordered[position]


def mcnemar_p(improved, worsened):
    """McNemar 精确检验（双侧）。

    配对设计下只关心"结论不同的题"：检索让它答对的 b 道、答错的 c 道。
    p 判断这个差异是否可能只是抽样噪声 —— 样本只有一两百题时非常必要。
    """
    total = len(improved) + len(worsened)
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(len(improved), len(worsened)) + 1))
    return min(1.0, 2 * tail / (2 ** total))


def load_records(config):
    path = RESULTS_DIR / ("%s.jsonl" % config)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if record.get("error") is None:
            records.append(record)
    return records


def golden_stems():
    """题目 id → 题干，供报告里引用具体题面。"""
    if not GOLDENS_PATH.exists():
        return {}
    payload = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))
    return {q["id"]: q["stem"] for q in payload["questions"]}


def report(configs):
    ordered = [name for name in configs if load_records(name)]
    stems = golden_stems()
    rows, by_config = {}, {name: {r["id"]: r for r in load_records(name)} for name in ordered}
    order_of = {name: index for index, name in enumerate(configs)}
    for config in ordered:
        records = list(by_config[config].values())
        correct = sum(1 for r in records if r["correct"])
        per_set = defaultdict(lambda: [0, 0])
        for record in records:
            cell = per_set[record["set_key"]]
            cell[0] += 1
            cell[1] += 1 if record["correct"] else 0
        rows[config] = {
            "n": len(records),
            "accuracy": correct / len(records),
            "correct": correct,
            "per_set": {k: (v[1], v[0]) for k, v in per_set.items()},
            "retrieval_p50": percentile([r["retrieval_ms"] for r in records], 0.50),
            "retrieval_p95": percentile([r["retrieval_ms"] for r in records], 0.95),
            "generation_p50": percentile([r["generation_ms"] for r in records], 0.50),
            "total_p50": percentile([r["retrieval_ms"] + r["generation_ms"] for r in records], 0.50),
        }

    lines = ["# 评测结果", "",
             "评测集：`eval/goldens.json` 里 gradeable 的官方单选题（答案键取自规则书第 6 章）。",
             "评测脚本：`python eval/run_eval.py`（结果按题增量落盘，可中断续跑）。", "",
             "| 配置 | 题数 | 准确率 | 正确/总数 | 检索 P50 | 检索 P95 | 生成 P50 | 端到端 P50 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    label = {
        "closed_book": "闭卷（无检索·基线）",
        "vector": "纯向量 top3",
        "vector_deep": "纯向量 top8",
        "vector_rerank": "向量20 + rerank3",
        "hybrid": "BM25+向量融合 top3",
    }
    for config in sorted(rows, key=lambda name: order_of.get(name, 99)):
        row = rows[config]
        lines.append("| %s | %d | **%.1f%%** | %d/%d | %.0f ms | %.0f ms | %.0f ms | %.0f ms |" % (
            label.get(config, config), row["n"], row["accuracy"] * 100,
            row["correct"], row["n"], row["retrieval_p50"], row["retrieval_p95"],
            row["generation_p50"], row["total_p50"]))

    if "closed_book" in rows:
        lines += ["", "## 相对闭卷基线的增益", "",
                  "闭卷基线代表模型自身的规则常识；低于或接近基线说明检索没有帮上忙。", ""]
        base = rows["closed_book"]["accuracy"]
        for config in sorted(rows, key=lambda name: order_of.get(name, 99)):
            if config == "closed_book":
                continue
            delta = (rows[config]["accuracy"] - base) * 100
            lines.append("- %s：%+.1f 个百分点" % (label.get(config, config), delta))

    set_keys = sorted({k for r in rows.values() for k in r["per_set"]})
    lines += ["", "## 分年度准确率", "",
              "| 配置 | " + " | ".join(set_keys) + " |",
              "| --- | " + " | ".join("---:" for _ in set_keys) + " |"]
    for config in sorted(rows, key=lambda name: order_of.get(name, 99)):
        cells = ["%d/%d" % rows[config]["per_set"][k] if k in rows[config]["per_set"] else "-"
                 for k in set_keys]
        lines.append("| %s | %s |" % (label.get(config, config), " | ".join(cells)))

    # 错误分析：哪些题所有配置都答错（多为检索覆盖不到的章节或依赖图片）
    common = set.intersection(*[set(by_config[name]) for name in ordered]) if ordered else set()
    always_wrong = sorted(i for i in common if not any(by_config[n][i]["correct"] for n in ordered))
    if always_wrong:
        lines += ["", "## 所有配置都答错的题（%d 道）" % len(always_wrong), "",
                  "这些题目的失败原因通常是检索语料里没有对应章节，而不是提示词问题。", ""]
        sample = by_config[ordered[0]]
        for identifier in always_wrong[:15]:
            record = sample[identifier]
            lines.append("- `%s`（%s）答案 %s，题干：%s" % (
                identifier, record["set_key"], record["gold"],
                stems.get(identifier, "")[:70]))
        if len(always_wrong) > 15:
            lines.append("- …以及另外 %d 道" % (len(always_wrong) - 15))

    # 检索是否真的帮上忙：只看两者都跑过的题，并做配对显著性检验
    if "closed_book" in by_config:
        base_point = by_config["closed_book"]
        lines += ["", "## 与闭卷基线的配对比较", "",
                  "同一批题目下比较，只看结论不同的题（improved / worsened），"
                  "并用 McNemar 精确检验判断差异是否可能只是噪声。", ""]
        for config in ordered:
            if config == "closed_book":
                continue
            improved, worsened = [], []
            for identifier, record in by_config[config].items():
                other = base_point.get(identifier)
                if not other:
                    continue
                if record["correct"] and not other["correct"]:
                    improved.append(identifier)
                elif other["correct"] and not record["correct"]:
                    worsened.append(identifier)
            probability = mcnemar_p(improved, worsened)
            verdict = "显著" if probability < 0.05 else "不显著"
            lines += ["- **%s**：检索后答对 %d 道，检索后答错 %d 道，"
                      "McNemar p=%.4f（%s）" % (
                          label.get(config, config), len(improved), len(worsened),
                          probability, verdict)]
            if improved:
                lines.append("  - 答对：%s" % "、".join("`%s`" % i for i in improved[:12]))
            if worsened:
                lines.append("  - 答错：%s" % "、".join("`%s`" % i for i in worsened[:12]))

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n已写出：%s" % REPORT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(ALL_CONFIGS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--all-questions", action="store_true",
                        help="包含非官方测试题（默认只跑官方题）")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    chosen = [name.strip() for name in args.configs.split(",") if name.strip()]
    if args.report:
        report(chosen)
    else:
        run(chosen, args.limit or None, official_only=not args.all_questions)
        report(chosen)
