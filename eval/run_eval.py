# -*- coding: utf-8 -*-
"""多配置对比评测：闭卷 / 纯向量 / 加深上下文 / 精排 / BM25+向量 混合。

设计与取舍：
  * 只评测 goldens.json 里 gradeable 的单选题（默认只取官方题），不用 LLM 当裁判，
    准确率是确定性数字。
  * 结果按题增量写入 eval/results/<config>.jsonl，可中断续跑（长跑必备）。
  * 第 6 章考卷已在建库阶段从索引里剔除，因此不存在"检索到答案原文"的泄漏。
  * **检索与提示词来自 rag_core.py**，与 query_rag.py 共用同一份实现，
    保证评测里测的就是命令行里跑的。

用法：
  python eval/run_eval.py --limit 10                # 冒烟测试
  python eval/run_eval.py                           # 全量（可续跑）
  python eval/run_eval.py --report                  # 只出报告表
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # 让子目录里的脚本能 import rag_core

import rag_core                        # noqa: E402  （必须在 sys.path 调整之后导入）

EVAL_DIR = Path(__file__).resolve().parent
GOLDENS_PATH = EVAL_DIR / "goldens.json"
RESULTS_DIR = EVAL_DIR / "results"
REPORT_PATH = EVAL_DIR / "report.md"

ALL_CONFIGS = list(rag_core.MODES)


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


def run(configs, limit, official_only, quiet=False, results_dir=None):
    from langchain_ollama import OllamaLLM

    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    questions = load_goldens(limit=limit, official_only=official_only)
    retriever = rag_core.Retriever(
        need_reranker=any(name in configs for name in ("vector_rerank", "vector_rerank_cards")),
        need_bm25="hybrid" in configs,
        need_cards=any(name in configs for name in rag_core.CARD_MODES),
        verbose=not quiet,
    )
    if not quiet and retriever.reranker:
        print("rerank 设备：%s" % retriever.device)
    llm = OllamaLLM(model=rag_core.LLM_MODEL, temperature=0)

    results_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        result_path = results_dir / ("%s.jsonl" % config)
        done = load_done(result_path)
        todo = [q for q in questions if q["id"] not in done]
        if not quiet:
            print("\n=== 配置 %s：待跑 %d 题（已完成 %d）===" % (config, len(todo), len(done)))
        with result_path.open("a", encoding="utf-8") as handle:
            for order, question in enumerate(todo, start=1):
                question_text = question["stem"]
                retrieved = retriever.retrieve(question_text, config)
                prompt = rag_core.build_mc_prompt(
                    config, rag_core.build_context(retrieved.documents),
                    question_text, question["options"])
                started = time.perf_counter()
                try:
                    raw_answer = llm.invoke(prompt)
                    error = None
                except Exception as exc:  # 单题失败不拖垮整轮
                    raw_answer, error = "", "%s: %s" % (type(exc).__name__, exc)
                generation_ms = (time.perf_counter() - started) * 1000

                predicted = rag_core.extract_choice(raw_answer or "")
                record = {
                    "id": question["id"],
                    "config": config,
                    "set_key": question["set_key"],
                    "official": question["official"],
                    "gold": question["answer"],
                    "predicted": predicted,
                    "correct": predicted == question["answer"],
                    "retrieval_ms": round(retrieved.elapsed_ms, 1),
                    "generation_ms": round(generation_ms, 1),
                    "context_chunks": len(retrieved.documents),
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


def load_records(config, results_dir=None):
    path = (Path(results_dir) if results_dir else RESULTS_DIR) / ("%s.jsonl" % config)
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


def report(configs, results_dir=None):
    ordered = [name for name in configs if load_records(name, results_dir)]
    stems = golden_stems()
    rows, by_config = {}, {name: {r["id"]: r for r in load_records(name, results_dir)}
                           for name in ordered}
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

    label = rag_core.MODE_LABEL
    lines = ["# 评测结果", "",
             "评测集：`eval/goldens.json` 里 gradeable 的官方单选题（答案键取自规则书第 6 章）。",
             "评测脚本：`python eval/run_eval.py`（结果按题增量落盘，可中断续跑）。",
             "检索与提示词来自 `rag_core.py`，与 `query_rag.py` 共用同一份实现。", "",
             "| 配置 | 题数 | 准确率 | 正确/总数 | 检索 P50 | 检索 P95 | 生成 P50 | 端到端 P50 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
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
                      "McNemar p=%.3f（%s）" % (
                          label.get(config, config), len(improved), len(worsened),
                          probability, verdict)]
            if improved:
                lines.append("  - 答对：%s" % "、".join("`%s`" % i for i in improved[:12]))
            if worsened:
                lines.append("  - 答错：%s" % "、".join("`%s`" % i for i in worsened[:12]))

    report_path = ((Path(results_dir) if results_dir else RESULTS_DIR).parent
                   / "report.md" if results_dir else REPORT_PATH)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n已写出：%s" % report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(ALL_CONFIGS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--all-questions", action="store_true",
                        help="包含非官方测试题（默认只跑官方题）")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--results-dir", default=None,
                        help="结果目录（默认 eval/results）。用于非破坏性试跑，"
                             "例如重复跑同一配置来测量 LLM 推理噪声下限")
    args = parser.parse_args()

    chosen = [name.strip() for name in args.configs.split(",") if name.strip()]
    if args.report:
        report(chosen, args.results_dir)
    else:
        run(chosen, args.limit or None, official_only=not args.all_questions,
            results_dir=args.results_dir)
        report(chosen, args.results_dir)
