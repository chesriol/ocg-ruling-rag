# -*- coding: utf-8 -*-
"""测量评测的推理噪声下限。

为什么需要
----------
`temperature=0` **并不等于确定性**。同一份提示词重复送给 qwen2.5:7b，
预测会以百分之几的量级抖动（实测两两差异 2%~6%）。
不先量出这个下限，就无法判断"某配置提升 1~2 个百分点"是真效果还是噪声。

本项目实测（198 道官方单选题，3 次重复）：准确率 50 / 48 / 51，
即 **±1.5 道题**；有 12 道题三次结果不完全一致。

用法：
  python eval/measure_noise.py --config closed_book --repeat 3
  python eval/measure_noise.py --config vector_rerank --repeat 3 --keep
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import rag_core  # noqa: E402
import run_eval  # noqa: E402


def load(path: Path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="closed_book", choices=list(rag_core.MODES))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep", action="store_true", help="保留结果目录以便复查")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="rag-noise-"))
    print("配置 %s，重复 %d 次，结果目录 %s\n" % (args.config, args.repeat, workdir))

    for index in range(args.repeat):
        target = workdir / ("run%d" % (index + 1))
        print("第 %d 次 ..." % (index + 1))
        run_eval.run([args.config], args.limit or None, official_only=True,
                     quiet=True, results_dir=target)

    runs = {}
    for index in range(args.repeat):
        path = workdir / ("run%d" % (index + 1)) / ("%s.jsonl" % args.config)
        if path.exists():
            runs["第%d次" % (index + 1)] = load(path)
    if len(runs) < 2:
        print("重跑失败，结果不足两次", file=sys.stderr)
        return 1

    print("\n=== 准确率 ===")
    for name, records in runs.items():
        correct = sum(1 for r in records.values() if r["correct"])
        print("  %-8s %3d/%-3d  %5.1f%%" % (name, correct, len(records),
                                            100 * correct / len(records)))

    print("\n=== 两两预测差异（提示词完全相同，差异只可能来自推理不确定性）===")
    for left, right in itertools.combinations(runs, 2):
        common = set(runs[left]) & set(runs[right])
        different = [i for i in common
                     if runs[left][i]["predicted"] != runs[right][i]["predicted"]]
        print("  %-8s vs %-8s  不同 %2d 道 (%.1f%%)" % (
            left, right, len(different), 100 * len(different) / max(len(common), 1)))

    unstable = Counter()
    for identifier in set.intersection(*[set(r) for r in runs.values()]):
        if len({runs[name][identifier]["predicted"] for name in runs}) > 1:
            unstable[identifier] += 1
    counts = [sum(1 for r in records.values() if r["correct"]) for records in runs.values()]
    print("\n=== 噪声下限 ===")
    print("  准确率跨度 %d ~ %d 道题（±%.1f）" % (min(counts), max(counts),
                                                  (max(counts) - min(counts)) / 2))
    print("  多次结果不完全一致的题：%d 道（%.1f%%）" % (
        len(unstable), 100 * len(unstable) / max(len(runs["第1次"]), 1)))
    print("\n  含义：小于这个幅度的配置差异无法与噪声区分，不应作为结论。")

    if args.keep:
        print("\n结果保留在：%s" % workdir)
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
