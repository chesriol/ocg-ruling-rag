# -*- coding: utf-8 -*-
"""抓取官方规则 Q&A 语料（经 db.ygoresources.com 的机器可读镜像）。

为什么走镜像而不是官网
----------------------
官方网站（www.db.yugioh-card.com）是 JS 渲染的 SPA：`faq_search.action` 返回的
77 KB HTML 里**没有问答正文**，内容全部由 AJAX 加载，直接爬 HTML 拿不到东西。
镜像提供结构化 JSON，并且记录官方原始出处，可逐条回溯。

为什么是日文
------------
实测官方数据库**没有中文 Q&A**：`request_locale=zh-CN` 的 FAQ 接口返回与日语版
同尺寸的页面外壳，不含「質問/回答」；镜像的 `qaData` 语言分支只有 `{ja, en}`。
所以裁定正文必然是日文 —— 唯一的本地化手段是把 `<<卡片内部id>>` 占位符
替换成中文卡名（卡名是裁定问题里信息量最大的部分），中文卡名取自镜像自己的
`/data/idx/card/name/cn`（7,461 条，一次请求拿全）。

枚举方式与抽样
--------------
没有批量接口，Q&A id 只能通过每张卡的 `qaIndex` 字段收集。全量需要 2.3 万次卡片请求
加数万次 Q&A 请求（实测 0.85 秒/次，总计 10 小时以上）。
因此这里对卡表做**等距抽样**（每 N 张取 1 张，N 由 --every 指定）——
抽样规则只看卡表顺序、与评测集无关，避免把知识库做成针对测试集的定制语料。

用法：
  python fetch_qa.py --every 20        # 抽样约 1/20 的卡（约 370 张）→ 数千条 Q&A
  python fetch_qa.py --every 20 --qa-limit 300   # 先小跑验证管线
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "qa-cache"
CARD_CACHE = CACHE_DIR / "cards"
QA_CACHE = CACHE_DIR / "qa"
NAMES_PATH = CACHE_DIR / "cn-names.json"
OUTPUT = DATA_DIR / "official-qa.jsonl"
MANIFEST = DATA_DIR / "official-qa-manifest.json"

BASE = "https://db.ygoresources.com"
OFFICIAL_URL = ("https://www.db.yugioh-card.com/yugiohdb/faq_search.action"
                "?fid={id}&ope=5&request_locale=ja")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
RE_PLACEHOLDER = re.compile(r"<<(\d+)>>")


def http_json(url, timeout=40):
    request = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/json", "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8", "replace"))


def cached_json(path: Path, url: str):
    """带磁盘缓存的取数：中断后重跑不会重复请求。"""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
    try:
        payload = http_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("null", encoding="utf-8")
            return None
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def ensure_names() -> dict:
    if NAMES_PATH.exists():
        payload = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
        return payload
    print("  拉取中文卡名索引 ...")
    raw = http_json("%s/data/idx/card/name/cn" % BASE)
    names = {}
    for name, ids in raw.items():
        for card_id in ids:
            names.setdefault(str(card_id), name)
    NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NAMES_PATH.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    return names


def localize(text: str, names: dict) -> str:
    """把 <<内部id>> 换成「中文卡名」；查不到的名字保留原 id（不编造）。"""
    def replace(match):
        return "「%s」" % names.get(match.group(1), match.group(1))
    return RE_PLACEHOLDER.sub(replace, text or "")


def parallel_map(function, items, workers, label, every=200):
    """并发取数并打印进度。缓存命中时不发请求，所以重跑很快。"""
    results = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, result in enumerate(pool.map(function, items), start=1):
            results.append(result)
            if index % every == 0:
                print("      %s %d/%d（%.0f 秒）" % (label, index, len(items), time.time() - started))
    return results


def collect_qa_ids(card_ids, sleep, workers):
    """逐卡取 qaIndex，合并出 Q&A id 集合（带缓存）。"""
    def one(card_id):
        time.sleep(sleep)
        return card_id, cached_json(CARD_CACHE / ("%s.json" % card_id),
                                    "%s/data/card/%s" % (BASE, card_id))

    qa_ids, card_of = set(), {}
    for card_id, payload in parallel_map(one, card_ids, workers, "卡片"):
        if not payload:
            continue
        for qa_id in payload.get("qaIndex") or []:
            qa_ids.add(int(qa_id))
            card_of.setdefault(int(qa_id), []).append(int(card_id))
    return qa_ids, card_of


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=20, help="每 N 张卡取 1 张")
    parser.add_argument("--qa-limit", type=int, default=0, help="限制抓取的 Q&A 条数（调试用）")
    parser.add_argument("--sleep", type=float, default=0.15, help="每次请求后的间隔秒数（礼貌限速）")
    parser.add_argument("--workers", type=int, default=4, help="并发请求数（对镜像站保持克制）")
    args = parser.parse_args()

    for directory in (CACHE_DIR, CARD_CACHE, QA_CACHE):
        directory.mkdir(parents=True, exist_ok=True)

    print("=== 1) 中文卡名索引 ===")
    names = ensure_names()
    print("  %d 个内部 id 有中文名" % len(names))

    print("\n=== 2) 对卡表等距抽样（每 %d 张取 1 张）===" % args.every)
    # 用卡名索引里出现过的 id 升序排列后等距抽样：规则只看卡表顺序，与评测集无关
    all_ids = sorted(int(key) for key in names)
    card_ids = all_ids[::args.every]
    print("  卡表 %d 张 → 抽样 %d 张" % (len(all_ids), len(card_ids)))

    print("\n=== 3) 收集 Q&A id（逐卡 qaIndex，带缓存，%d 并发）===" % args.workers)
    started = time.time()
    qa_ids, card_of = collect_qa_ids(card_ids, args.sleep, args.workers)
    print("  得到 %d 个 Q&A id，耗时 %.0f 秒" % (len(qa_ids), time.time() - started))

    ordered = sorted(qa_ids)
    if args.qa_limit:
        ordered = ordered[:args.qa_limit]

    print("\n=== 4) 抓取 Q&A 正文（%d 条，%d 并发）===" % (len(ordered), args.workers))

    def one(qa_id):
        time.sleep(args.sleep)
        return qa_id, cached_json(QA_CACHE / ("%s.json" % qa_id),
                                  "%s/data/qa/%s" % (BASE, qa_id))

    started = time.time()
    written = skipped = 0
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for index, (qa_id, payload) in enumerate(
                parallel_map(one, ordered, args.workers, "Q&A", every=100), start=1):
            if not payload:
                skipped += 1
                continue
            japanese = (payload.get("qaData") or {}).get("ja") or {}
            question = (japanese.get("question") or "").strip()
            answer = (japanese.get("answer") or "").strip()
            if not question and not answer:
                skipped += 1
                continue
            cards = [names.get(str(c), str(c)) for c in (payload.get("cards") or [])]
            handle.write(json.dumps({
                "qa_id": qa_id,
                "cards": cards,
                "question": localize(question, names),
                "answer": localize(answer, names),
                "source": OFFICIAL_URL.format(id=qa_id),
                "language": "ja-localized-cn-cardnames",
            }, ensure_ascii=False) + "\n")
            written += 1
    elapsed = time.time() - started
    print("  完成：写出 %d 条，跳过 %d 条，耗时 %.0f 秒" % (written, skipped, elapsed))

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "db.ygoresources.com（官方数据库 FAQ 的机器可读镜像）",
        "official_source_url_template": OFFICIAL_URL,
        "language": "日文正文 + 中文卡名（官方库无中文 Q&A，实测确认）",
        "sampling": {
            "rule": "卡表按 id 升序后每 %d 张取 1 张；规则与评测集无关" % args.every,
            "cards_total": len(all_ids),
            "cards_sampled": len(card_ids),
            "qa_ids_found": len(qa_ids),
            "qa_fetched": written,
        },
        "output": OUTPUT.name,
        "sha256": digest,
        "records": written,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n语料：%s（%d 条）" % (OUTPUT, written))
    print("manifest：%s" % MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
