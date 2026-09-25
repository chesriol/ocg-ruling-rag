# -*- coding: utf-8 -*-
"""下载中文卡片数据库（mycard/ygopro-database）。

为什么不在仓库里放卡库：7.7 MB 的第三方数据，且可随时重新下载。
卡库本身带版本（master 分支每次更新都会变），所以下载后会把 sha256 记进
`data/cards-manifest.json`，抓取管线与索引 manifest 都对得上版本。

用法：
  python fetch_cards.py           # 缺什么下什么
  python fetch_cards.py --force   # 强制重新下载
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST = DATA_DIR / "cards-manifest.json"
BASE = "https://raw.githubusercontent.com/mycard/ygopro-database/master/locales"

# zh-CN 是民间译名，与 ocg-rule 站点同一套命名（实测覆盖评测题 90.7% 的引号内卡名）；
# zh-SC 是官方简中，只作别名补充（覆盖率 30.7%，命名体系不同）。
LOCALES = ("zh-CN", "zh-SC")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(locale: str, target: Path) -> None:
    url = "%s/%s/cards.cdb" % (BASE, locale)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=600) as response, target.open("wb") as handle:
        total = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
            total += len(block)
    print("  ✓ %-6s %5.2f MB  %3d 秒" % (locale, total / 1048576, int(time.time() - started)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    recorded = {}
    if MANIFEST.exists():
        try:
            recorded = json.loads(MANIFEST.read_text(encoding="utf-8")).get("files", {})
        except Exception:
            recorded = {}

    files = {}
    for locale in LOCALES:
        name = "cards-%s.cdb" % locale
        target = DATA_DIR / name
        if target.exists() and not args.force:
            print("  = %-6s 已存在，跳过" % locale)
        else:
            print("  下载 %s ..." % locale)
            try:
                download(locale, target)
            except Exception as exc:
                print("  ✗ %s 下载失败：%s" % (locale, exc), file=sys.stderr)
                continue
        files[name] = {
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "url": "%s/%s/cards.cdb" % (BASE, locale),
            "previous_sha256": recorded.get(name, {}).get("sha256"),
        }

    if not files:
        print("没有可用的卡库文件", file=sys.stderr)
        return 1

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "mycard/ygopro-database",
        "files": files,
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n卡库就绪：%s" % "、".join(files))
    print("manifest：%s" % MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
