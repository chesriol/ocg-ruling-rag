# -*- coding: utf-8 -*-
"""卡片文本检索：从中文卡库按卡名精确取卡片效果文本。

为什么单独做一层，而不是把卡片也塞进向量库
--------------------------------------------
中文卡库有 1.5 万张卡。全部向量化会带来两个问题：
  1. 578 条规则片段会被 1.5 万条卡片文本彻底淹没，规则检索质量必然下降；
  2. 卡名是**精确匹配**问题 —— 问题里写的是「灰流丽」就是「灰流丽」，
     用向量近似去猜反而更差。

所以这里的做法与参考实现一致：**从问题里识别卡名 → 精确取卡片文本 → 作为证据注入**。

数据源与命名体系
----------------
  data/cards-zh-CN.cdb  mycard/ygopro-database 民间译名（与 ocg-rule 站点同一套命名）
  data/cards-zh-SC.cdb  官方简中，作为别名补充

实测：评测题的「」引号内卡名，zh-CN 命中 90.7%，zh-SC 仅 30.7%。
命名体系不同（zh-CN「平行瞬间移动」vs zh-SC「平行瞬移」），所以两个库都加载、
zh-CN 的名字优先显示、zh-SC 的名字只作别名参与匹配。
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent
CARD_DB_PATHS = (
    ROOT / "data" / "cards-zh-CN.cdb",
    ROOT / "data" / "cards-zh-SC.cdb",
)

# ygopro 卡库的类型位标志（只解这三类；子类型位标志复杂且易错，宁可不输出错信息）
TYPE_MONSTER = 0x1
TYPE_SPELL = 0x2
TYPE_TRAP = 0x4

MAX_CARDS = 6              # 单题最多注入几张卡，避免上下文被卡片撑爆
MIN_SUBSTRING_NAME = 3     # 无引号提及时的最短卡名，降低误匹配
RE_QUOTED = re.compile(r"「([^」]{1,40})」")


@dataclass(frozen=True)
class Card:
    card_id: int
    name: str
    kind: str          # 怪兽 / 魔法 / 陷阱
    stats: str         # 怪兽的等级与攻守，如 "★3 攻0/守1800"；其他为空
    desc: str
    aliases: tuple

    def as_text(self) -> str:
        """注入提示词时使用的卡片文本。"""
        header = "%s\n[%s%s]" % (self.name, self.kind,
                                 (" " + self.stats) if self.stats else "")
        return "%s\n%s" % (header, self.desc)


def describe(type_flags: int, atk: int, defence: int, level: int):
    """返回 (类别, 数值描述)。只输出能确定的信息。"""
    if type_flags & TYPE_SPELL:
        return "魔法", ""
    if type_flags & TYPE_TRAP:
        return "陷阱", ""
    if type_flags & TYPE_MONSTER:
        parts = []
        if level and level > 0:
            parts.append("★%d" % level)
        if atk is not None:
            parts.append("攻%s" % (atk if atk >= 0 else "?"))
        if defence is not None:
            parts.append("守%s" % (defence if defence >= 0 else "?"))
        return "怪兽", " ".join(parts)
    return "未知", ""


class CardIndex:
    """卡名 → 卡片文本的精确索引。找不到数据文件时自动降级为空索引。"""

    def __init__(self, paths=CARD_DB_PATHS, verbose: bool = False):
        self.cards = {}          # card_id -> Card
        self.names = {}          # 任意别名 -> card_id
        self.loaded_paths = []
        for path in paths:
            if Path(path).exists():
                self._load(Path(path))
                self.loaded_paths.append(Path(path).name)
        if verbose and self.names:
            print("卡片库：%d 张卡 / %d 个可匹配名字（%s）"
                  % (len(self.cards), len(self.names), "、".join(self.loaded_paths)))

    def _load(self, path: Path) -> None:
        connection = sqlite3.connect("file:%s?mode=ro" % path.as_posix(), uri=True)
        try:
            rows = connection.execute(
                "SELECT t.id, t.name, t.desc, d.type, d.atk, d.def, d.level "
                "FROM texts t LEFT JOIN datas d ON d.id = t.id").fetchall()
        finally:
            connection.close()

        for card_id, name, desc, type_flags, atk, defence, level in rows:
            name = (name or "").strip()
            if not name:
                continue
            existing = self.cards.get(card_id)
            if existing is None:
                kind, stats = describe(type_flags or 0, atk, defence, level or 0)
                self.cards[card_id] = Card(card_id, name, kind, stats,
                                           (desc or "").strip(), (name,))
            else:
                # 同一张卡的另一套译名：并入别名，但显示名保持不变
                self.cards[card_id] = Card(existing.card_id, existing.name,
                                           existing.kind, existing.stats,
                                           existing.desc, existing.aliases + (name,))
            self.names.setdefault(name, card_id)

        # 无引号提及时的最长优先匹配需要按名字长度降序
        self._long_names = sorted(
            (n for n in self.names if len(n) >= MIN_SUBSTRING_NAME),
            key=len, reverse=True)

    @property
    def available(self) -> bool:
        return bool(self.names)

    def lookup(self, text: str) -> list:
        """从问题文本里识别卡片，返回去重后的 Card 列表（最多 MAX_CARDS 张）。"""
        if not self.names or not text:
            return []
        found, seen = [], set()

        def add(name):
            card_id = self.names.get(name)
            if card_id is None or card_id in seen:
                return
            seen.add(card_id)
            found.append(self.cards[card_id])

        # 1) 「」引号内是站点写卡名的固定约定，最可靠
        for quoted in RE_QUOTED.findall(text):
            add(quoted)
        # 2) 再补无引号的提及（长名优先，避免短名误匹配）
        for name in self._long_names:
            if len(found) >= MAX_CARDS:
                break
            if name in text:
                add(name)
        return found[:MAX_CARDS]

    def as_documents(self, cards) -> list:
        """把卡片转成与规则片段同构的 Document，供统一编号与引用。"""
        return [Document(
            page_content=card.as_text(),
            metadata={
                "role": "card-text",
                "card_name": card.name,
                "card_id": card.card_id,
                "source_file": "cards.cdb",
            }) for card in cards]
