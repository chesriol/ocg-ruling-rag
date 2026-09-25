# -*- coding: utf-8 -*-
"""rag_core 引用校验的单元测试。

纯逻辑、零依赖、不联网、不加载模型 —— 任何机器上 clone 下来都能直接跑：

    python tests/test_rag_core.py

（刻意不用 pytest：这个项目不需要为了跑 10 个断言多装一个依赖。）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_core  # noqa: E402

# (用例名, 答案文本, 本次提供的资料段数, 期望的 cited, 期望的 invalid, 期望的 unused)
CASES = [
    ("单个编号 [1]", "根据 [1]，可以发动。", 3, (1,), (), (2, 3)),
    ("编造编号 [4]（只给了 3 段资料）",
     "根据参考资料 [1] 和 [4]，不能发动。", 3, (1, 4), (4,), (2, 3)),
    ("带前缀 [编号：2]", "见 [编号：2]。", 3, (2,), (), (1, 3)),
    ("带前缀半角冒号 [编号:3]", "见 [编号:3]。", 3, (3,), (), (1, 2)),
    ("一格内多个编号 [1,2]", "见 [1,2]。", 3, (1, 2), (), (3,)),
    ("顿号分隔 [1、3]", "见 [1、3]。", 3, (1, 3), (), (2,)),
    ("编号 0 也属越界", "见 [0]。", 3, (0,), (0,), (1, 2, 3)),
    ("完全没有引用", "可以发动。", 3, (), (), (1, 2, 3)),
    ("非编号方括号不算引用", "这张卡是 [UR] 稀有度。", 3, (), (), (1, 2, 3)),
    ("引用全部资料", "见 [1][2][3]。", 3, (1, 2, 3), (), ()),
    ("无资料时不报未引用", "资料里没有相关信息。", 0, (), (), ()),
    # 下面两条是实测踩过的误报：模型会照抄资料里的章节标题，标题里的页码不是引用编号
    ("章节标题 [4.4. 战斗阶段流程 141] 不算引用",
     "根据参考资料 [1] 和 [4.4. 战斗阶段流程 141] 判断。", 3, (1,), (), (2, 3)),
    ("页码引用 [第 2 页] 不算引用",
     "见 [第 2 页] 的说明。", 3, (), (), (1, 2, 3)),
    ("真引用与假引用混在同一句",
     "见 [2] 与 [4.4. 战斗阶段流程 141]。", 3, (2,), (), (1, 3)),
]


def main() -> int:
    failures = 0
    for name, answer, count, cited, invalid, unused in CASES:
        report = rag_core.check_citations(answer, count)
        problems = []
        if report.cited != cited:
            problems.append("cited=%s，期望 %s" % (report.cited, cited))
        if report.invalid != invalid:
            problems.append("invalid=%s，期望 %s" % (report.invalid, invalid))
        if report.unused != unused:
            problems.append("unused=%s，期望 %s" % (report.unused, unused))
        if report.ok != (not invalid):
            problems.append("ok=%s 与 invalid=%s 不一致" % (report.ok, report.invalid))
        if report.has_citation != bool(cited):
            problems.append("has_citation=%s 与 cited=%s 不一致" % (report.has_citation, report.cited))
        if invalid and "不存在" not in report.warning():
            problems.append("有越界编号但 warning 未说明")
        if not invalid and invalid == () and cited and "不存在" in report.warning():
            problems.append("无越界编号却出现越界警告")
        if problems:
            failures += 1
            print("✗ %s" % name)
            for problem in problems:
                print("      %s" % problem)
        else:
            print("✓ %s" % name)

    # 有效/越界集合必须互不重叠且覆盖 cited
    extra = 0
    for name, answer, count, _c, _i, _u in CASES:
        report = rag_core.check_citations(answer, count)
        if set(report.valid) | set(report.invalid) != set(report.cited):
            print("✗ %s：valid ∪ invalid ≠ cited" % name)
            extra += 1
        if set(report.valid) & set(report.invalid):
            print("✗ %s：valid ∩ invalid 非空" % name)
            extra += 1

    total = len(CASES)
    print("\n%d/%d 通过%s" % (total - failures, total, "" if not extra else "（另有 %d 项不一致）" % extra))
    return 1 if (failures or extra) else 0


if __name__ == "__main__":
    raise SystemExit(main())
