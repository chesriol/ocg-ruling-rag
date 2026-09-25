# -*- coding: utf-8 -*-
"""从《游戏王 OCG 大师规则》PDF 第 6 章解析出官方规则检定测试题，产出评测用 goldens。

为什么这么做：
  PDF 第 6 章自带 2018/2019/2020/2021 四套 KONAMI 官方规则检定测试（单选、含标准答案），
  以及 2017/2020 两套非官方规则测试。这是现成的、带 ground truth 的评测集，
  不需要自己出题，也不需要用 LLM 当裁判。

用法：
  python eval/parse_goldens.py
输出：
  eval/goldens.json   题目 + 选项 + 答案键 + 可评测标记
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "data" / "ocg-rule-readthedocs-io-zh-cn-latest.pdf"
OUT_PATH = Path(__file__).resolve().parent / "goldens.json"

# 第 6 章的 6 个测试集；official 表示是否为 KONAMI 官方检定测试
# set_key 用于生成稳定的题目 id（6.6 内含单选/多选/判断三个小节，题号会重复）
SETS = [
    ("6.1", "2018年游戏王OCG规则检定测试", True, "ocg2018"),
    ("6.2", "2019年游戏王OCG规则检定测试", True, "ocg2019"),
    ("6.3", "2020年游戏王OCG规则检定测试", True, "ocg2020"),
    ("6.4", "2021年游戏王OCG规则检定测试", True, "ocg2021"),
    ("6.5", "规则测试2017", False, "rt2017"),
    ("6.6", "规则测试2020", False, "rt2020"),
]
SECTION_CODE = {"single": "s", "multi": "m", "judge": "j"}

# 页眉 / 页脚（例："ocg-rule Documentation"、"6.1. 2018 年游戏王 OCG 规则检定测试 303"）
RE_HEADER = re.compile(r"^\s*ocg-rule Documentation\s*$")
RE_FOOTER = re.compile(r"^\s*6\.\d\.\s*\S.*\s\d{1,3}\s*$")
# 题目行：行首数字 + 点
RE_QUESTION = re.compile(r"^\s*(\d{1,3})\s*[.、]\s*(.*)$")
# 选项行：行首 A-G + 点/顿号
RE_OPTION = re.compile(r"^\s*([A-G])\s*[.、]\s*(.*)$")
# 答案键：区间式 "1-5ADCBD" / "1~5CBBDC" / "1-10BBECAEAACB"
RE_ANSWER_RANGE = re.compile(r"^\s*(\d{1,3})\s*[-~～]\s*(\d{1,3})\s*([A-E]{1,10})\s*$")
# 答案键：逐题式 "2. D（...）"
RE_ANSWER_LINE = re.compile(r"^\s*(\d{1,3})\s*[.、]\s*(.+)$")
# 需要看图才能作答的题目
IMAGE_REF_MARKERS = ("如下图", "下图", "图中", "如图", "上表", "下表", "表中", "图1", "图 1", "图示")


def load_chapter_six_text(pdf_path: Path):
    """抽取 PDF 全文，清洗页眉页脚，返回 (全文, 每页起始偏移)。"""
    reader = PdfReader(str(pdf_path))
    try:
        labels = list(reader.page_labels)
    except Exception:
        labels = [str(i + 1) for i in range(len(reader.pages))]

    pieces, page_offsets, cursor = [], [], 0
    for index, page in enumerate(reader.pages):
        label = labels[index] if index < len(labels) else str(index + 1)
        raw = page.extract_text() or ""
        kept = []
        for line in raw.splitlines():
            if RE_HEADER.match(line) or RE_FOOTER.match(line):
                continue
            kept.append(line.rstrip())
        text = "\n".join(kept).strip()
        if not text:
            continue
        page_offsets.append((label, cursor))
        pieces.append(text)
        cursor += len(text) + 1
    return "\n".join(pieces), page_offsets


def page_of(offset: int, page_offsets) -> str:
    label = page_offsets[0][0] if page_offsets else ""
    for entry_label, start in page_offsets:
        if start <= offset:
            label = entry_label
        else:
            break
    return label


def squash(value: str) -> str:
    """去掉所有空白，用于抵消 PDF 抽取时插入的空格。"""
    return re.sub(r"\s+", "", value)


def split_sets(full_text: str):
    """按 6.1 ~ 6.6 切分成 6 个测试集文本。

    PDF 抽出的标题里会插入不规则空格（"2018 年游戏王 OCG 规则检定测试"），
    所以按"去掉空白后逐行比对"来定位标题，比正则可靠。
    """
    lines = full_text.splitlines(keepends=True)
    offsets, cursor = [], 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    wanted = {squash("%s %s" % (set_id, name)): (set_id, name, official, set_key)
              for set_id, name, official, set_key in SETS}
    marks = []
    for index, line in enumerate(lines):
        key = squash(line)
        if key in wanted:
            set_id, name, official, set_key = wanted[key]
            marks.append((offsets[index], set_id, name, official, set_key))
    marks.sort()

    found = {mark[1] for mark in marks}
    for set_id, name, _official, _set_key in SETS:
        if set_id not in found:
            print("  ! 未找到章节标题：%s %s" % (set_id, name), file=sys.stderr)

    blocks = []
    for position, (start, set_id, name, official, set_key) in enumerate(marks):
        end = marks[position + 1][0] if position + 1 < len(marks) else len(full_text)
        blocks.append((set_id, name, official, set_key, start, full_text[start:end]))
    return blocks


def strip_part_headers(text: str) -> str:
    return re.sub(r"^\s*6\.\d\.\d\s*(正文|基本部分|应用部分|答案)\s*$", "", text, flags=re.M)


def parse_questions(block: str, set_id: str, official: bool, page_offsets, base_offset: int):
    """解析正文区的题目与选项。"""
    body = block
    # 截掉答案区，正文与答案分开解析
    answer_split = re.search(r"^\s*6\.\d\.\d\s*答案\s*$", body, re.M)
    body_only = body[:answer_split.start()] if answer_split else body
    answer_only = body[answer_split.end():] if answer_split else ""

    body_only = re.sub(r"^\s*6\.\d\.\d\s*(正文|基本部分|应用部分)\s*$", "", body_only, flags=re.M)

    SECTION_OF_HEADER = {"单选题": "single", "多选题": "multi", "判断题": "judge"}
    questions, current, expected = [], None, 1
    part, section = "", "single"
    for raw_line in body_only.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in SECTION_OF_HEADER:
            # 小节切换（如 6.6 的 单选/多选/判断），题号在小节内重新从 1 开始
            section = SECTION_OF_HEADER[line]
            expected = 1
            continue
        if re.fullmatch(r"基本部分|应用部分", line):
            part = line
            continue

        question_match = RE_QUESTION.match(line)
        option_match = RE_OPTION.match(line)

        if question_match and not option_match:
            number = int(question_match.group(1))
            starts_new_group = number == 1 and current is not None
            if starts_new_group:
                expected = 1
            if number == expected or (current is None and number == 1):
                if current:
                    questions.append(current)
                current = {
                    "set_id": set_id,
                    "official": official,
                    "part": part,
                    "section": section,
                    "number": number,
                    "stem_lines": [question_match.group(2)],
                    "options": {},
                    "_last_option": None,
                    "_offset": base_offset + body_only.find(line),
                }
                expected = number + 1
                continue

        if current is None:
            continue
        if option_match:
            key = option_match.group(1)
            current["options"][key] = option_match.group(2).strip()
            current["_last_option"] = key
        elif current["_last_option"]:
            current["options"][current["_last_option"]] += line
        else:
            current["stem_lines"].append(line)

    if current:
        questions.append(current)
    return questions, answer_only


def parse_answer_key(answer_text: str, set_id: str):
    """解析答案区，返回 {题号: 原始答案文本}。"""
    answers, section = {}, "single"
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in ("单选题", "多选题", "判断题"):
            section = {"单选题": "single", "多选题": "multi", "判断题": "judge"}[line]
            continue
        range_match = RE_ANSWER_RANGE.match(line)
        if range_match:
            start, end, letters = int(range_match.group(1)), int(range_match.group(2)), range_match.group(3)
            for offset, letter in enumerate(letters):
                key = "%s:%d" % (section, start + offset)
                answers[key] = letter
            continue
        if section == "judge" and re.fullmatch(r"\d{1,3}\s*[-~～]\s*\d{1,3}\s*[√×✓✗]+", line):
            numbers = re.match(r"(\d{1,3})\s*[-~～]\s*(\d{1,3})", line)
            marks = re.search(r"([√×✓✗]+)$", line).group(1)
            for offset, mark in enumerate(marks):
                answers["judge:%d" % (int(numbers.group(1)) + offset)] = mark
            continue
        if section == "multi":
            multi_match = re.fullmatch(r"(\d{1,3})\s*([A-G]{1,5})", line)
            if multi_match:
                answers["multi:%d" % int(multi_match.group(1))] = multi_match.group(2)
                continue
        line_match = RE_ANSWER_LINE.match(line)
        if line_match:
            answers["single:%d" % int(line_match.group(1))] = line_match.group(2).strip()
    return answers


def finalize_answer(raw: str):
    """从答案文本里抽出最终答案（处理『裁定变更』标注）。"""
    if raw is None:
        return None, False, False
    text = str(raw)
    # "…结果答案为A" / "…最终答案为A"
    explicit = re.search(r"(?:结果答案|最终答案|答案为)\s*为?\s*([A-G]{1,5})", text)
    ambiguous = False
    if explicit:
        return explicit.group(1), False, False
    if "裁定变更" in text or "变更" in text:
        ambiguous = True
    stripped = re.sub(r"[（(][^）)]*[）)]", "", text).strip()
    letters = re.findall(r"[A-G]", stripped)
    if not letters:
        return None, True, ambiguous
    if len(letters) == 1:
        return letters[0], False, ambiguous
    return "".join(letters), (len(letters) > 1), True


def build():
    if not PDF_PATH.exists():
        raise SystemExit("找不到 PDF：%s" % PDF_PATH)
    print("读取 PDF：%s" % PDF_PATH.name)
    full_text, page_offsets = load_chapter_six_text(PDF_PATH)

    chapter_six = full_text.find("CHAPTER 6")
    if chapter_six < 0:
        chapter_six = 0
    print("第 6 章起于全文偏移 %d" % chapter_six)

    records, summary = [], {}
    for set_id, name, official, set_key, start, block in split_sets(full_text):
        questions, answer_text = parse_questions(
            block, set_id, official, page_offsets, base_offset=start)
        answers = parse_answer_key(answer_text, set_id)
        # 本节是否含多个小节（6.6 有单选/多选/判断），有则 id 里必须带上小节码
        sections_in_set = {q["section"] for q in questions}
        needs_section_code = len(sections_in_set) > 1

        matched = 0
        for question in questions:
            # 先按题目所属小节取答案，取不到再跨小节兜底（6.5 的答案区没有小节标题）
            key = "%s:%d" % (question["section"], question["number"])
            if key not in answers:
                for candidate in ("single", "multi", "judge"):
                    if "%s:%d" % (candidate, question["number"]) in answers:
                        key = "%s:%d" % (candidate, question["number"])
                        break
            raw = answers.get(key)
            if raw is not None:
                matched += 1
            answer, multi, ambiguous = finalize_answer(raw)

            stem = "".join(question["stem_lines"]).strip()
            options = {k: v.strip() for k, v in question["options"].items()}
            option_keys = sorted(options)
            has_image = any(marker in stem for marker in IMAGE_REF_MARKERS)

            if answer is None:
                kind = "unsupported"
            elif question["section"] != "single" or multi or len(str(answer)) > 1:
                kind = "multi_choice"
            elif option_keys and str(answer) not in option_keys:
                kind = "answer_not_in_options"
            elif not options:
                kind = "short_answer"
            else:
                kind = "single_choice"

            gradeable = (
                kind == "single_choice"
                and not has_image
                and not ambiguous
                and answer in option_keys
                and len(options) >= 2
            )

            records.append({
                "id": "%s-%s%02d" % (
                    set_key,
                    SECTION_CODE[question["section"]] + "-" if needs_section_code else "q",
                    question["number"],
                ),
                "set_id": set_id,
                "set_key": set_key,
                "set_name": name,
                "official": official,
                "section": question["section"],
                "part": question["part"] or ("基本部分" if question["number"] <= 28 else "应用部分"),
                "number": question["number"],
                "type": kind,
                "stem": stem,
                "options": options,
                "answer": answer,
                "answer_raw": raw,
                "answer_ambiguous": ambiguous,
                "has_image_ref": has_image,
                "gradeable": gradeable,
                "source_page_label": page_of(question["_offset"], page_offsets),
            })
        summary[set_id] = {
            "name": name,
            "official": official,
            "questions": len(questions),
            "answers": len(answers),
            "matched": matched,
        }
        print("  %s %-28s 题目 %3d / 答案 %3d / 匹配 %3d"
              % (set_id, name, len(questions), len(answers), matched))

    payload = {
        "schema_version": 1,
        "source": {
            "pdf": PDF_PATH.name,
            "chapter": "CHAPTER 6 OCG 规则测试",
            "site": "https://ocg-rule.readthedocs.io/zh-cn/latest/",
        },
        "summary": summary,
        "stats": {
            "total": len(records),
            "gradeable": sum(1 for r in records if r["gradeable"]),
            "official_gradeable": sum(1 for r in records if r["gradeable"] and r["official"]),
            "image_dependent": sum(1 for r in records if r["has_image_ref"]),
            "by_type": {t: sum(1 for r in records if r["type"] == t)
                        for t in sorted({r["type"] for r in records})},
        },
        "questions": records,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n统计：%s" % json.dumps(payload["stats"], ensure_ascii=False))
    print("已写出：%s" % OUT_PATH)


if __name__ == "__main__":
    build()
