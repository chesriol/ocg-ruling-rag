# -*- coding: utf-8 -*-
"""构建检索索引（幂等 + 分角色 + 自描述）。

相对上一版修掉的四个问题：
  1. **不再把考卷向量化**。规则书第 6 章是 6 套规则检定测试（含标准答案）。
     把它们索引进去，等于给模型发答案；用它做评测则是数据泄漏。
     这里在入库阶段就整体剔除 —— 不依赖"检索时记得加过滤"这种人为纪律。
  2. **构建幂等**。旧版注释写"直接覆盖旧库"，实际 Chroma.from_documents 是追加，
     重跑一次索引就翻倍。这里先删同一 collection 再写入，并用内容哈希做确定性 id。
  3. **目录页噪声被剔除**。旧库里 27 条 ". . . ." 目录点线段落会被检索命中。
  4. **写出 manifest**。角色分布、来源哈希、剔除原因都可查，构建结果自证。

用法：
  python build_vector_store.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "data" / "ocg-rule-readthedocs-io-zh-cn-latest.pdf"
NOTE_PATH = ROOT / "data" / "yugioh_knowledge.txt"
QA_PATH = ROOT / "data" / "official-qa.jsonl"
GOLDENS_PATH = ROOT / "eval" / "goldens.json"
DB_DIR = ROOT / "chroma_db"
MANIFEST_PATH = DB_DIR / "manifest.json"

COLLECTION = "ocg-rule"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# 注意：collection 名从旧的默认值 "langchain" 改成 "ocg-rule"，
# 读取端（query_rag.py / eval/run_eval.py）必须传同一个名字。

# 只保留真正可作规则依据的角色；其余在建库阶段丢弃
KEPT_ROLES = {"rule-doc", "rule-note", "official-qa"}
ROMAN_PAGE_LABELS = {"i", "ii", "iii", "iv", "v"}

# 考卷起始页；优先从 eval/goldens.json 推导，避免两处硬编码不一致
DEFAULT_TEST_PAGE_MIN = 294


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chapter_six_start_page() -> int:
    """从评测集反推第 6 章（考卷）的起始页。"""
    if not GOLDENS_PATH.exists():
        return DEFAULT_TEST_PAGE_MIN
    try:
        payload = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))
        pages = [int(q["source_page_label"]) for q in payload["questions"]
                 if str(q.get("source_page_label", "")).isdigit()]
        return min(pages) if pages else DEFAULT_TEST_PAGE_MIN
    except Exception:
        return DEFAULT_TEST_PAGE_MIN


def classify(text: str, page_label: str, test_page_min: int) -> str:
    """给片段定角色：目录 / 站点信息 / 考卷 / 规则正文。"""
    if text.count(". . .") > 3:
        return "toc"
    if page_label in ROMAN_PAGE_LABELS:
        return "site-info"
    if page_label.isdigit() and int(page_label) >= test_page_min:
        return "rule-test"
    return "rule-doc"


def clean_metadata(metadata: dict) -> dict:
    """Chroma 只接受 str/int/float/bool，且不接受 None。"""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool) or isinstance(value, (str, int, float)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def chunk_id(source: str, page_label: str, index: int, text: str) -> str:
    """确定性 id：同一份内容重复构建得到同一个 id，可安全 upsert。"""
    digest = hashlib.sha1(("%s|%s|%d|%s" % (source, page_label, index, text)).encode("utf-8"))
    return digest.hexdigest()


def splitter() -> RecursiveCharacterTextSplitter:
    # 规则书逻辑连贯，切太碎会断章取义：chunk 1000 / overlap 200，优先按段落与中文标点切
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )


# PDF 每页都带页眉「ocg-rule Documentation」和页脚「<章节号> <标题> <页码>」。
# 不清掉有两个实际害处：样板文字会被编码进向量，而且模型会把章节标题当正文引用
# （实测它照抄过「[4.4. 战斗阶段流程 144]」）。
RE_PAGE_HEADER = re.compile(r"^[ \t]*ocg-rule Documentation[ \t]*$", re.M)
RE_PAGE_FOOTER = re.compile(r"^[ \t]*\d+(?:\.\d+)*\.?[ \t]+\S.*?[ \t](\d{1,3})[ \t]*$", re.M)
RE_BLANK_RUN = re.compile(r"\n{3,}")


def clean_page_text(text: str, page_label: str):
    """去掉本页的页眉与页脚，返回 (清洗后文本, 删掉页眉数, 删掉页脚数)。

    关键：页脚**只在行尾数字恰好等于本页 page_label 时才删**。
    实测 349 页共 343 处页脚，其中 337 处行尾数字与本页 page_label 完全相等，
    其余 6 处在罗马数字前言页（那几页整体不会入索引）。用页码做精确匹配，
    就不会误删"以数字开头、以数字结尾"的正文行 —— 这是纯正则做不到的。
    """
    header_count = [0]
    footer_count = [0]

    def drop_header(_match):
        header_count[0] += 1
        return ""

    text = RE_PAGE_HEADER.sub(drop_header, text)

    if page_label.isdigit():
        def drop_footer(match):
            if match.group(1) != page_label:
                return match.group(0)
            footer_count[0] += 1
            return ""
        text = RE_PAGE_FOOTER.sub(drop_footer, text)

    return RE_BLANK_RUN.sub("\n\n", text).strip(), header_count[0], footer_count[0]


def qa_splitter() -> RecursiveCharacterTextSplitter:
    """官方 Q&A 用更大的块：一条问答通常 800 字符左右，尽量整条保留。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )


def load_qa_documents():
    """读取官方 Q&A 语料（fetch_qa.py 产出）。

    这些记录不适合走 PDF 那套切片策略：它们是"一条一问一答"的独立单元，
    切碎会破坏问答对应关系，所以用更大的块长单独处理。
    """
    if not QA_PATH.exists():
        print("未找到官方 Q&A 语料（可运行 python fetch_qa.py 生成）：%s" % QA_PATH.name)
        return []
    records = []
    with QA_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            if not question and not answer:
                continue
            records.append(Document(
                page_content="%s\n%s" % (question, answer),
                metadata={
                    "role": "official-qa",
                    "qa_id": str(item.get("qa_id", "")),
                    "source": QA_PATH.name,
                    "source_url": item.get("source", ""),
                    "cards": "、".join(item.get("cards") or [])[:200],
                }))
    print("读取官方 Q&A：%d 条（本地化：日文正文 + 中文卡名）" % len(records))
    return records


def load_chunks():
    if not PDF_PATH.exists():
        raise SystemExit("找不到规则书 PDF：%s" % PDF_PATH)

    pages = PyPDFLoader(str(PDF_PATH)).load()
    print("读取 PDF：%s（%d 页）" % (PDF_PATH.name, len(pages)))

    stats = {"headers_removed": 0, "footers_removed": 0}
    documents = []
    for page in pages:
        label = str(page.metadata.get("page_label") or "")
        cleaned, headers, footers = clean_page_text(page.page_content, label)
        stats["headers_removed"] += headers
        stats["footers_removed"] += footers
        documents.append(Document(page_content=cleaned, metadata=dict(page.metadata)))
    print("清洗页眉 %d 处、页脚 %d 处" % (stats["headers_removed"], stats["footers_removed"]))

    chunks = splitter().split_documents(documents)
    if NOTE_PATH.exists():
        chunks += splitter().split_documents(TextLoader(str(NOTE_PATH), encoding="utf-8").load())
        print("读取补充笔记：%s" % NOTE_PATH.name)
    return chunks, stats


def build():
    test_page_min = chapter_six_start_page()
    chunks, cleaning = load_chunks()

    kept, dropped = [], {"toc": 0, "site-info": 0, "rule-test": 0}
    for chunk in chunks:
        metadata = chunk.metadata or {}
        source = Path(str(metadata.get("source", "unknown"))).name
        page_label = str(metadata.get("page_label", "") or "")
        role = ("rule-note" if source == NOTE_PATH.name
                else classify(chunk.page_content, page_label, test_page_min))
        if role not in KEPT_ROLES:
            dropped[role] = dropped.get(role, 0) + 1
            continue
        kept.append((chunk, source, page_label, role))

    # 官方 Q&A 以整条为单位入库（角色固定，不经 PDF 的角色判定）
    qa_documents = load_qa_documents()
    qa_chunks = qa_splitter().split_documents(qa_documents) if qa_documents else []
    for chunk in qa_chunks:
        kept.append((chunk, QA_PATH.name, "", "official-qa"))

    print("切分片段 %d → 入库 %d，剔除 %s" % (len(chunks), len(kept), dropped))
    if not kept:
        raise SystemExit("没有可入库的片段，请检查数据与角色规则")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if DB_DIR.exists():
        # 整个索引目录重建，而不是往已有 collection 里追加。
        # 旧索引由 from_documents 追加式构建，且混入了考卷与目录页，无法就地修正。
        shutil.rmtree(DB_DIR)
        print("已删除旧索引目录（重建而非追加）")

    store = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings,
                   collection_name=COLLECTION)

    texts, metadatas, ids, role_counts = [], [], [], {}
    for index, (chunk, source, page_label, role) in enumerate(kept):
        metadata = clean_metadata(chunk.metadata or {})
        metadata.update({"role": role, "source_file": source, "page_label": page_label})
        identifier = chunk_id(source, page_label, index, chunk.page_content)
        metadata["chunk_id"] = identifier
        texts.append(chunk.page_content)
        metadatas.append(metadata)
        ids.append(identifier)
        role_counts[role] = role_counts.get(role, 0) + 1

    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": COLLECTION,
        "embedding_model": EMBED_MODEL,
        "chunking": {"chunk_size": 1000, "chunk_overlap": 200,
                     "official_qa": {"chunk_size": 1200, "chunk_overlap": 150}},
        "cleaning": {
            **cleaning,
            "note": "页眉固定为「ocg-rule Documentation」；页脚形如「4.4. 战斗阶段流程 144」，"
                    "仅在行尾数字等于本页 page_label 时删除，避免误删正文。",
        },
        "sources": {
            PDF_PATH.name: {"sha256": sha256_file(PDF_PATH), "bytes": PDF_PATH.stat().st_size},
        },
        "role_policy": {
            "kept": sorted(KEPT_ROLES),
            "dropped": dropped,
            "test_page_min": test_page_min,
            "note": "第 6 章为规则检定测试（含答案键），入库即构成答案泄漏，故整体剔除；"
                    "目录页与前言页同样不作为规则依据。",
        },
        "stored_chunks": len(texts),
        "role_counts": role_counts,
    }
    if NOTE_PATH.exists():
        manifest["sources"][NOTE_PATH.name] = {
            "sha256": sha256_file(NOTE_PATH), "bytes": NOTE_PATH.stat().st_size}
    if QA_PATH.exists():
        manifest["sources"][QA_PATH.name] = {
            "sha256": sha256_file(QA_PATH), "bytes": QA_PATH.stat().st_size}
        manifest["official_qa"] = {
            "records": len(qa_documents),
            "chunks": len(qa_chunks),
            "language": "日文正文 + 中文卡名（官方库无中文 Q&A，实测确认）",
            "sampling_note": "由 fetch_qa.py 对卡表等距抽样取得，规则与评测集无关",
        }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("入库完成：%s" % json.dumps(role_counts, ensure_ascii=False))
    print("manifest：%s" % MANIFEST_PATH)
    print("索引目录：%s" % DB_DIR)


if __name__ == "__main__":
    build()
