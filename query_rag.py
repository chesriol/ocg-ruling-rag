# -*- coding: utf-8 -*-
"""游戏王 OCG 规则助手（命令行）。

检索、证据格式与提示词全部来自 `rag_core.py`，与 `eval/run_eval.py` 共用同一份实现 ——
这样"评测里测的"就是"命令行里跑的"，两边不会悄悄跑偏。

用法：
  python query_rag.py
  python query_rag.py --no-rerank           # 关闭精排，对比速度
  python query_rag.py --mode vector_deep    # 指定任意检索配置
"""
from __future__ import annotations

import argparse
import sys
import time

from langchain_ollama import OllamaLLM

import rag_core


def answer(retriever, llm, question, mode):
    result = retriever.retrieve(question, mode)
    if not result.documents:
        print("没有检索到相关资料。")
        return result, 0.0
    prompt = rag_core.build_qa_prompt(question, result.documents)
    started = time.perf_counter()
    for piece in llm.stream(prompt):           # 流式输出，避免长时间"思考中..."
        sys.stdout.write(piece)
        sys.stdout.flush()
    return result, (time.perf_counter() - started) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true", help="关闭精排，用于对比")
    parser.add_argument("--mode", choices=rag_core.MODES, default=None,
                        help="检索配置（默认 vector_rerank；加 --no-rerank 时为 vector）")
    args = parser.parse_args()

    mode = args.mode or ("vector" if args.no_rerank else "vector_rerank")
    retriever = rag_core.Retriever(
        need_reranker=(mode == "vector_rerank"),
        need_bm25=(mode == "hybrid"),
    )
    llm = OllamaLLM(model=rag_core.LLM_MODEL, temperature=0)

    print("OCG 规则助手已启动（检索：%s；精排设备：%s；输入 q 退出）"
          % (rag_core.MODE_LABEL.get(mode, mode), retriever.device))

    while True:
        try:
            question = input("\n你问：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() == "q":
            break
        print("AI答：", end="")
        result, generation_ms = answer(retriever, llm, question, mode)
        if result.documents:
            print("\n\n依据：")
            for order, document in enumerate(result.documents, start=1):
                print("  [%d] 规则书 %s" % (order, rag_core.page_label_of(document)))
        print("耗时：检索 %.0f ms / 生成 %.0f ms" % (result.elapsed_ms, generation_ms))


if __name__ == "__main__":
    main()
