# 文件：test_qwen.py
# 功能：测试Python能否成功调用你下载好的Qwen2.5模型

from langchain_ollama import OllamaLLM

# 1. 初始化模型（注意：模型名字必须和你在终端ollama list里看到的一模一样）
llm = OllamaLLM(model="qwen2.5:7b")

# 2. 发一条简单的消息
response = llm.invoke("你好，请用一句话介绍一下什么是人工智能。")

# 3. 打印结果
print("模型回答：", response)