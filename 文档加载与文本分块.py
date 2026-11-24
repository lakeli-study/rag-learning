from langchain_text_splitters import RecursiveCharacterTextSplitter
# 1. 模拟一篇长文档（实际项目中这里会是读取 PDF/TXT 的内容）
raw_document = """
大模型（Large Language Model）是人工智能领域的重要突破。
它拥有数以亿计的参数，能够理解并生成人类语言。

RAG（检索增强生成）技术解决了大模型幻觉的问题。
它通过外挂知识库，让模型在回答前先去查阅资料。

文本分块（Chunking）是 RAG 流程中的第一步。
我们需要把长文章切成小块，以便存入向量数据库。
通常我们会设置一个重叠窗口（Overlap），防止关键信息被切断。
"""

# 2. 初始化分割器
# chunk_size: 每一块的大小（字符数），通常设为 200-500
# chunk_overlap: 重叠窗口大小，保留上一块的尾巴，保证语义连贯
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,      # 为了演示效果，我设得很小，实际建议 300-500
    chunk_overlap=10,   # 重叠 10 个字符
    separators=["\n\n", "\n", " ", ""] # 切割优先级：先按空行，再按换行，再按空格
)

# 3. 执行切割
chunks = text_splitter.split_text(raw_document)

# 4. 打印结果（验收环节）
print(f"原文档长度: {len(raw_document)} 字符")
print(f"切分后块数: {len(chunks)} 块")
print("-" * 30)

for i, chunk in enumerate(chunks):
    print(f"[第 {i+1} 块] (长度: {len(chunk)}): {chunk}")