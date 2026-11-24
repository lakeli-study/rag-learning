from langchain_core.documents.base import Document


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

loader=PyPDFLoader("knowledge.pdf")
documents=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=60
)

# 13. 切分文档 (稍微改写，确保返回的是列表)
chunks = splitter.split_documents(documents)

# 14. 打印调试，看看切分出来的到底是什么
print(f"切分后的 chunks 类型: {type(chunks)}")
print(f"文档数量: {len(chunks)}")

# 16. 准备空列表
records = []

# 17. 遍历 (这里就是你截图里带提示的 enumerate)
# 注意：如果你用 VS Code，它会自动提示 enumerate 的类型，不用管那个灰色的 [Document]
for i, chunk in enumerate(chunks):
    records.append({
        "id": i,
        "text": chunk.page_content,
        "source": chunk.metadata.get("source", ""),
        "page": chunk.metadata.get("page", None),
    })

# 26. 核心修复：显式强制转换为 DataFrame
# 这样无论 records 是什么状态，df 都会是一个真正的表格对象
df = pd.DataFrame(records)
print(f"转换后的 df 类型: {type(df)}")

# 28. 过滤和保存
df["char_len"] = df["text"].str.len()
df = df[df["char_len"] >= 30]
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]

# 32. 保存 CSV
df.to_csv("chunks.csv", encoding="utf-8", index=False)
print(f"成功保存 CSV，共 {len(df)} 条数据。")