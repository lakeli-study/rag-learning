import numpy as np
from dotenv import load_dotenv
import os

from langchain_community.embeddings import ZhipuAIEmbeddings

load_dotenv()

#初始化embedding模型
embeddings = ZhipuAIEmbeddings(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    model="embedding-3"
)

chunks=[
"大模型是人工智能的重要突破",
"RAG技术解决了大模型的幻觉问题",
"今天天气真不错，适合去散步。"
]

print("正在将文本转换为向量...")
vectors=embeddings.embed_documents(chunks)

vac_array=np.array(vectors)

print(f"向量维度: {vac_array.shape[1]}")
print(f"向量数量: {vac_array.shape[0]}")
print(f"成功生产向量！")
print(f"向量数组的形状: {vac_array.shape}")
print(f"第1个向量的5个维度：{vac_array[0][:5]}")






