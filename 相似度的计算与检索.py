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
documents=[
    "大模型是人工智能的重要突破",
    "RAG技术解决了大模型的幻觉问题",
    "今天天气真不错，适合去散步。"
]
#把知识库转换为向量矩阵
#相当于建立索引
doc_vectors=np.array(embeddings.embed_documents(documents))
print(f"知识库矩阵形状: {doc_vectors.shape}")

#开始用户提问环节
query_text="怎么解决大模型的幻觉问题？"
print(f"\n用户问题: {query_text}")

#必须用embed_query方法,不要用embed_documents方法，前后结果不同
#query_vector=embeddings.embed_query(query_text)
query_vector=np.array(embeddings.embed_query(query_text))
print(f"用户问题向量形状: {query_vector.shape}")

#计算点积

dot_product=np.dot(query_vector.reshape(1,-1),doc_vectors.T).flatten()
print(f"用户问题向量与知识库向量的点积: {dot_product}")

#计算模长
query_norm=np.linalg.norm(query_vector)
doc_norm=np.linalg.norm(doc_vectors,axis=1)

print(f"用户问题向量的模长: {query_norm}")
print(f"知识库向量的模长: {doc_norm}")

#计算相似度
similarity=dot_product/(query_norm*doc_norm)
print(f"用户问题向量与知识库向量的相似度: {similarity}")

print("\n---检索结果---")
for i,s in enumerate(similarity):
    print(f"文档{i}: {s:.4f}")

best_match_index=np.argmax(similarity)
print(f"\n最佳匹配文档索引: {best_match_index}")
print(f"最佳匹配文档: {documents[best_match_index]}")
print(f"最佳匹配相似度: {similarity[best_match_index]:.4f}")
