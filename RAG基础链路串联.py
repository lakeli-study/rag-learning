from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
import os
from langchain.chains import RetrievalQA
# ========== 0. 设置智谱 API Key ==========
# 读取 .env 配置文件
load_dotenv() 
# 从系统环境变量中获取密钥
api_key = os.environ.get("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("ZHIPUAI_API_KEY 环境变量未设置，请在 .env 文件中配置")



# ========== 1. 加载文档 ==========
loader = PyPDFLoader("knowledge.pdf")
documents = loader.load()
print(f"加载了 {len(documents)} 份文档")

# ========== 2. 文本切分 ==========
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=60
)
chunks = splitter.split_documents(documents)
print(f"切分出 {len(chunks)} 个文本块")

# ========== 3. 向量化并存入向量库 ==========
embeddings = ZhipuAIEmbeddings(model="embedding-3")
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
vectorstore.save_local("./faiss_db")
print("向量库已保存到 ./faiss_db")

# ========== 4. 配置检索器 ==========
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ========== 5. 配置 LLM 和提示词 ==========
prompt_template = """你是一个基于文档回答问题的助手。
请只根据以下上下文回答问题。
如果上下文中没有答案，请回答“未找到相关信息”。

上下文：
{context}

问题：
{question}
"""

prompt = PromptTemplate.from_template(prompt_template)

llm = ChatZhipuAI(model="glm-4-flash", temperature=0)


# ======= 6. 串联 RAG 链路 =======
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
# ========== 7. 测试问答 ==========
def ask(question):
    result = qa_chain.invoke({"query": question})
    print("=" * 50)
    print("问题：", question)
    print("回答：", result["result"])
    print("参考片段：")
    for doc in result["source_documents"]:
        print("-", doc.page_content[:120])
    print("=" * 50)

ask("RAG 是什么？")
ask("RAG 如何减少大模型幻觉？")
ask("chunk_overlap 的作用是什么？")
ask("今天天气怎么样？")