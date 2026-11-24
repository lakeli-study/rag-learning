# RAG 入门

本仓库整理了 RAG（检索增强生成）入门学习过程中的示例代码，包含文档加载与分块、向量化、相似度检索、FAISS 向量库构建、链路串联以及简单的评估脚本。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `文档加载与文本分块.py` | 使用 `RecursiveCharacterTextSplitter` 演示文本切分与重叠窗口 |
| `pandas管理文档原数据.py` | 将切分后的文档整理为 DataFrame 并保存为 `chunks.csv` |
| `embedding调用.py` | 调用智谱 Embedding 模型，将文本转换为向量 |
| `相似度的计算与检索.py` | 计算查询向量与文档向量的余弦相似度并检索 |
| `RAG基础链路串联.py` | 完整 RAG 链路：加载 PDF → 分块 → 向量化入 FAISS → 检索问答 |
| `evaluate.py` | 简单的评估 / 回归测试框架示例 |
| `knowledge.pdf` | 示例知识库文档 |
| `chunks.csv` | 分块后的文档元数据示例 |
| `fail_cases.json` | 评估失败案例示例 |
| `faiss_db/` | 已构建的 FAISS 向量库索引 |

## 环境配置

1. 安装依赖：

   ```bash
   pip install python-dotenv langchain-community langchain-text-splitters langchain-core langchain pandas numpy faiss-cpu pypdf
   ```

2. 配置 API 密钥：复制 `.env.example` 为 `.env`，填入你自己的智谱 AI API Key。

   ```bash
   cp .env.example .env
   ```

   `.env` 已被 `.gitignore` 忽略，**请勿将包含真实密钥的 `.env` 提交到仓库**。

   > 注意：原代码中变量名不统一 —— `RAG基础链路串联.py` 读取 `ZHIPUAI_API_KEY`，而 `embedding调用.py` 和 `相似度的计算与检索.py` 读取 `ZHIPU_API_KEY`。`.env.example` 中已同时列出，按需填写即可。

## 运行示例

```bash
python 文档加载与文本分块.py
python RAG基础链路串联.py
```
