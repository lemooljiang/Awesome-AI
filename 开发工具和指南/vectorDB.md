## Vector Database

向量数据库是一种用于存储和处理向量数据的数据库系统。向量数据是由一系列数值组成的数据，通常用于表示特征或属性。向量数据库的设计目标是能够高效地处理和查询大规模的向量数据集。近年来，由于人工智能在解决涉及自然语言、图像识别和其他非结构化数据形式的用例方面发挥着越来越大的作用，使用嵌入式技术将非结构化数据（文本、音频、视频等）编码为向量供机器学习模型使用的情况呈爆炸式增长。向量数据库已成为企业交付和扩展这些用例的有效解决方案。

目前市场上有几个代表性的向量数据库，如下：
- [Chroma](#chroma)
- [Faiss](#faiss)
- [Qdrant](#qdrant)
- [Milvus](#milvus)
- [Pinecone](#pinecone)
- [DeepsetAI](#deepsetai)
- [pgvector](#pgvector)
- [Supabase](#supabase)
- [Vespa](#vespa)
- [Weaviate](#weaviate)
- [HNSWLib](#hnswlib)


## Chroma 
Chroma是一个新的AI原生开源嵌入式数据库，非常轻量和易用。Chroma是开源嵌入式数据库，它使知识、事实和技能可插入，从而轻松构建LLM应用程序。它可以运行在内存中（可保存在磁盘中），也可做为数据库服务器来使用（这和传统数据库类似）。

[文档 ｜](https://docs.trychroma.com/?lang=py)
[github |](https://github.com/chroma-core/chroma)
```py
pip install chromadb

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

from langchain.document_loaders import TextLoader
loader = TextLoader('../../../state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
 
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)
 
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

## Faiss
Facebook AI 相似度搜索是一种用于稠密向量的高效相似度搜索和聚类的库。它包含了能够搜索任意大小的向量集合的算法，甚至包括可能不适合内存的向量集合。它还包含用于评估和参数调整的支持代码。

[github](https://github.com/facebookresearch/faiss)
```py
pip install faiss-cpu

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
 
from langchain.document_loaders import TextLoader
loader = TextLoader('../../../state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
 
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
 
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
 
```

## Qdrant 
Qdrant是一个开源向量数据库，具有扩展过滤支持的向量相似度引擎。它可以在内存中使用，也可使用 [云版本](https://cloud.qdrant.io)。

Qdrant 完全用 Rust 语言开发，实现了动态查询计划和有效负载数据索引。向量负载支持多种数据类型和查询条件，包括字符串匹配、数值范围、地理位置等。有效负载过滤条件允许您构建几乎任何应该在相似性匹配之上工作的自定义业务逻辑。

[官网 ｜](https://qdrant.tech)
```py
pip install qdrant-client

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
 
loader = TextLoader('../../../state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
 
embeddings = OpenAIEmbeddings()

# 内存中
qdrant = Qdrant.from_documents(
    docs, embeddings, 
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
)

# 磁盘存储
qdrant = Qdrant.from_documents(
    docs, embeddings, 
    path="/tmp/local_qdrant",
    collection_name="my_documents",
)

query = "What did the president say about Ketanji Brown Jackson"
found_docs = qdrant.similarity_search(query)
print(found_docs[0].page_content) 
```

## Milvus
Milvus是一个存储、索引和管理由深度神经网络和其他机器学习（ML)模型生成的大规模嵌入向量的数据库。它是开源的，旨在提供高效的向量相似度搜索和分析。它支持多种数据类型和查询方式，并提供了丰富的 API 和 SDK。

[官网 ｜](https://milvus.io/)
[云版本 ｜](https://zilliz.com/what-is-milvus)
```py
pip install pymilvus

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import TextLoader
 
from langchain.document_loaders import TextLoader
loader = TextLoader('../../../state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
 
embeddings = OpenAIEmbeddings()
vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
)
docs = vector_db.similarity_search(query)
docs[0]
```


## Pinecone 
Pinecone使建立高性能的矢量搜索应用变得容易。它是一个可管理的、云原生的向量数据库，具有简单的API，没有基础设施方面的麻烦。

[官网 ｜](https://www.pinecone.io/)
[文档 |](https://docs.pinecone.io/docs/overview)

## DeepsetAI
Deepest is not a vector database itself but a complete semantic search pipeline in one solution. You can plug in models and other vector databases in it.

Has open source as well as a managed cloud version

## pgvector
pgvector is an open-source library that can turn your Postgres DB into a vector database.

## Supabase
Supabase is a managed Postgresql solution that implements storing embeddings using the pgvector extension.
[官网](https://supabase.com/)

## Vespa
Vespa is a product from Yahoo. It’s available both as Open Source Download and as a managed Cloud solution.

## Weaviate
类 Graphql接口支持的表达查询语法。这允许您对丰富的实体数据运行探索性数据科学查询。该产品最重要的元素是向量搜索、对象存储和用于布尔关键字搜索的倒排索引的组合，以避免存储与对象/倒排索引分开的向量数据的不同数据库之间的数据漂移和延迟。Wow-effect：有一个令人印象深刻的问答组件——它可以带来一个令人惊叹的元素来演示作为现有或新产品的一部分的新搜索功能。

[官网](https://weaviate.io/)

## HNSWLib
Only available on Node.js.HNSWLib is an in-memory vectorstore that can be saved to a file. It uses HNSWLib.
