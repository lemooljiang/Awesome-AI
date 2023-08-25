# <center> Chroma  </center>

<!-- TOC -->

- [<center> Chroma  </center>](#center-chroma--center)
    - [向量数据库 Chroma](#向量数据库-chroma)
    - [bug](#bug)
    - [基本使用](#基本使用)
    - [Chroma embedding](#chroma-embedding)
    - [Chroma docker](#chroma-docker)
    - [langchain中的使用](#langchain中的使用)
    - [添加文本](#添加文本)
    - [更新和删除数据](#更新和删除数据)
    - [相关度数值检索](#相关度数值检索)
    - [metadata数据过滤](#metadata数据过滤)
    - [直接存入数据库和查询](#直接存入数据库和查询)
    - [httpClient](#httpclient)

<!-- /TOC -->


## 向量数据库 Chroma
[github |](https://github.com/chroma-core/chroma)
[文档 |](https://docs.trychroma.com/)
[chroma使用指南 |](https://zhuanlan.zhihu.com/p/632629938)
[参考 ｜](https://zhuanlan.zhihu.com/p/628187163)
[过滤条件 ｜](https://zhuanlan.zhihu.com/p/640424318)

Chroma是一个新的AI原生开源嵌入式数据库，非常轻量和易用。Chroma是开源嵌入式数据库，它使知识、事实和技能可插入，从而轻松构建LLM应用程序。它可以运行在内存中（可保存在磁盘中），也可做为数据库服务器来使用（这和传统数据库类似）。
```py
pip install chromadb   # 0.4.3  
// pip install chromadb -U 升级
//python3.11版无法安装！

# 预先依赖 
# chromadb有一堆预先的依赖。如果已经安装了langchain，就不用安装别的。否则要先安装torch
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple    //2.0.1  一堆 nvidia-cublas
//pip install torch
```

## bug
```py
1. 使用`delete`函数后，经常会出现如下错误，暂未修复：
Delete of nonexisting embedding ID: 77d25c18-3774-11ee-a7f1-fb75c83274a1
```

## 基本使用
[参考 ](https://docs.trychroma.com/api-reference)
```py
import chromadb

# 创建客户端
# client = chromadb.Client() 内存模式
client = chromadb.PersistentClient(path="./chromac")  # 数据保存在磁盘
# chroma_client = chromadb.HttpClient(host="localhost", port=8000) docker客户端模式

# 遍历集合
client.list_collections()
# 创建新集合
collection = client.create_collection("testname")
# 获取集合
collection = client.get_collection("testname")
# 创建或获取集合
collection = client.get_or_create_collection("testname")
# 删除集合
client.delete_collection("testname")

# 创建或获取集合
collection = client.get_or_create_collection(name="my_collection2")
# collection = client.create_collection(name="my_collection2")
# collection = client.create_collection(name="my_collection", embedding_function=emb_fn)
# collection = client.get_collection(name="my_collection", embedding_function=emb_fn)
# Chroma集合创建时带有一个名称和一个可选的嵌入函数。如果提供了嵌入函数，则每次获取集合时都必须提供。

# 获取集合中最新的5条数据
collection.peek()


# 添加数据
collection.add(
    documents=["2022年2月2号，美国国防部宣布：将向欧洲增派部队，应对俄乌边境地区的紧张局势.", " 2月17号，乌克兰军方称：东部民间武装向政府军控制区发动炮击，而东部民间武装则指责乌政府军先动用了重型武器发动袭击，乌东地区紧张局势持续升级"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)

# 如果 Chroma 收到一个文档列表，它会自动标记并使用集合的嵌入函数嵌入这些文档（如果在创建集合时没有提供嵌入函数，则使用默认值）。Chroma也会存储文档本身。如果文档过大，无法使用所选的嵌入函数嵌入，则会出现异常。

# 每个文档必须有一个唯一的相关ID。尝试.添加相同的ID两次将导致错误。可以为每个文档提供一个可选的元数据字典列表，以存储附加信息并进行过滤。

# 或者，您也可以直接提供文档相关嵌入的列表，Chroma将存储相关文档，而不会自行嵌入。
# collection.add(
#     embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
#     documents=["This is a document", "This is another document"],
#     metadatas=[{"source": "my_source"}, {"source": "my_source"}],
#     ids=["id1", "id2"]
# )

# 改数据
# 更新所提供 id 的嵌入、元数据或文档。
def update(ids: OneOrMany[ID],
           embeddings: Optional[OneOrMany[Embedding]] = None,
           metadatas: Optional[OneOrMany[Metadata]] = None,
           documents: Optional[OneOrMany[Document]] = None) -> None

# 更新所提供 id 的嵌入、元数据或文档，如果不存在，则创建它们。
def upsert(ids: OneOrMany[ID],
        embeddings: Optional[OneOrMany[Embedding]] = None,
        metadatas: Optional[OneOrMany[Metadata]] = None,
        documents: Optional[OneOrMany[Document]] = None) -> None          

# 删除数据
# 根据 ID 和/或 where 过滤器删除嵌入数据
def delete(ids: Optional[IDs] = None,
           where: Optional[Where] = None,
           where_document: Optional[WhereDocument] = None) -> None
# collection.delete(ids=["3", "4", "5"])

# 查询数据
results = collection.query(
    query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)
或者：
results = collection.query(
    query_texts=["俄乌战争发生在哪天？"],
    n_results=2
)

print(156, results)
156 {'ids': [['id1', 'id2']], 'embeddings': None, 'documents': [['2022年2月2号，美国国防部宣布：将向欧洲增派部队，应对俄乌边境地区的紧张局势.',
 ' 2月17号，乌克兰军方称：东部民间武装向政府军控制区发动炮击，而东部民间武装则指责乌政府军先动用了重型武器发动袭击，乌东地区紧张局势持续升级']], 
 'metadatas': [[{'source': 'my_source'}, {'source': 'my_source'}]], 'distances': [[1.2127416133880615, 1.3881784677505493]]}

# 删除数据库
res = httpClient.reset() #重置整个数据库， 要慎用！ 一般情况下此项设置为not allowed
```

## Chroma embedding
嵌入式是AI(人工智能)表示任何类型数据的原生方式，因此非常适合与各种AI(人工智能)工具和算法配合使用。它们可以表示文本、图像以及音频和视频。无论是本地使用安装的库，还是调用API，创建嵌入式都有很多选择。

Chroma为流行的嵌入式提供商提供了轻量级封装，使您可以轻松地在应用程序中使用它们。您可以在创建Chroma集合时设置一个嵌入函数，该函数将被自动使用，您也可以自己直接调用它们。
```py
from chromadb.utils import embedding_functions

# 默认值：all-MiniLM-L6-v2
# 默认情况下，Chroma 使用Sentence Transformers all-MiniLM-L6-v2模型来创建嵌入。该嵌入模型可以创建可用于各种任务的句子和文档嵌入。此嵌入功能在您的机器上本地运行，并且可能需要您下载模型文件（这将自动发生）。
default_ef = embedding_functions.DefaultEmbeddingFunction()

res = default_ef("hello world")
print(456, res)
# 向量化速度很慢，不推荐使用

# 自定义嵌入函数
# 您可以创建自己的嵌入函数以与 Chroma 一起使用，它只需要实现EmbeddingFunction协议即可。
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        # embed the documents somehow
        return embeddings


# OpenAI 的嵌入 API
# Chroma 为 OpenAI 的嵌入 API 提供了一个方便的包装器。此嵌入功能在 OpenAI 的服务器上远程运行，并且需要 API 密钥。您可以通过在OpenAI注册一个帐户来获取 API 密钥。此嵌入功能依赖于openai python 包，您可以使用pip install openai.
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="YOUR_API_KEY",
    model_name="text-embedding-ada-002"
)       
```

## Chroma docker
```py
mkdir chromadb & cd chromadb
git clone https://github.com/chroma-core/chroma.git

# 构建数据库服务端
# 1.从docker-compose构建
docker compose up -d --build

docker image ls
    REPOSITORY   TAG       IMAGE ID       CREATED          SIZE
    server       latest    1db4c828e77a   36 seconds ago   649MB
# 2. 直接从dicker hub中摘取
docker pull lemooljiang/chroma-server:latest
# 后台过行
docker run -d --name chromadb lemooljiang/chroma-server \
uvicorn chromadb.app:app --workers 1 --host 0.0.0.0 --port 8000 --proxy-headers --log-config log_config.yml

# 用户端
pip install chromadb
# client-only
pip install chromadb-client
# 请注意，chromadb-client软件包是完整Chroma库的子集，并不包含所有依赖项。如果您想使用完整的Chroma库，可以安装chromadb包。最重要的是，没有默认的嵌入函数。如果您在 add() 文档时没有使用嵌入函数，您必须手动指定一个嵌入函数并为其安装依赖项。

import chromadb
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
```

## langchain中的使用
[参考](https://python.langchain.com/docs/integrations/vectorstores/chroma)
```py
# /home/knowqa/know_env/lib/python3.10/site-packages/langchain/vectorstores
from langchain.vectorstores import Chroma

# langchain 默认文档 collections [Collection(name=langchain)]
# 持久化数据
persist_directory = './chromadb'
vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()
# 直接加载数据
vectordb = Chroma(persist_directory="./chromadb", embedding_function=embeddings)

eg:
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import os
os.environ["OPENAI_API_KEY"] = 'sk-xxxxxx'

loader = TextLoader('./russia.txt', encoding='gbk')  #中文必须带 encoding='gbk'
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
# chunk_size=1000表示每次读取或写入数据时，数据的大小为400个字节, 约200~400个汉字
# 对于英文LangChain一般会使用RecursiveCharacterTextSplitter处理。由于中文的复杂性，会使用到jieba等处理工具预处理中文语句。

docs = text_splitter.split_documents(documents)
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = vectordb.similarity_search(query)
print(docs[0].page_content)  # 默认是返回4条数据， k=4


# 直接加载数据库，然后查询相似度的文本
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
query = "On what date did the war between Russia and Ukraine take place?"
retriever = vectordb.as_retriever(search_type="mmr")
s = retriever.get_relevant_documents(query)
print(123, s)
/////
123 [Document(page_content='加意见和建议...乌克兰的战争”标语。', metadata={'source': './russiaX.pdf'}), Document(page_content='综合路透社、雅..导弹。', metadata={'source': './russiaX.pdf'}), Document(page_content='乌克兰东部问....
print(556, s[0].page_content)  # 选第一条的内容

# 或者这样查询
s = vectordb.similarity_search(query)
# print(s[0].page_content)

# 特别注意
在存入数据后有时不能立即查到新添加的数据，此时，关停后重启加载即可！


# 直接用get获取数据
res = vectordb.get(limit=2) 
print(266, res)
//266 {'ids': ['e8661882-358c-11ee-a7f1-fb75c83274a1', 'e8661883-358c-11ee-a7f1-fb75c83274a1'], 'embeddings': None, 'metadatas': [{'source': './uploads/dazhihui.txt'}, {'source': './uploads/dazhihui.txt'}], 'documents': ['大智汇健康科技...', '二、团队介绍...']}
```

## 添加文本
`add_texts`方法比文本传入的方法更灵活些。它可以控制文本的大小，从十几个字到上千字都可以，这和文本传入有很大不同。它有更多的定制性。缺点是要一条条传入，在数据不大的情况下可以这么做。
```py
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_name=collection)
test = "示例文本"
res = vectordb.add_texts(texts=[text])
print(259, res)
//返回ids列表
//['a05e3d0c-ab40-11ed-a853-e65801318981']

# 特别注意
在存入数据后有时不能立即查到新添加的数据，此时，关停后重启加载即可！
```

## 更新和删除数据
```py
# 以id的形式删除数据
# 将旧数据删除后，再添加新数据
# def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_name=collection)
res = vectordb.delete(ids=['7d7c61ae-3774-11ee-a7f1-fb75c83274a1']) 
print(433, res) #None    

# 以id的形式更新数据
# def update_document(self, document_id: str, document: Document) -> None

# 查询文档id
# langchain中没有将文档id返回的方法，需要自定义。
def similarity_search_all(
    self,
    query: str,
    k: int = DEFAULT_K,
    filter: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> List[Tuple[Document, float]]:
    """Run similarity search with Chroma with distance.

    Args:
        query (str): Query text to search for.
        k (int): Number of results to return. Defaults to 4.
        filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

    Returns:
        List[Tuple[Document, float]]: List of documents most similar to
        the query text and cosine distance in float for each.
        Lower score represents more similarity.
    """
    if self._embedding_function is None:
        results = self.__query_collection(
            query_texts=[query], n_results=k, where=filter
        )
    else:
        query_embedding = self._embedding_function.embed_query(query)
        results = self.__query_collection(
            query_embeddings=[query_embedding], n_results=k, where=filter
        )

    # return _results_to_docs_and_scores(results)
    return results

res = vectordb.similarity_search_all(query=text, k=2) 
//{'ids': [['7d7c61ae-3774-11ee-a7f1-fb75c83274a1', 'f208a315-358c-11ee-a7f1-fb75c83274a1']], 'distances': [[0.4394338843847249, 0.4735199946090204]], 'metadatas': [[{'source': './uploads/tel.txt'}, {'source': './uploads/yishejun.txt'}]], 'embeddings': None, 'documents': [['电话:155xx444', '大智汇的...XX路']]}    
```

## 相关度数值检索
Chroma中除了`similarity_search`,还有另一个更适宜的函数`similarity_search_with_score`。它不仅会返回数据，还会同时将相关度数值（score）一起返回。对数据进行判断，再决定是否要采用此数据。
```py
# similarity_search_with_score，
# 它不仅允许你返回文档，还允许你返回查询与文档的距离分值。
# 这个分值是余弦值， 同时越低越是相关。
# 源码如下，写得还是比较清楚，返回的数据是文档和数值
def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """

docs = vectordb.similarity_search_with_score(query)
docs[0]
//(Document(page_content='Tonight. I c..inds, who will...xcellence.', metadata={'source': '../../../state_of_the_union.txt'}),1.1972057819366455)

eg2: 加filter
docs = vectordb.similarity_search_with_score(query=ask, k=2, filter=dict(source='./uploads/yisheng_update.docx')) 
// [(Document(page_content='更多详细咨询营养师...的原因。', metadata={'source': './uploads/yisheng_update.docx'}), 186.72679092063976), (Document(page_content='益生菌就定植于..最直接的关系，能够在.的...畅，自然和谐统一的状态。', metadata={'source': './uploads/yisheng_update.docx'}), 221.41649154602675)]

相关度数值（score）还是挺迷的，到底多少算是相关，多少算是不相关呢？这里，我做了一些测试。 用的都是中文，向量计算embedding分别采用了OpenAIEmbeddings， text2vec这两个库，计算出的数据经过一番比较，得到：
OpenAIEmbeddings中，低于0.385的相关度高，高于0.4的基本不相关；
text2vec中，低于256的算是相关度高，高于300的就基本不相关了！

eg:
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import dotenv_values

env_vars = dotenv_values('.env')

# 加载数据库
persist_directory = './chromac'
collection = 'bigccx'
embedding = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=env_vars['OPENAI_API_KEY']
)

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_name=collection)

# 查询相似度的文本
def queryVectorDB(ask):	
	s = vectordb.similarity_search_with_score(query=ask, k=1) 
	if len(s) == 0:
		return ""
	else:
		if s[0][1] < 0.385:   # 文本关联强则返回，不相关则不返回. shiba < 256  openai < 0.385
			return s[0][0].page_content
		else:
			return ""


eg2:  text2vec模型质量一般，推荐使用OpenAIEmbeddings
from langchain.vectorstores import Chroma
import sentence_transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import PromptTemplate

# 加载数据库
persist_directory = './chromac'
collection = 'bighao'
embedding = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_name=collection)

# 查询相似度的文本
def queryVectorDB(ask):	
	s = vectordb.similarity_search_with_score(query=ask, k=1, filter=dict(source='./uploads/yisheng_update.docx')) 
	# print(699, s[0][0].page_content, 99, s[0][1])
	if len(s) == 0:
		return ""
	else:
		if s[0][1] < 256:   # 文本关联强则返回，不相关则不返回
			return s[0][0].page_content
		else:
			return ""
```

## metadata数据过滤
```py
# Filtering on metadata
# It can be helpful to narrow down the collection before working with it.
# For example, collections can be filtered on metadata using the get method.
# filter collection for updated source
example_db.get(where={"source": "some_other_source"})
    {'ids': [], 'embeddings': None, 'metadatas': [], 'documents': []}

results_with_scores = db.similarity_search_with_score("foo", filter=dict(page=1))
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

    Content: foo, Metadata: {'page': 1}, Score: 5.159960813797904e-15
    Content: foo, Metadata: {'page': 2}, Score: 5.159960813797904e-15
    Content: foo, Metadata: {'page': 3}, Score: 5.159960813797904e-15
    Content: foo, Metadata: {'page': 4}, Score: 5.159960813797904e-15
```

## 直接存入数据库和查询
不推荐使用，中文的关联度很低！ 默认的embedding函数对中文的理解很差，另选其它。
```py
# 存入数据库
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import uuid

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",  persist_directory="./chromadb"))
collection = client.get_or_create_collection(name="my_collection8")
loader = TextLoader('./russia.txt', encoding='gbk')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

Adocs = []
Ids = []
for i in range(len(docs)):
    Adocs.append(docs[i].page_content)
    Ids.append(str(uuid.uuid4()))
print(566, Adocs, 666, Ids )
collection.add(documents=Adocs, ids=Ids)
print(126, "saveDb ok", collection)


# 查询  中文查询的结果不理想，不推荐使用！！
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",  persist_directory="./chromadb"))
collection = client.get_collection(name="my_collection8")
results = collection.query(
    query_texts=["俄乌战争发生在哪天？"],
    n_results=1
)
print(156, results)
{'ids': [['ab51abbe-6d3f-4c0e-a2cc-b245a3811ae9']], 'embeddings': None, 'documents': [['综合路透社、雅虎新闻等网站 2023 年 3 月 9 日报道，乌克兰官员称，3 月 9 日早些时候，俄 罗斯发动空袭，袭击了乌克兰多个地区，包括黑海港口敖德萨和乌克兰第二大城市哈尔科夫， 导致多个地区断电。这是俄罗斯时隔 25 天以后再次发动大规模袭击，俄军上次大规模导弹袭 击还是 2 月 10 日。俄军当时使用了巡飞弹和巡航导弹。']], 'metadatas': [[None]], 'distances': [[0.9033191204071045]]}
# print(366, results['documents'][0][0])
```

## httpClient
用docker运行了服务端，直接和它相连
```py
import chromadb
# from langchain.vectorstores import Chroma
from chromaX import ChromaX

# 加载和实例化数据库
# 数据库地址 /home/chromadb/chroma/chroma 
collection = 'testNN'
embedding = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=env_vars['OPENAI_API_KEY']
)
httpClient = chromadb.HttpClient(host='localhost', port=8000)
vectordb = ChromaX(collection_name=collection, embedding_function=embedding, client=httpClient)
print(33, vectordb)
```