# <center> LangChain  </center>

## 下载与资源
[github |](https://github.com/langchain-ai/langchain)
[文档｜](https://python.langchain.com/docs/introduction/)
[工具包 |](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)
[中文文档｜](https://python.langchain.com.cn/docs/)
[Langchain中文教程 |](https://www.langchain.com.cn/)
[handbook |](https://www.pinecone.io/learn/langchain-intro/)
[中文说明 ｜](https://liaokong.gitbook.io/llm-kai-fa-jiao-cheng/)
[中文说明2 ｜](https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide) 
[自动化 ｜](https://github.com/logspace-ai/langflow)
[基于LangChain的优秀项目资源库 |](https://mp.weixin.qq.com/s/G9aqBFzd5j8wVPTH160pZA)
[SERP API |](https://serpapi.com/)
[supabase |](https://supabase.com/)
[AutoGPT |](https://github.com/Dogtiti/-Next-Web)
[chatpdf |](https://github.com/austin2035/chatpdf)
[chat-with-documents |](https://github.com/ciocan/langchain-chat-with-documents)
[本地知识库问答 ｜](https://juejin.cn/post/7236028062873550908)
[js文档 ｜](https://js.langchain.com/docs/get_started/introduction/)


## 安装
```
//创建虚拟环境
cd /home
mkdir pythonEnv
virtualenv pythonEnv/rest_demo
//virtualenv -p /usr/bin/python3.8 pythonEnv/rest_demo 
source pythonEnv/rest_demo/bin/activate  //激活环境
deactivate   // 退出环境

pip install langchain   # 0.0.240  0.1.13  0.1.17  0.3.7
//pip install langchain -U  升级
// pip install openai -U   升级
//pip install langchain -i https://pypi.tuna.tsinghua.edu.cn/simple  国内一定要切换源
pip list
```

## 功能模块
![langchain.jpg](https://ipfs.ilark.io/ipfs/QmXtWs1jtAzZN5Ye6qy9FJKavDh8nbTHgExTS7smpLu1nF)

Model I/O：管理大语言模型（Models），及其输入（Prompts）和格式化输出（Output Parsers）

Data Connection：管理主要用于建设私域知识（库）的向量数据存储（Vector Stores）、内容数据获取（Document Loaders）和转化（Transformers），以及向量数据查询（Retrievers）

Memory：用于存储和获取 对话历史记录 的功能模块

Chains：用于串联 Memory ↔️ Model I/O ↔️ Data Connection，以实现 串行化 的连续对话、推测流程

Agents：基于 Chains 进一步串联工具（Tools），从而将大语言模型的能力和本地、云服务能力结合

Callbacks：提供了一个回调系统，可连接到 LLM 申请的各个阶段，便于进行日志记录、追踪等数据导流

//更新
1、LangGraph核心组件: Graphs、State、Nodes、Edges、Send、checkpointer

2、LangGraph 实现：可控性、持久化、Human-in-the-loop、
streaming、React agent

3、Agent使用案例： Chatbots，Multi-Agent Systems, Planning Agent

## 开始应用
```py
# pip install langchain-openai
pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple  // 0.2.8

# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxx"
# If you are behind an explicit proxy, you can use the OPENAI_PROXY environment variable to pass through
# os.environ["OPENAI_PROXY"] = "http://proxy.yourcompany.com:8080"
# 如果希望通过代理来访问可以配置上
# os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
text = "天空为什么是蓝的？"
print(llm(text))
```

## 联网搜索
让我们的 OpenAI api 联网搜索，并返回答案给我们。这里我们需要借助 Serpapi 来进行实现，Serpapi 提供了 google 搜索的 api 接口。
[参考](https://mp.weixin.qq.com/s/D9e4zzGQdKCnNYxmRFaqBQ)
```py
pip install google-search-results

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
import os
os.environ["OPENAI_API_KEY"] = 'sk-vxxxx'
os.environ["SERPAPI_API_KEY"] = '97xxxxxx'


# 加载 OpenAI 模型
llm = OpenAI(temperature=0.2,max_tokens=2048) 

 # 加载 serpapi 工具
tools = load_tools(["serpapi"])

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
# tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 运行 agent
agent.run("What's the date today? What great events have taken place today in history?")

> Entering new AgentExecutor chain...
 I should look up the current date and then search for any events that have taken place on that day.
Action: Search
Action Input: Current date
Observation: Today's current date and time with time zone and date picker: Select locale. en-US. Wednesday, May 3, 2023. 5:00:05 PM. PST8PDT (GMT-7).
Thought: I should now search for any events that have taken place on this day.
Action: Search
Action Input: Events that have taken place on May 3
Observation: On this day - May 3
Thought: I now know the events that have taken place on this day.
Final Answer: Today is Wednesday, May 3, 2023. On this day in history, the first successful ascent of Mount Everest was made by Edmund Hillary and Tenzing Norgay in 1953.
```

## 加载文档
UnstructuredFileLoader可以加载多种文本，txt, md, html, pdf,docx这些都可以，但在国内不能用，只能用TextLoader

[参考](https://python.langchain.com/docs/integrations/document_loaders/unstructured_file)
[参考2](https://mp.weixin.qq.com/s/OwOmXey_bYNcHpgSFYcVyA)
```py
# # Install package
# pip install --upgrade --quiet  "unstructured[all-docs]"
pip install "unstructured[all-docs]"
// pip install "unstructured[all-docs]" -i https://pypi.tuna.tsinghua.edu.cn/simple
// pip install -i https://pypi.tuna.tsinghua.edu.cn/simple unstructured
// pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pdf2image
// pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
// pip install -i https://pypi.tuna.tsinghua.edu.cn/simple unstructured-inference
// pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pikepdf

除了Python包，还需要下载 nltk_data，这东西非常大，下载起来非常慢。所以们可以事先下好，放到固定的位置。
下载地址：https://github.com/nltk/nltk_data/tree/gh-pages
下载完后，将其中的packages文件夹内的全部内容拷贝到固定位置，例如上面的 C:\Users\xxx\AppData\Roaming\nltk_data

from langchain_community.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader("./example_data/state_of_the_union.txt")
docs = loader.load()

# 注意：UnstructuredFileLoader国内用不了！ 用TextLoader
from langchain_community.document_loaders import TextLoader
loader = TextLoader('./russia.txt', encoding='gbk')  #中文必须带 encoding='gbk'
documents = loader.load()

//////
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

loader = UnstructuredWordDocumentLoader("example_data/fake.docx")
data = loader.load()
```

## 加载网络资料
```py
from langchain_community.document_loaders import WebBaseLoader

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
```

## 文本分割
实现文本分割器倒不算难，但也有几个坑，主要是对中文的支持。读取文本时必须使用encoding='gbk'，否则分割不了。
[大语言模型应用中的文本分块策略](https://mp.weixin.qq.com/s/S8RecRgiGO_rLbnwC_iExQ)
```py
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader('./russia.txt', encoding='gbk')  #中文必须带 encoding='gbk'
# 文本最好做一些加工，段落最好有两个换行
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
# chunk_size=400表示每次读取或写入数据时，数据的大小为400个字节, 约300~400个汉字
docs = text_splitter.split_documents(documents)
print(112, docs)
//// 得到的结果如下：
112 [Document(page_content='2022年1月10号到13号，...兵与俄罗斯作战！', metadata={'source': './russia.txt'}), Document(page_content='2022年2月24日，随...316人受伤。', metadata={'source': './russia.txt'}), ...]

# 另一种实现方法
from langchain_text_splitters import RecursiveCharacterTextSplitter
with open('./russia.txt', encoding='gbk') as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap  = 0,
    length_function = len,
)

texts = text_splitter.create_documents([state_of_the_union])

print(123, texts)

注： 这两种方法实现的结果都差不多！
```

## embeddings
[text2vec |](https://github.com/shibing624/text2vec)
[参考 |](https://zhuanlan.zhihu.com/p/622017658)
[本地知识库 |](https://juejin.cn/post/7210005376653361213?searchId=20230821101516BD321EF6F8D9E28CA861)
[text2vec-base |](https://huggingface.co/shibing624/text2vec-base-chinese/tree/main)
[m3e |](https://huggingface.co/moka-ai)
[FlagEmbedding |](https://github.com/FlagOpen/FlagEmbedding)
```py
# OpenAIEmbeddings
pip install langchain-openai

from dotenv import dotenv_values
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import dotenv_values

env_vars = dotenv_values('.env')
# os.environ["OPENAI_API_KEY"] = env_vars['OPENAI_API_KEY']
# If you are behind an explicit proxy, you can use the OPENAI_PROXY environment variable to pass through
# os.environ["OPENAI_PROXY"] = "http://proxy.yourcompany.com:8080"
# 如果希望通过代理来访问可以配置上
# os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
# embedding = OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small ", #text-embedding-ada-002
    openai_api_key=env_vars['OPENAI_API_KEY']
    openai_api_base=env_vars['OPENAI_API_BASE']
)
# embedding = OpenAIEmbeddings(
#     model="text-embedding-ada-002",
#     openai_api_key=env_vars['OPENAI_API_KEY'],
#     openai_api_base=env_vars['OPENAI_API_BASE']
#     openai_organization="ilark",
#     chunk_size=400,
#     request_timeout=60
# )

text = "This is a test query."
query_result = embeddings.embed_query(text)
print(66, query_result)
//[-0.005034489091485739, 0.005091584753245115, -0.005185624584555626, .......]


# HuggingFaceEmbeddings（text2vec）
分割使用的模型和语言大模型类似，也要使用语义库。那么，如果用本地的语义库来分割，要装的包还挺多的，如果不熟悉，很容易就掉进坑里。
pip install sentence-transformers

import sentence_transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')
# 较好的中文模型，需要下载包，最好有GPU资源
# 默认存入 /root/.cache/huggingface, 大小约849M
# /root/.cache/torch/sentence_transformers 大小约391M
# 在短文本的表现上不好

text = "This is a test document."

# query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])

print(123, doc_result)
//[[-0.6536310911178589, 0.17037765681743622, 0.19515305757522583, ......

# 两种方式的差别：
OpenAIEmbeddings：
使用简单，并且效果比较好；
会消耗openai的token，特别是大段文本时，消耗的token还不少，如果知识库是比较固定的，可以考虑将每次生成的embedding做持久化，这样就不需要再调用openai了，可以大大节约token的消耗；
可能会有数据泄露的风险，如果是一些高度私密的数据，不建议直接调用。

HuggingFaceEmbeddings：
可以在HuggingFace上面选择各种sentence-similarity模型来进行实验，数据都是在本机上进行计算
需要一定的硬件支持，最好是有GPU支持，不然生成数据可能会非常慢
生成的向量效果可能不是很好，并且HuggingFace上的中文向量模型不是很多。

# 其它模型
# GanymedeNil/text2vec-large-chinese
#  /root/.cache/huggingface
# 默认存入 /root/.cache/torch/sentence_transformers  大小约2.5G
embedding = HuggingFaceEmbeddings(model_name='GanymedeNil/text2vec-large-chinese')
# 模型特点： 能正确查找到最关联的数据，但是，余弦值分布太分散，无法分辨是相关还是不相关的数据，无法实用！
# 有短文本上计算很差


# model config
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "ernie-medium": "nghuyong/ernie-3.0-medium-zh",
    "ernie-xbase": "nghuyong/ernie-3.0-xbase-zh",
    "text2vec-base": "GanymedeNil/text2vec-base-chinese",
    'simbert-base-chinese': 'WangZeJun/simbert-base-chinese',
    'paraphrase-multilingual-MiniLM-L12-v2': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
```

## 基于文本的问答
[](https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html)

```py
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import os
os.environ["OPENAI_API_KEY"] = 'sk-vgxxxxxx'

loader = TextLoader('./russia.txt', encoding='gbk') 

index = VectorstoreIndexCreator().from_loaders([loader])

query = "what date did the war between Russia and Ukraine take place?"
res = index.query(query)
print(23, res)
```

## Wikipedia API
[参考](https://www.codenong.com/s-getting-started-with-pythons-wikipedia-api/)
```py
pip install wikipedia

from langchain.utilities import WikipediaAPIWrapper

wikipedia = WikipediaAPIWrapper()
res = wikipedia.run('what is bitcoin?')
print(res)


import wikipedia
wikipedia.set_lang("zh")  
# print(wikipedia.summary("what is bitcoin?"))
print(11, wikipedia.summary("比特币"))
print(23, wikipedia.page("比特币").content)
# 'zh-cn': '中文（中国大陆）', 'zh-hans': '中文（简体）'
```

## 封装ChatGLM的LLM
[ChatGLM 集成进LangChain工具](https://juejin.cn/post/7226157821708681277)
```py
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from typing import Dict, List, Optional, Tuple, Union

import requests
import json

class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # headers中添加上content-type这个参数，指定为json格式
        headers = {'Content-Type': 'application/json'}
        data=json.dumps({
          'prompt':prompt,
          'temperature':self.temperature,
          'history':self.history,
          'max_length':self.max_token
        })
        # print("ChatGLM prompt:",prompt)
        # 调用api
        response = requests.post("{your_host}/api",headers=headers,data=data)
		# print("ChatGLM resp:",response)
        if response.status_code!=200:
          return "查询结果错误"
        resp = response.json()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history+[[None, resp['response']]]
        return resp['response']
```


