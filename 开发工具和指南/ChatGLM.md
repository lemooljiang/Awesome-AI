# <center>ChatGLM</center>

## 下载与资源
[github |](https://github.com/THUDM/ChatGLM-6B)
[github2 |](https://github.com/THUDM/ChatGLM2-6B)
[官网 |](https://chatglm.cn/)
[智谱AI |](https://open.bigmodel.cn)
[文档 |](https://open.bigmodel.cn/doc/api)
[课程 ｜](https://huggingface.co/learn/nlp-course/zh-CN/chapter0/1?fw=pt)
[本地知识库 ｜](https://github.com/imClumsyPanda/langchain-ChatGLM)
[本地知识库2 ｜](https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui)
[DB-GPT |](https://github.com/csunny/DB-GPT)
[ChatLongDoc |](https://github.com/webpilot-ai/ChatLongDoc)
[基于 ChatGLM-6B 搭建个人专属知识库 |](https://baijiahao.baidu.com/s?id=1765950735847976093&wfr=spider&for=pc)
[安装 |](https://baijiahao.baidu.com/s?id=1769236478823582890&wfr=spider&for=pc)
[ChatGLM-6B 部署与 P-Tuning 微调实战](https://baijiahao.baidu.com/s?id=1765123631287305087)
[视频链接](https://www.bilibili.com/video/BV1414y1m7mE)
[用Langchain-ChatGLM解析小说 |](https://mp.weixin.qq.com/s/6TckgOO3ZKS9lDhHOq5h0A)
[Langchain-ChatGLM中间件的深度学习 |](https://mp.weixin.qq.com/s/m6JZvUPU2lzRSPlbKtXABA)
[集成进LangChain |](https://juejin.cn/post/7226157821708681277)
[LLM实现QA问答中的一些细节笔记 |](https://zhuanlan.zhihu.com/p/627439522)

ChatGLM-6B模型是一个62亿参数规模的中英双语对话语言模型，它使用了清华大学KEG实验室与智谱AI公司共同构建的一个包含超过1000亿词汇量的中英双语数据集进行预训练。该数据集涵盖了各种类型和领域的文本数据，包括新闻、百科、社交媒体、小说、电影剧本等，并且特别增加了大量的对话数据，如电视剧台词、聊天记录、问答平台等。通过这样一个丰富多样的数据集，ChatGLM-6B模型可以学习到更加全面和深入的语言知识，并且可以更好地适应不同风格和主题的对话场景

通过这些优化措施，ChatGLM-6B模型可以在消费级的显卡上进行本地部署，并且可以实现实时的对话交互。根据清华大学KEG实验室与智谱AI公司提供的数据，ChatGLM-6B模型在INT4量化级别下最低只需6GB显存就可以运行，并且在RTX 3090显卡上的推理速度可以达到每秒10个句子（每个句子包含20个词）。


## 环境
```
//python 3.10.6  pip 23.1.2
virtualenv glm_env
source glm_env/bin/activate
source glm_env/Scripts/activate   //windows
```


## 安装
```py
pip install zhipuai

import zhipuai
zhipuai.model_api.invoke(
    model="具体的模型编码",
    ... # 具体模型参数
)

import zhipuai

zhipuai.api_key = "your api key"
response = zhipuai.model_api.invoke(
    model="chatglm_lite",
    prompt=[
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "我是人工智能助手"},
        {"role": "user", "content": "你叫什么名字"},
        {"role": "assistant", "content": "我叫chatGLM"},
        {"role": "user", "content": "你都可以做些什么事"},
    ]
)

def invoke_example():
    response = zhipuai.model_api.invoke(
        model="chatglm_lite",
        prompt=[{"role": "user", "content": "人工智能"}],
        top_p=0.7,
        temperature=0.9,
    )
    print(response)
```

## 流式传输
[流式chatGPT接口 |](https://blog.csdn.net/time_forgotten/article/details/130437413)
[github |](https://github.com/wemio/chatGPTFlaskWebAPI)
```py
# SSE 响应是字符串流格式，先看下具体响应示例
def sse_invoke_example():
    response = zhipuai.model_api.sse_invoke(
        model="chatglm_lite",
        prompt=[{"role": "user", "content": "人工智能"}],
        top_p=0.7,
        temperature=0.9,
    )

    for event in response.events():
        if event.event == "add":
            print(event.data)
        elif event.event == "error" or event.event == "interrupted":
            print(event.data)
        elif event.event == "finish":
            print(event.data)
            print(event.meta)
        else:
            print(event.data)
///////
id: "fb981fde-0080-4933-b87b-4a29eaba8d17"
event: "add"
data: "作为一个"
 
id: "fb981fde-0080-4933-b87b-4a29eaba8d17"
event: "add"
data: "大型语言模型"
 
id: "fb981fde-0080-4933-b87b-4a29eaba8d17"
event: "add"
data: "我可以"
 
... ...
 
Id: "fb981fde-0080-4933-b87b-4a29eaba8d17"
event: "finish"
meta: {"request_id":"123445676789","task_id":"75931252186628","task_status":"SUCCESS","usage":{"prompt_tokens":215,"completion_tokens":302,"total_tokens":517}}

eg:
def streaming(query, temperature):
    response = zhipuai.model_api.sse_invoke(
        model = "chatglm_lite",
        prompt = query,
        top_p = 0.7,
        temperature = temperature,
    )
    for event in response.events():
        if event.event == "add":
            yield event.data
        elif event.event == "error" or event.event == "interrupted":
            print(444, event, file=sys.stderr)
            return "error", 500
        elif event.event == "finish":
            return
        else:
            return
```

## 自定义LLM包装器
```py
pip install langchain -i https://pypi.tuna.tsinghua.edu.cn/simple  (国内要切换源)

from typing import Dict, List, Optional, Tuple, Union
from langchain.llms.base import LLM
import zhipuai

# os.environ["zhipuai.api_key"] = "d86xxxx"
zhipuai.api_key = "d862xxxxx"

class ChatLLM(LLM):
    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response = zhipuai.model_api.invoke(
            model="chatglm_lite",
            prompt=[{"role": "user", "content": prompt}],
            top_p=0.7,
            temperature=0.9,
        )
        print(45, response['data']['choices'][0]['content'])
        return response['data']['choices'][0]['content']
```


## text2vec
[text2vec](https://github.com/shibing624/text2vec)
[text2vec](https://pypi.org/project/text2vec/)
[PaddleNLP Embedding AP](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/embeddings.htm)

Text2vec: Text to Vector, Get Sentence Embeddings. 文本向量化，把文本(包括词、句子、段落)表征为向量矩阵。
text2vec实现了Word2Vec、RankBM25、BERT、Sentence-BERT、CoSENT等多种文本表征、文本相似度计算模型，并在文本语义匹配（相似度计算）任务上比较了各模型的效果。
```py
pip install torch # conda install pytorch  2.0.1
pip install -U text2vec  # 1.2.1
```

## 文本向量表征
基于pretrained model计算文本向量
```py
>>> from text2vec import SentenceModel
>>> m = SentenceModel()
>>> m.encode("如何更换花呗绑定银行卡")
//[-3.80448222e-01 -5.13956249e-01  4.24977601e-01 -3.34187746e-01
  4.87365782e-01 -4.59591091e-01  7.30618536e-01  2.61669636e-01
 -1.09546185e-01 -1.56673267e-01  1.00216138e+00  1.01447773e+00
  6.24635041e-01 -5.28593898e-01 ......]

# shibing624/text2vec-base-chinese是text2vec.SentenceModel指定的默认模型
```

## embedding
```py
import sys

sys.path.append('..')
from text2vec import SentenceModel
from text2vec import Word2Vec


def compute_emb(model):
    # Embed a list of sentences
    sentences = [
        '卡',
        '银行卡',
        '如何更换花呗绑定银行卡',
        '花呗更改绑定银行卡',
        'This framework generates embeddings for each input sentence',
        'Sentences are passed as a list of string.',
        'The quick brown fox jumps over the lazy dog.'
    ]
    sentence_embeddings = model.encode(sentences)
    print(type(sentence_embeddings), sentence_embeddings.shape)

    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("101 Sentence:", sentence)
        print("102 Embedding shape:", embedding.shape)
        print("103 Embedding head:", embedding[:10])
        print()


if __name__ == "__main__":
    # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
    t2v_model = SentenceModel("shibing624/text2vec-base-chinese")
    compute_emb(t2v_model)

    # 支持多语言的句向量模型（Sentence-BERT），英文语义匹配任务推荐，支持fine-tune继续训练
    sbert_model = SentenceModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    compute_emb(sbert_model)

    # 中文词向量模型(word2vec)，中文字面匹配任务和冷启动适用
    w2v_model = Word2Vec("w2v-light-tencent-chinese")
    compute_emb(w2v_model)
```

## moka-ai/m3e-base
[moka-ai]( https://huggingface.co/moka-ai/m3e-base)
```py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('moka-ai/m3e-base')

#Our sentences we like to encode
sentences = [
    '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
    '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
    '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
]

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")    
```

## 句子相似度计算
```py
from text2vec import Similarity

# Two lists of sentences
sentences1 = ['如何更换花呗绑定银行卡',
              'The cat sits outside',
              'A man is playing guitar',
              'The new movie is awesome']

sentences2 = ['花呗更改绑定银行卡',
              'The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

sim_model = Similarity()
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        score = sim_model.get_score(sentences1[i], sentences2[j])
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], score))
//句子余弦相似度值score范围是[-1, 1]，值越大越相似。        
```


## Gradio
[文档 |](https://huggingface.co/learn/nlp-course/zh-CN/chapter9/7?fw=pt)
[Gradio入门到进阶](https://blog.csdn.net/sinat_39620217/article/details/130343655)

在前面的部分中, 我们已经使用 Interface 类探索并创建了演示。在本节中, 我们将介绍我们 新开发的称为gradio.Blocks低级API。

现在, 接口和块之间有什么区别?
    接口: 一个高级 API, 让你只需提供输入和输出列表即可创建完整的机器学习演示。
    块: :一个低级的 API, 它允许你完全控制你的应用程序的数据流和布局。您可以使用块(如 “构建块”)构建非常复杂的多步骤应用程序。
```py
import gradio as gr

def flip_text(x):
    return x[::-1]

demo = gr.Blocks()

with demo:
    gr.Markdown(
        """
    # Flip Text!
    Start typing below to see the output.
    """
    )
    input = gr.Textbox(placeholder="Flip this text")
    output = gr.Textbox()

    input.change(fn=flip_text, inputs=input, outputs=output)

# demo.launch()
demo.launch(server_name='0.0.0.0', # ip for listening, 0.0.0.0 for every inbound traffic, 127.0.0.1 for local inbound
            server_port=7860, # the port for listening
            show_api=False, # if display the api document
            share=False, # if register a public url
            inbrowser=False)
```
