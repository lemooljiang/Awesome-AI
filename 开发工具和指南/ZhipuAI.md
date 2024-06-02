# <center>ZhipuAI</center>

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
pip install zhipuai   // 1.0.7  2.0.1
// pip install zhipuai  -i https://pypi.tuna.tsinghua.edu.cn/simple
//升级
pip install --upgrade zhipuai 
```


## 基本使用
```py
from zhipuai import ZhipuAI
from dotenv import dotenv_values
import sys
import json


env_vars = dotenv_values('../.env')
client = ZhipuAI(api_key=env_vars['ZHIPU_API_KEY'])


def zhipuChatV2( ):
    response = client.chat.completions.create(
        model = "glm-4",
        messages = [{"role": "user", "content": "人工智能"}],
        top_p = 0.7,
        temperature = 0.9,
    )
    # print(33, response,36, type(response), file=sys.stderr)
    print(33, response, file=sys.stderr)
    print(366, response.choices[0].message, file=sys.stderr)
    # print(response.choices[0].message)
    return response.choices[0].message.content
```

## 函数调用
```py
from zhipuai import ZhipuAI
import json


client = ZhipuAI(api_key="xxx")

def query_train_info(date, departure , destination):
    #此外是调用的API，这里做测试时直接给出结果
    print(6689, "query_train_info",date, departure, destination)
    return "北京-广州-9966"

def parse_function_call(model_response,messages):
    # 处理函数调用结果，根据模型返回参数，调用对应的函数。
    # 调用函数返回结果后构造tool message，再次调用模型，将函数结果输入模型
    # 模型会将函数调用结果以自然语言格式返回给用户。
    print(556, model_response.choices[0].message)
    if model_response.choices[0].message.tool_calls:
        tool_call = model_response.choices[0].message.tool_calls[0]
        args = tool_call.function.arguments
        function_result = {}
        if tool_call.function.name == "query_train_info":
            function_result = query_train_info(**json.loads(args))
        messages.append({
            "role": "tool",
            "content": f"{json.dumps(function_result)}",
            "tool_call_id":tool_call.id
        })
        response = client.chat.completions.create(
            model="glm-4",
            messages=messages
        )
        print(666, response.choices[0].message)
        # messages.append(response.choices[0].message.model_dump())


def test_func_call():
    messages = []   
    messages.append({"role": "system", "content": "不要假设或猜测传入函数的参数值。如果用户的描述不明确，请要求用户提供必要信息"})
    messages.append({"role": "user", "content": "帮我查询1月23日，北京到广州的航班"})
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_train_info",
                "description": "根据始发地、目的地和日期，查询对应日期的航班号",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "departure": {
                            "description": "出发地",
                            "type": "string"
                        },
                        "destination": {
                            "description": "目的地",
                            "type": "string"
                        },
                        "date": {
                            "description": "日期",
                            "type": "string",
                        }
                    },
                    "required": [ "departure", "destination", "date" ]
                },
            }
        }
    ]
     
    response = client.chat.completions.create(
        model="glm-4",
        messages=messages,
        tools=tools,
    )
    print(112, response.choices[0].message)
    messages.append(response.choices[0].message.model_dump())
    parse_function_call(response,messages)

if __name__ == "__main__":
    test_func_call() 

112
content=None role='assistant' tool_calls=[CompletionMessageToolCall(id='call_8477880785126313244', function=Function(arguments='{"date":"2022-01-23","departure":"北京","destination":"广州"}', name='get_flight_number'), type='function')]

556
556 content=None role='assistant' tool_calls=[CompletionMessageToolCall(id='call_8477880819486041138', function=Function(arguments='{"date":"2022-01-23","departure":"北京","destination":"广州"}', name='query_train_info'), type='function')]

6689 query_train_info 2022-01-23 北京 广州
666 content='根据您提供的信息，我查询到了1月23日从北京到广州的航班信息，航班号为“北京-广州-9966”。请注意，这是一个航班号，具体的航班时间、航空公司和其他详细信息需要您通过航空公司或航班查询服务进一步确认。如果您需要更多帮助，请告知我。' role='assistant' tool_calls=None    
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


## 向量计算
```py
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="xxxx")

response = client.embeddings.create(
    model="embedding-2", #填写需要调用的模型名称
    input="你好",
)

print(662, response.data[0].embedding)
```
