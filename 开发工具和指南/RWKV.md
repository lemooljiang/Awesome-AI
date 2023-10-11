# <center>RWKV</center>

## 下载与资源
[huggingface |](https://huggingface.co/RWKV)
[官网 |](www.rwkv.com)
[论文 |](https://arxiv.org/abs/2305.13048)
[github |](https://github.com/BlinkDL/ChatRWKV)
[电脑使用安装包 |](https://github.com/josStorer/RWKV-Runner)
[手机使用安装包 |](https://github.com/ZTMIDGO/RWKV-Android)
[RWKV的Chat模型 |](https://zhuanlan.zhihu.com/p/618011122)
[中文小说续写体验 ｜](https://modelscope.cn/studios/BlinkDL/RWKV-CHN/summary)
[微调教学 |](https://zhuanlan.zhihu.com/p/638326262)
[本地部署 |](https://github.com/cgisky1980/ai00_rwkv_server/blob/main/README_zh.md)


## API节点
https://better-chat-rwkv.ai-creator.net/
https://rwkv.ai-creator.net/jpntuned/v1/chat/completions
https://rwkv.ai-creator.net/chntuned/v1/chat/completions

目前尚不需要密钥，后续我们会高度优先考虑妥善存储您在我方的API密钥。

## python使用
```py
# 方法一
import openai as RWKV
import sys

RWKV.api_base = "https://rwkv.ai-creator.net/chntuned/v1"
RWKV.api_key = ""

def rwkvChatX(query, temperature):
    completion = RWKV.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = query,
        temperature = temperature
    )
    print(6617, "completion", completion.choices[0].message.content)
    return completion.choices[0].message.content


# 方法二    
import requests

def send_get_request(prompt):
    try:
        url= "https://rwkv.ai-creator.net/chntuned/v1/chat/completions"
        message=[{"role": "user", "content": prompt}]
        data = {
            "messages":message,
            "temperature":0.3,
            "max_tokens":1500
        }
        headers = {'content-type': 'application/json', 'Authorization': 'Bearer '}
        response = requests.post(url=url, json=data, headers=headers)
        result = eval(response.text)
        response_text = result["choices"][0]["message"]["content"]
        print("res:",response_text)
        return ""

    except requests.exceptions.HTTPError as e:
        print('HTTPError:', e.response.status_code, e.response.reason)
    except requests.exceptions.RequestException as e:
        print('RequestException:', e)

response_data = send_get_request("introduce youself")
print(response_data)
```

## nodejs使用
```js
//方法一
import { Configuration, OpenAIApi } from 'openai'


const configuration = new Configuration({
  apiKey: "",
  basePath: "https://rwkv.ai-creator.net/chntuned/v1"
})

const RWKV = new OpenAIApi(configuration)

async function test(query){
    const response = await RWKV.createChatCompletion({
        messages: query,
        temperature: 0.2
      })
      console.log(55, response.data.choices[0].message)
}
let s = [{"role": "user", "content": "介绍一下你自己"}]
test(s)


//方法二
import fetch from "node-fetch"

const url= "https://rwkv.ai-creator.net/chntuned/v1/chat/completions"
async function test(){
	const response = await fetch(url, {
	  method: 'POST',
	  headers: {
		'content-type': 'application/json'
	  },
	  body: JSON.stringify({
	    messages: [
	      {"role": "system", "content": "You are a helpful assistant."},
	      {"role": "user", "content": "介绍一下你自己"}
	    ],
	    "temperature":0.3,
	  })
	})
    const result = await response.text()
	const res = JSON.parse(result)
	console.log(566, res.choices[0].message.content)

}
test()
```


data: {"object": "chat.completion.chunk", "model": "RWKV-paper-reviewer-4-World-7B-20230831-ctx16k", "choices": [{"delta": {"content": "!"}, "index": 0, "finish_reason": null}]}

data: {"object": "chat.completion.chunk", "model": "RWKV-paper-reviewer-4-World-7B-20230831-ctx16k", "choices": [{"delta": {"content": " How"}, "index": 0, "finish_reason": null}]}

123 data: [DONE]

1112 Error [ERR_HTTP_HEADERS_SENT]: Cannot set headers after they are sent to the client
    at new NodeError (node:internal/errors:399:5)
    at ServerResponse.setHeader (node:_http_outgoing:645:11)
    at ServerResponse.header (D:\vue-work\openai_server\node_modules\_express@4.18.2@express\lib\response.js:794:10)
    at ServerResponse.send (D:\vue-work\openai_server\node_modules\_express@4.18.2@express\lib\response.js:174:12)
    at ServerResponse.json (D:\vue-work\openai_server\node_modules\_express@4.18.2@express\lib\response.js:278:15)
    at ServerResponse.send (D:\vue-work\openai_server\node_modules\_express@4.18.2@express\lib\response.js:162:21)
    at file:///D:/vue-work/openai_server/aiServer_v5_test.js:232:28
    at process.processTicksAndRejections (node:internal/process/task_queues:95:5) {
  code: 'ERR_HTTP_HEADERS_SENT'
}

RWKV-novel-4-World-7B-20230810-ctx128k