# <center>Openai</center>

## 下载与资源
[官网 |](https://openai.com/)
[Openai文档 |](https://platform.openai.com/docs/introduction)
[Node.js Library |](https://github.com/openai/openai-node)
[工具指南(python) |](https://github.com/openai/openai-cookbook)   
[手机接码平台 |](https://sms-activate.org/cn)
[开发参考 ｜](https://github.com/adrianhajdin/project_openai_codex)
[微调模型 |](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates)

## Openai API
[参数 ｜](https://platform.openai.com/docs/api-reference/chat/create)
```js
npm install openai --save  //"openai": "^3.2.1"
npm install express cors --save

import express from 'express'
import cors from 'cors'
import { Configuration, OpenAIApi } from 'openai'


const configuration = new Configuration({
  apiKey: "your openai api key",
})
// 如果希望通过代理来访问, 加上
basePath: "https://example.com/v1"

const openai = new OpenAIApi(configuration)

const app = express()
app.use(cors())
app.use(express.json())

app.get('/', async (req, res) => {
  res.status(200).send({
    message: 'Hello ilark AI!'
  })
})


// 模型是 gpt-3.5-turbo，性价比最高的模型
//temperature 取0-1之间，值越高，相关度越低
app.post('/gpt', async (req, res) => {
  try {
    const prompt = req.body.prompt
    const temperature = req.body.temperature

    const response = await openai.createChatCompletion({
        model: "gpt-3.5-turbo", 
        messages: query,
        temperature: 0.2, // Higher values means the model will take more risks.
        max_tokens: 1500, // The maximum number of tokens to generate in the completion. Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
        top_p: 1, // alternative to sampling with temperature, called nucleus sampling
        frequency_penalty: 0, // Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        presence_penalty: 0, // Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
      })

    console.log(55, response.data.choices[0].message)

    res.status(200).send({
      bot: response.data.choices[0].message
    });

  } catch (error) {
    console.error(error)
    res.status(500).send(error || 'Something went wrong');
  }
})
//特别注意：这里的max_tokens是指输出的最大token值，不是指模型的max_tokens值！比如gpt-3.5-turbo的max_tokens是4096，但这里的max_tokens却只能填1500！
```

## python版本
[参考](https://platform.openai.com/docs/api-reference/chat/create?lang=python)
```py
pip install openai  # 0.27.8

import os
import openai

openai.api_key = "YOUR API-KEY"

# 如果希望通过代理来访问
openai.api_base = "https://example.com/v1"
# os.environ['HTTP_PROXY'] = "xxx"
# os.environ['HTTPS_PROXY'] = "xxx"
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)
print(completion.choices[0].message)

# 流传输
def openaiStreaming(query, temperature):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = query,
        temperature = temperature,
        max_tokens = 2000,
        stream = True
    )
    for trunk in response:
        # print(56, trunk)
        if trunk['choices'][0]['finish_reason'] is not None:
            data = '[DONE]'
            return 'ok', 200
        else:
            data = trunk['choices'][0]['delta'].get('content','')
        # yield "data: %s\n\n" % data.replace("\n","<br>")
        yield data
        # return flask.Response(stream(),mimetype="text/event-stream")
///
56 {
  "id": "chatcmpl-7gPI3f8yPerbCv1HzmC41UY4GpXAB",
  "object": "chat.completion.chunk",
  "created": 1690341347,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u4e3a"
      },
      "finish_reason": null
    }
  ]
}
.........
56 {
  "id": "chatcmpl-7gPI3f8yPerbCv1HzmC41UY4GpXAB",
  "object": "chat.completion.chunk",
  "created": 1690341347,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {},
      "finish_reason": "stop"
    }
  ]
}
```


## Embeddings
[youtube-gpt |](https://github.com/davila7/youtube-gpt)
[Chatbot |](https://github.com/FaustoNisida/Chatbot-Long-Short-Term-Memory)
[chatpdf |](https://github.com/postor/chatpdf-minimal-demo)
[Code a Project like ChatPDF |](https://postor.medium.com/how-to-code-a-project-like-chatpdf-e40441cb4168)
[构建一个智能文档查询 |](https://mp.weixin.qq.com/s/bYxFySJEWPUHd2591jVHEQ)

嵌入（Embeddings）是文本的一种数字表示，可以用来衡量两段文本之间的关系。我们的第二代嵌入模型，text-embedding-ada-002，嵌入对于搜索、聚类、推荐、异常检测和分类任务都很有用。

嵌入是一个浮点数字的向量（列表）。两个向量之间的距离衡量它们的关联性。它可以用于事先对模型的数据准备，也就是更新模型的新数据。例如chatgpt的数据只到2021年，没有后面的数据，或是专门类的数据不够，它就不会得到正确答案。这时，可以先准备新数据的嵌入向量，查询时先通过向量查询出新数据，把新数据一起传入chatgpt以得到正解。所以，嵌入（Embeddings）是给chatgpt投喂新数据。
```js
const { Configuration, OpenAIApi } = require("openai");
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);
const response = await openai.createEmbedding({
  model: "text-embedding-ada-002",
  input: "The food was delicious and the waiter...",
});
//response  response.data.data[0].embedding

eg:
let jsonData = [
  {
    "text": "2022年1月10号到13号，俄罗斯分别与美国和北约开展对话.......",
    "embedding": [
      -0.012431818,
      -0.021277534, ......
      ]
  },
  {
    "text": "美国国防部宣布：将向欧洲增派部队，应对俄乌边境地区的紧张局势.....",
    "embedding": [
      -0.012431818,
      -0.021277534, ......
      ]
  }
]
//查找两段文本的相似度
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i]
      normA += Math.pow(vecA[i], 2)
      normB += Math.pow(vecB[i], 2)
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
}

//从数据集中查找拟合度最高的两段文本
function getSimilarTextFromDb(inputEmbedding, jsonData) {
  let result = []
  jsonData.forEach(embedding => {
      let similarity = cosineSimilarity(inputEmbedding, embedding.embedding)
      // console.log("similarity", similarity)
      if (similarity > 0.8) {
          result.push({
              text: `${embedding.text}`,
              similarity: similarity
          })
      }
  })
  result.sort((a, b) => b.similarity - a.similarity)
  let topTwo = result.slice(0, 2)
  return topTwo.map(r => r.text).join("")
}

//1. 先是查找文本的向量值
//2. 再和预先准备的数据集比对，选出拟合度最高的文本
//3. 选出的文本和用户的关键词拼接，再查询
const prompt = req.body.prompt  
const inputEmbeddingResponse = await openai.createEmbedding({       
  model: "text-embedding-ada-002",
  input: prompt
})
const inputEmbedding = inputEmbeddingResponse.data.data[0].embedding
const context = getSimilarTextFromDb(inputEmbedding, jsonData)
// console.log(289,"getSimilarTextFromDb", context)

let promptX = `我希望你充当俄罗斯和乌克兰的问题专家。我会向您提供有关俄罗斯和乌克兰问题所需的所有信息，而您的职责是用简易和严谨的语言解答我的问题。\n${context}\n${prompt}`
const response = await openai.createCompletion({
  model: "text-davinci-003",
  prompt: promptX,
  temperature: 0.2, 
  max_tokens: 3000, 
  top_p: 1, 
  frequency_penalty: 0, 
  presence_penalty: 0, 
})
```

## streaming
边看边输的方式就是现在的推流(streaming)模式，像现在的视频站、直播等，肯定都是用的数据流，都是边看边播的。ChatGPT API也是可以实现同样的功能的。前端得到第一个字就开始输出，这让用户体验更佳。
[参考 |](https://github.com/openai/openai-node/issues/18)
[node stream |](https://github.com/node-fetch/node-fetch#streams)
```js
// 服务器端实现
try {
    res.setHeader('Cache-Control', 'no-cache')
    res.setHeader('Content-Type', 'text/event-stream')
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Connection', 'keep-alive')
    res.flushHeaders() // flush the headers to establish SSE with client

    const response = openai.createChatCompletion({
      model: "gpt-3.5-turbo",
      messages: query,
      max_tokens: 3000,
      temperature: 0.2,
      stream: true,    //推流模式打开
    }, { responseType: 'stream' })

    response.then(resp => {
      resp.data.on('data', data => {
          const lines = data.toString().split('\n').filter(line => line.trim() !== '')
          for (const line of lines) {
              const message = line.replace(/^data: /, '')
              if (message === '[DONE]') {
                // console.log(996, "done")
                  res.end()
              }
              let strTemp = getReg(message)
              if(strTemp != null){
                // console.log(1188, "strTemp", strTemp)
                res.write(strTemp)
              }
          }
      })
    })
  }
//输出的结果是这样：
data: {"id":"chatcmpl-6tolky9FdPyKCKiXbjF7rcWS1QeaA","object":"chat.completion.chunk","created":1678761216,"model":"gpt-3.5-turbo-0301","choices":[{"delta":{"content":"有"},"index":0,"finish_reason":null}]}
data: {"id":"chatcmpl-6tolky9FdPyKCKiXbjF7rcWS1QeaA","object":"chat.completion.chunk","created":1678761216,"model":"gpt-3.5-turbo-0301","choices":[{"delta":{"content":"关"},"index":0,"finish_reason":null}]}  
//使用正则提取中间的字符串，清洗得到的字符串
function getReg(text){
  //"content":"我是中"},"index"  提取中间的字符串
  let reg = /(?<=\"content\"\:\")[\s\S]*?(?=\"\}\,\"index\")/g
  let a = text.match(reg)
  if(a === null){
    return
  }
  return a.join('')
}

//用户端
let query = [{role: "user", content: data.get('prompt')}]
let dataObj = {
     method: 'POST',
     headers: {
          'Content-Type': 'application/json',
     },
     body: JSON.stringify({
          query
          })
}

const response = await fetch(url, dataObj)
let that = this
messageDiv.innerHTML = " "
if (response.ok) {	
let i = 0
let getStream = function (reader) {
     return reader.read().then(function (result) {
          // 如果数据已经读取完毕，直接返回
          if (result.done) {
               console.log(889, "result done")
               that.clickFlag = false
               clearInterval(loadInterval)  
               loading.textContent = ''
               return
          }
          // 取出本段数据（二进制格式）
          let chunk = result.value
          let text = that.utf8ArrayToStr(chunk)
          if(i === 0){
               text = text.replace(/\\n/g,'')  //去除首段换行
          } else{
               text = text.replace(/\\n/g,'<br/>')
          }
          // console.log(5667, "i", i, text)
          // 将本段数据追加到网页之中
          messageDiv.innerHTML += text
          i ++
          // 递归处理下一段数据
          return getStream(reader)
     })
}
getStream(response.body.getReader())  
```

## count-tokens
[npm ｜](https://www.npmjs.com/package/gpt-3-encoder)
[openai测试 ｜](https://platform.openai.com/tokenizer)

GPT系列模型使用Token处理文本，Token是文本中发现的常见字符序列。这些模型了解这些Token之间的统计关系，并擅长在一个Token序列中产生下一个Token。

一个有用的经验法则是，对于普通英语文本来说，一个Token通常对应于~4个字符的文本。这大约相当于一个单词的3/4, 所以100个Token~=75个英文单词。
以中文统计的话，100个Token~=50个汉字。
```js
cnpm install gpt-3-encoder --save  //"^1.1.4"

import {encode, decode} from 'gpt-3-encoder'

const str = '看了山和海，也看了人山和人海。'
const encoded = encode(str)
console.log(11, 'Encoded this string looks like: ', encoded) 
/*[
    40367,   233, 12859,   228,   161,
    109,   109,   161,   240,   234,
  38184,   115,   171,   120,   234,
  20046,   253, 40367,   233, 12859,
    228, 21689,   161,   109,   109,
    161,   240,   234, 21689, 38184,
    115, 16764
] */
console.log(12, 'Encoded lenght', encoded.length)  //32

console.log('We can look at each token and what it represents')
for(let token of encoded){
  console.log({token, string: decode([token])})
}

const decoded = decode(encoded)
console.log(22, 'We can decode it back into:\n', decoded)
```

## Dall-E
```js
const { Configuration, OpenAIApi } = require("openai");
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration)
let response = await openai.createImage({
  prompt: "A cute baby sea otter",
  n: 1,
  size: "512x512",
  response_format: "b64_json"
})
//response.data.data[0].b64_json

const response = await openai.createImage({
  prompt: "A cute baby sea otter",
  n: 2,
  size: "1024x1024",
})
//response_format不设置，则默认返回 url
//response.data.data[0]..url

//参数：
n： integer Optional  Defaults to 1,  Must be between 1 and 10.
size： string  Optional Defaults to 1024x1024,  Must be one of 256x256, 512x512, or 1024x1024.
response_format： string Optional Defaults to url , Must be one of url or b64_json
```

## 图片上传到IPFS
```js
//response.data.data[0].b64_json
let imgurls = []
let img_length = response.data.data.length
for(let i = 0; i < img_length; i++){
  let content = Buffer.from(response.data.data[i].b64_json, 'base64')
  let resX = await ipfs.add(content)
  let imgHash = resX.path
  // console.log(88, resX, 456, imgHash)
  imgurls.push(ipfs_host+imgHash)
}
```

## 反向代理
```py
# 海外服务器 nginx 
server { 
	server_name  example.com;

    location / {
        proxy_pass  https://api.openai.com/;
        proxy_ssl_server_name on;
        proxy_set_header Host api.openai.com;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

	error_page   500 502 503 504  /50x.html;
	location = /50x.html {
		root   /usr/share/nginx/html;
	}

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot
}

# 用户端 加上一条 api_base 
openai.api_base = "https://example.com/v1"
```

## 计费说明
[pricing](https://openai.com/pricing)
$0.0200/1000tokens(约750英文单词)，1个token对应约4个字符。<br>
图片生成是每张消耗 $0.016 

ChatGPT API<br>
$0.002/1000tokens(约750英文单词)

Whisper API  <br>
$0.006/分钟

