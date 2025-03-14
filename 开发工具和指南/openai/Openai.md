# <center>Openai</center>

## 下载与资源
[官网 |](https://openai.com/)
[npm |](https://www.npmjs.com/package/openai)
[Openai文档 |](https://platform.openai.com/docs/introduction)
[Node.js Library |](https://github.com/openai/openai-node)
[工具指南(python) |](https://github.com/openai/openai-cookbook)   
[手机接码平台 |](https://sms-activate.org/cn)
[开发参考 ｜](https://github.com/adrianhajdin/project_openai_codex)
[微调模型 |](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates)

## Openai API
[参数 ｜](https://platform.openai.com/docs/api-reference/chat/create)
```js
npm install openai --save  //"openai": "^4.16.1"
// npm update openai
npm install express cors --save

import express from 'express'
import cors from 'cors'
import OpenAI from "openai"

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "https://example.com/v1"   //如果希望通过代理来访问, 国内要加上
})
//https://api.openai.com/v1/chat/completions
// 测试：
curl https://example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-Axxxxx" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

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

    const response = await openai.chat.completions.create({
        model: "gpt-3.5-turbo", 
        // messages: query,
        messages: [{"role": "system", "content": "You are a helpful assistant."},
                   {"role": "user", "content": "Who won the world series in 2020?"},
                   {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                   {"role": "user", "content": "Where was it played?"}]
        temperature: 0.2, // Higher values means the model will take more risks.
        max_tokens: 1600, // The maximum number of tokens to generate in the completion. Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
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
//特别注意：这里的max_tokens是指输出的最大token值，不是指模型的max_tokens值！比如gpt-3.5-turbo的max_tokens是4096，但这里的max_tokens却只能填1600！
```

## GPT-4o
GPT-4o (“o” for “omni”) is our most advanced model. It is multimodal (accepting text or image inputs and outputting text), and it has the same high intelligence as GPT-4 Turbo but is much more efficient—it generates text 2x faster and is 50% cheaper. Additionally, GPT-4o has the best vision and performance across non-English languages of any of our models. GPT-4o is available in the OpenAI API to paying customers. Learn how to use GPT-4o in our text generation guide.

GPT-4o （"o "表示 "omni"）是我们最先进的型号。它是多模态的（接受文本或图像输入并输出文本），具有与 GPT-4 Turbo 相同的高智能，但效率更高--生成文本的速度快 2 倍，成本低 50%。此外，GPT-4o 在非英语语言方面的视觉和性能也是我们所有型号中最好的。GPT-4o 可通过 OpenAI API 提供给付费用户。在我们的文本生成指南中了解如何使用 GPT-4o。
```js
import OpenAI from "openai"
import dotEnv from "dotenv"

dotEnv.config()
const apiKey = process.env.API_KEY

const Openai = new OpenAI({
  apiKey: apiKey
})

async function main() {
  const response = await Openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "图片里有什么" },
          {
            type: "image_url",
            image_url: {
              "url": "https://ipfs.ilark.io/ipfs/QmadtZxXPTVS9q2qArZHpZaRjYmF9o5HMxj6Hdgc59dGpR",
            },
          },
        ],
      },
    ],
  });
  console.log(635, response.choices[0])
}
main()
/*635 {
  index: 0,
  message: {
    role: 'assistant',
    content: '图片中展示了一个手工工艺品的工作台面。主要物体是一个带有复杂雕刻的绿色陶瓷花瓶，花瓶高大且装饰华丽。周围还有一些其他陶瓷物品，包括一个盖碗和几只小碗。背景中可以看到一些工 具、材料和一个正在工作的空间。工作台上比较凌乱，显示出这是一个正在进行手工制作的场景。'
  },
  logprobs: null,
  finish_reason: 'stop'
}
*/
```

## python版本
[参考](https://platform.openai.com/docs/api-reference/chat/create?lang=python)
```py
pip install openai  # 0.27.8  1.14.2

from openai import OpenAI
from dotenv import dotenv_values
import sys

env_vars = dotenv_values('.env')

client = OpenAI(
  base_url="https://example.com/v1",
  api_key=env_vars['OPENAI_API_KEY'],
)

def aiChat(query):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages = query,
        temperature = 0.3
    )
    print(6662, completion.choices[0].message.content, file=sys.stderr)
    return completion.choices[0].message.content
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
[ChatGPT流式streaming回复 |](https://juejin.cn/post/7222440107214241829)
[SSE长连接 |](https://blog.csdn.net/m0_46672781/article/details/130296397)
[Stream 流式 |](https://juejin.cn/post/7249286903207641146)
```js
// 服务器端实现
try {
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Connection', 'keep-alive')
  res.flushHeaders()

  const stream = await openai.chat.completions.create({
    model: "gpt-4-1106-preview",
    messages: query,
    max_tokens: 1600,  //这里是指输出的最大tokens，不是指模型的最大tokens
    temperature: temperature,
    stream: true,
  });
  for await (const chunk of stream) {
    // console.log(222,"chunk:", chunk)
    // console.log(126, "chunk:", chunk.choices[0]?.delta?.content)
    // process.stdout.write(chunk.choices[0]?.delta?.content || '');
    let strTemp = chunk.choices[0]?.delta?.content
    if(strTemp != null){
        console.log(658,"strTemp:", strTemp)
        outString += strTemp
        res.write(strTemp)
      }
  }
  console.log(444, "end")
  return res.end()
  } catch (error) {
    console.log(1112, error)
    return res.status(500).send('Something went wrong')
  }
///
{
  id: 'chatcmpl-8IdRXgCjduVw3JVCBGexhqUp5MbsJ',
  object: 'chat.completion.chunk',
  created: 1699452215,
  model: 'gpt-3.5-turbo-1106',
  system_fingerprint: 'fp_eeff13170a',
  choices: [ { index: 0, delta: [Object], finish_reason: null } ]
}
{
  id: 'chatcmpl-8IdRXgCjduVw3JVCBGexhqUp5MbsJ',
  object: 'chat.completion.chunk',
  created: 1699452215,
  model: 'gpt-3.5-turbo-1106',
  system_fingerprint: 'fp_eeff13170a',
  choices: [ { index: 0, delta: [Object], finish_reason: null } ]
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
async function main() {
  let response = await Openai.images.generate({
    model: "dall-e-3",
    prompt: "红衣美女",
    n: 1,
    size: "1024x1024",
    response_format: "b64_json"
  })
  console.log(589, "dalle", response.data)
}
//response_format不设置，则默认返回 url
//response.data[0].url
//response.data[i].b64_json

//参数：
n： integer Optional  Defaults to 1,  Must be between 1 and 10.
size： string  Optional Defaults to 1024x1024,  Must be one of 256x256, 512x512, or 1024x1024.
response_format： string Optional Defaults to url , Must be one of url or b64_json
```

## 图片上传到IPFS
```js
//response.data[0].b64_json
let imgurls = []
let img_length = response.data.length
for(let i = 0; i < img_length; i++){
  let content = Buffer.from(response.data[i].b64_json, 'base64')
  let resX = await ipfs.add(content)
  let imgHash = resX.path
  // console.log(88, resX, 456, imgHash)
  imgurls.push(ipfs_host+imgHash)
}
```

## 读图
图像大模型：GPT-4o, GPT-4o mini
```js
async function main() {
  const response = await Openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "图片里有什么" },
          {
            type: "image_url",
            image_url: {
              "url": "https://ipfs.ilark.io/ipfs/QmadtZxXPTVS9q2qArZHpZaRjYmF9o5HMxj6Hdgc59dGpR",
            },
          },
        ],
      },
    ],
  });
  console.log(65, response.choices[0])
}
```

## 计算图片tokens
向模型提供图片有两种主要方式：通过传递图片链接或在请求中直接传递 base64 编码的图片。

计算成本
与文本输入一样，图片输入也是以代币计量和收费的。一张图片的代币成本由两个因素决定：图片大小和每个 image_url 块上的细节选项。所有带 "细节：低 "选项的图片每张都需要花费 85 个代币。"细节：高 "选项的图片首先会被缩放到 2048 x 2048 的正方形内，并保持长宽比不变。然后，再进行缩放，使图像最短的边长为 768px。最后，我们计算图片由多少个 512px 的正方形组成。每个方块需要花费 170 个代币。最后总计还要加上 85 个代币。

下面是一些演示上述操作的示例。
一张 1024 x 1024 正方形图像的细节：高模式下耗费 765 个代币
1024 小于 2048，因此无需调整初始大小。
最短的边是 1024，因此我们将图像缩小到 768 x 768。
表示图像需要 4 个 512 平方英寸的方块，因此最终令牌成本为 170 * 4 + 85 = 765。

2048 x 4096 图像细节：高模式下需要花费 1105 个代币
我们将图像缩小到 1024 x 2048，以适应 2048 平方英寸的大小。
最短的边是 1024，因此我们进一步缩小到 768 x 1536。
需要 6 个 512px 的磁贴，因此最终的代币成本为 170 * 6 + 85 = 1105。
```js
import url from 'node:url'
import https from 'node:https'
import { imageSize } from 'image-size'

async function getSize(imgUrl){
    const options = url.parse(imgUrl)
    return new Promise(resolve => {
        https.get(options, function (response) {
            const chunks = []
            response
                .on('data', function (chunk) {
                chunks.push(chunk)
                })
                .on('end', function () {
                const buffer = Buffer.concat(chunks)
                console.log(223, imageSize(buffer))
                resolve(imageSize(buffer))
                })
            })
    })
}


async function calImgToken(imgUrl) {
    let size = await getSize(imgUrl)
    console.log(556, size)
    let width = size.width
    let height = size.height
    console.log(669, width, height)
    let newWidth = 768
    let newHeight = 768
    let aspect_ratio = width / height 
    
    //第一步：将图片缩小到2048*2048的范围内
    if (width > 2048 || height > 2048) {
      if (aspect_ratio > 1) {
        newWidth = 2048
        newHeight = parseInt(2048 / aspect_ratio)
      } else {
        newHeight = 2048
        newWidth = parseInt(2048 * aspect_ratio)
      }
    }

    // 第二步：将最短边缩小到768
    if (width >= height && height > 768) {
      newWidth = Math.floor((768 / height) * width)
    } else if (height > width && width > 768) {
      newHeight = Math.floor((768 / width) * height)
    }
    
    // 第三步：计算占几个512*512的方块，并计算出token
    const tiles_width = Math.ceil(newWidth / 512)
    const tiles_height = Math.ceil(newHeight / 512)
    const total_tokens = 85 + 170 * (tiles_width * tiles_height)
    console.log(569, total_tokens)
    return total_tokens
}
// calImgToken(1868, 892)  //1445
calImgToken("https://ipfs.ilark.io/ipfs/QmbwEHigZpVFNWrUJgP6u28t9NhAW2xbqUm7BMAzSEc2ND")
```

## 函数调用
函数调用(Function calling)可以极大地增强大语言模型的功能，可以增强推理效果或进行其他外部操作，包括信息检索、数据库操作、知识图谱搜索与推理、操作系统、触发外部操作等工具调用场景。在设计各类bot或agent时非常有用。

![functioncall.jpg](https://ipfs.ilark.io/ipfs/QmSGU9RvhkxfGLNTznEK9pWhH8fCPSu3qWzuzZvm8RtNxA)

函数调用图示

如上所示，在第二步中，可以提供特定的接口函数（也就是外部API），额外增加特定的数据以增强大语言模型的能力。在特定的工作流中，比如查询天气，查询车次等，大语言模型都能够办到啰！

下面我使用的是OpenAI中的示例，进行测试后，得到的结果和代码，大家可以参考。
```js
import OpenAI from "openai"
import dotEnv from "dotenv"

dotEnv.config()
const apiKey = process.env.API_KEY

const openai = new OpenAI({
  apiKey: apiKey 
})

// Example dummy function hard coded to return the same weather
// In production, this could be your backend API or an external API
function getCurrentWeather(location, unit = "fahrenheit") {
  if (location.toLowerCase().includes("tokyo")) {
    return JSON.stringify({ location: "Tokyo", temperature: "10", unit: "celsius" });
  } else if (location.toLowerCase().includes("san francisco")) {
    return JSON.stringify({ location: "San Francisco", temperature: "72", unit: "fahrenheit" });
  } else if (location.toLowerCase().includes("paris")) {
    return JSON.stringify({ location: "Paris", temperature: "22", unit: "fahrenheit" });
  } else {
    return JSON.stringify({ location, temperature: "unknown" });
  }
}

async function runConversation() {
  // Step 1: send the conversation and available functions to the model
  const messages = [
    { role: "user", content: "What's the weather like in San Francisco, Tokyo, and Paris?" },
  ];

  const tools = [
    {
      type: "function",
      function: {
        name: "get_current_weather",
        description: "Get the current weather in a given location",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city and state, e.g. San Francisco, CA",
            },
            unit: { type: "string", enum: ["celsius", "fahrenheit"] },
          },
          required: ["location"],
        },
      },
    },
  ];


  const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo-0125",
    messages: messages,
    tools: tools,
    tool_choice: "auto", // auto is default, but we'll be explicit
  });
  const responseMessage = response.choices[0].message;
  console.log(111, "response", response)
  /*
  {
    id: 'chatcmpl-90oyNdEDQSxsvpMpwxZ3KeYShovtH',
    object: 'chat.completion',
    created: 1709982967,
    model: 'gpt-3.5-turbo-0125',
    choices: [
      {
        index: 0,
        message: [Object],
        logprobs: null,
        finish_reason: 'tool_calls'
      }
    ],
    usage: { prompt_tokens: 88, completion_tokens: 77, total_tokens: 165 },
    system_fingerprint: 'fp_4f0b692a78'
  }
  */

  console.log(123, "responseMessage", responseMessage)
  /* promptX
  {
  role: 'assistant',
  content: null,
  tool_calls: [
    {
      id: 'call_0QMDlATqYjBjgdySQVCL0PI4',
      type: 'function',
      function: [Object]
    },
    {
      id: 'call_e8ceipN3i2OUiF3Z38qNEEs5',
      type: 'function',
      function: [Object]
    },
    {
      id: 'call_cHlSgOIcsOXwpiOpzHIev6vF',
      type: 'function',
      function: [Object]
    }
  ]
  }
  */

  //console.log(396, "function", responseMessage.tool_calls[0].function)
  /*
  {
    name: 'get_current_weather',
    arguments: '{"location": "San Francisco", "unit": "celsius"}'
  }
  */

  // Step 2: check if the model wanted to call a function
  const toolCalls = responseMessage.tool_calls
  if (responseMessage.tool_calls) {
    // call the function
    // Note: the JSON response may not always be valid; be sure to handle errors
    const availableFunctions = {
      get_current_weather: getCurrentWeather,
    }; // only one function in this example, but you can have multiple
    messages.push(responseMessage); // extend conversation with assistant's reply
    console.log(225,"messages", messages)
    /*
    [{
      role: 'user',
      content: "What's the weather like in San Francisco, Tokyo, and Paris?"
    },
    {
      role: 'assistant',
      content: null,
      tool_calls: [ [Object], [Object], [Object] ]
    }]
    */

    for (const toolCall of toolCalls) {
      const functionName = toolCall.function.name;
      const functionToCall = availableFunctions[functionName];
      const functionArgs = JSON.parse(toolCall.function.arguments);
      const functionResponse = functionToCall(
        functionArgs.location,
        functionArgs.unit
      );
      messages.push({
        tool_call_id: toolCall.id,
        role: "tool",
        name: functionName,
        content: functionResponse,
      }); // extend conversation with function response
    }
    console.log(365,"messages2", messages)
  /*
  [{
    role: 'user',
    content: "What's the weather like in San Francisco, Tokyo, and Paris?"
  },
  {
    role: 'assistant',
    content: null,
    tool_calls: [ [Object], [Object], [Object] ]
  },
  {
    tool_call_id: 'call_0QMDlATqYjBjgdySQVCL0PI4',
    role: 'tool',
    name: 'get_current_weather',
    content: '{"location":"San Francisco","temperature":"72","unit":"fahrenheit"}'
  },
  {
    tool_call_id: 'call_e8ceipN3i2OUiF3Z38qNEEs5',
    role: 'tool',
    name: 'get_current_weather',
    content: '{"location":"Tokyo","temperature":"10","unit":"celsius"}'
  },
  {
    tool_call_id: 'call_cHlSgOIcsOXwpiOpzHIev6vF',
    role: 'tool',
    name: 'get_current_weather',
    content: '{"location":"Paris","temperature":"22","unit":"fahrenheit"}'
  }]
  */
    
    //Step 3: get a new response from the model where it can see the function response
    const secondResponse = await openai.chat.completions.create({
      model: "gpt-3.5-turbo-0125",
      messages: messages,
    }); 
    console.log(569, secondResponse)
    return secondResponse.choices;
  }
}


runConversation().then(console.log).catch(console.error);
```

经过三步之后会得到最终的结果，目前来看效果还不错！**其实函数调用(Function calling)和外挂向量数据库有类似的地方，都是要给大语言模型提供额外的信息以得到更为准确的结果。** 如果你正在开发bot或agent，那就用起来吧，它会使大语言模型的能力得到拓展！


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

GPT-4o
gpt-4o  US$5.00/1Mtokens (Input)  US$15.00/1M tokens(Output)

GPT-4 Turbo
gpt-4-turbo  US$10.00/1Mtokens (Input)  US$30.00/1M tokens(Output)

GPT-3.5 Turbo
gpt-3.5-turbo-0125  US$0.50/1Mtokens (Input)  US$1.50/1M tokens(Output)  1000000
