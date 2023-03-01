# <center>Openai</center>

## 下载与资源
[官网 |](https://openai.com/)
[Openai文档 |](https://platform.openai.com/docs/introduction)
[手机接码平台 |](https://sms-activate.org/cn)

## 计费说明
$0.0200/1000tokens <br>
1个token是一个英文单词，一个汉字2个token，即￥0.136/500字

图片生成是每张消耗 $0.016 

## prompt
1. 翻译
在段落前加上`Translate this into English: \n\n`，后面再加上你要翻译的段落。你要翻译成哪国语言，就把‘English’换成其它语种，比如'Japanese'、'Chinese'这些。不过因为最近chatgpt爆火，服务器繁忙，翻译过程太费时间，用户体验不是很理想。

2. 编程
在段首加上`用JavaScript实现以下需求：\n\n`，后面接上需求即可。

3. 解释代码
在代码片段后加上`\n\n\"\"\"\n 用中文解释以上代码功能:`即可帮你解读代码。

4. 模拟面试
`为面试"job"创建一张10个问题的清单`, 把job替换成你想面试的职位即可。

5. 续写文本
在需要续写的文本后加上`\n\n 请按照以上风格续写`即可。

6. 摘要功能
在需要生成摘要的文本后加上`\n\nTl;dr`即可。

## Openai API
```js
npm install openai --save
npm install express cors --save

let express = require('express')
let cors = require('cors')
let { Configuration, OpenAIApi } = require('openai')


const configuration = new Configuration({
  apiKey: "your openai api key",
})

const openai = new OpenAIApi(configuration)

const app = express()
app.use(cors())
app.use(express.json())

app.get('/', async (req, res) => {
  res.status(200).send({
    message: 'Hello ilark AI!'
  })
})

//AI的文本功能，模型是text-davinci-003
//temperature 取0-1之间，值越高，相关度越低
app.post('/word', async (req, res) => {
  try {
    const prompt = req.body.prompt
    const temperature = req.body.temperature

    const response = await openai.createCompletion({
      model: "text-davinci-003",
      prompt: `${prompt}`,
      temperature: temperature, // Higher values means the model will take more risks.
      max_tokens: 1500, // The maximum number of tokens to generate in the completion. Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
      top_p: 1, // alternative to sampling with temperature, called nucleus sampling
      frequency_penalty: 0, // Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
      presence_penalty: 0, // Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    });

    res.status(200).send({
      bot: response.data.choices[0].text
    });

  } catch (error) {
    console.error(error)
    res.status(500).send(error || 'Something went wrong');
  }
})

//AI的代码功能，模型是code-davinci-002
app.post('/code', async (req, res) => {
  try {
    const prompt = req.body.prompt;
    const response = await openai.createCompletion({
      model: "code-davinci-002",
      prompt: `${prompt}`,
      temperature: 0,
      max_tokens: 1200,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
      stop: ["\"\"\""],
    });
    res.status(200).send({
      bot: response.data.choices[0].text
    });

  } catch (error) {
    console.error(error)
    res.status(500).send(error || 'Something went wrong');
  }
})

//AI的图片功能，模型是dall-e 2
app.post('/image', async (req, res) => {
  try {
    const prompt = req.body.prompt;
    let response = await openai.createImage({
      prompt: `${prompt}`,
      n: 1,
      size: "512x512",
    });
    res.status(200).send({
      image_url: response.data.data[0].url
    });

  } catch (error) {
    console.error(error)
    res.status(500).send(error || 'Something went wrong');
  }
})

app.listen(6200, () => console.log('AI server started on http://localhost:6200'))
```

## ChatGPT API
[npm ](https://www.npmjs.com/package/chatgpt)
```js
npm install chatgpt --save

import { ChatGPTAPI } from 'chatgpt'

const api = new ChatGPTAPI({
  apiKey: process.env.OPENAI_API_KEY
})

const res = await api.sendMessage('Hello World!')
console.log(res.text)

//If you want to track the conversation, you'll need to pass the parentMessageid and conversationid:
const api = new ChatGPTAPI({ apiKey: process.env.OPENAI_API_KEY })

// send a message and wait for the response
let res = await api.sendMessage('What is OpenAI?')
console.log(res.text)

// send a follow-up
res = await api.sendMessage('Can you expand on that?', {
  conversationId: res.conversationId,
  parentMessageId: res.id
})
console.log(res.text)

// send another follow-up
res = await api.sendMessage('What were we talking about?', {
  conversationId: res.conversationId,
  parentMessageId: res.id
})
console.log(res.text)

//You can add streaming via the onProgress handler:
const res = await api.sendMessage('Write a 500 word essay on frogs.', {
  // print the partial response as the AI is "typing"
  onProgress: (partialResponse) => console.log(partialResponse.text)
})

// print the full text at the end
console.log(res.text)
```

