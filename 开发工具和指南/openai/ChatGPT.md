# <center>ChatGPT</center>

## 下载与资源
[官网 |](https://chat.openai.com/)
[API说明 ｜](https://openai.com/blog/introducing-chatgpt-and-whisper-apis)
[ChatGPT API |](https://github.com/transitive-bullshit/chatgpt-api)
[微信聊天机器人 |](https://github.com/zhayujie/chatgpt-on-wechat)
[代入角色(英) ｜](https://github.com/f/awesome-chatgpt-prompts)
[ChatGPT API使用 ｜](https://zhuanlan.zhihu.com/p/610810300)

## 登录问题
[Cloudflare Warp解除IP的限制](https://blog.larkneer.com/trend/@lemooljiang/7gt4ukb8)

## ChatGPT API
```js
const response = await openai.createChatCompletion({
    model: "gpt-3.5-turbo",
    messages: [{role: "user", content: "Hello world"}],
    max_tokens: 1500,
    temperature: 0.2
  })
console.log(156, "gpt", response.data.choices[0].message)

//parameters
model="gpt-3.5-turbo",
messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Who won the world series in 2020?"},
      {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
      {"role": "user", "content": "Where was it played?"}
  ]

//curl调用
this.axios.request({
  method: 'post',
  url: 'https://api.openai.com/v1/chat/completions',
  headers: {
    'X-Requested-With': 'XMLHttpRequest',
    'Authorization': 'Bearer your openai api key'
    },
  data:{
    model: "gpt-3.5-turbo",
    messages: [{"role": "user", "content": this.prompt}]
  }
})
.then(arg => {
  this.answer = JSON.parse(arg.request.response).choices[0].message.content
})
```

## 创建上下文
post提交数据的时候, 把之前所有的对话都提交上去, AI就会读取上传的内容, 以此形成上下文语境。
```js
messages: [{"role": "user", "content": this.prompt}] 
//只有一条是没有上下文环境的

messages =  [
	{ 'role': 'user', 'content': '你好。 今天是多云天气' },
	{ 'role': 'assistant', 'content': '你好。 我很抱歉听到不幸的天气' },
	{ 'role': 'user', 'content': '是的，是这样。 不过我很好。' },
	{ 'role': 'assistant', 'content': '我希望如此。 让我们继续今天的工作吧！' },
	{ 'role': 'user', 'content': '是的。 哦，顺便问一下，我是怎么说今天的天气的？' },
	{ 'role': 'assistant', 'content': '今天是阴天' }
] 
//把以前的对话数据组成数组格式，传给ChatGPT即可！


result = response["choices"][0]["text"].strip()
last_result = result
turns += [question] + [result]  # 只有这样迭代才能连续提问理解上下文

// 取最近的10轮对话
if len(turns) <= 10:  # 为了防止超过字数限制程序会爆掉，所以提交的话轮语境为10次。
    text = " ".join(turns)
else:
    text = " ".join(turns[-10:])
// 限制输入的token


//text-davinci-003模型的参数格式略有不同。它只能传入字符串。
const response = await openai.createCompletion({
    model: "text-davinci-003",
    prompt: `${prompt}`,
    temperature: temperature, 
    max_tokens: 4000, 
    top_p: 1, 
    frequency_penalty: 0, 
    presence_penalty: 0, 
})
messages =  [
	{ 'role': 'user', 'content': '你好。 今天是多云天气' },
	{ 'role': 'assistant', 'content': '你好。 我很抱歉听到不幸的天气' },
	{ 'role': 'user', 'content': '是的，是这样。 不过我很好。' },
	{ 'role': 'assistant', 'content': '我希望如此。 让我们继续今天的工作吧！' },
	{ 'role': 'user', 'content': '是的。 哦，顺便问一下，我是怎么说今天的天气的？' },
	{ 'role': 'assistant', 'content': '今天是阴天' }
] 
// 把数组里面的元素用`\n\n`连接
function getPreviousConversationContent(data) {
    let len = data.length
    let arr = [];
    for (var i = 0; i < len; i++) {
      let item = data[i]
      arr.push(item.content)
    }
    console.log(123, arr, "arr")
    return arr.join("\n\n")
  }
let s =  getPreviousConversationContent(messages)  
console.log(s, typeof s)
```