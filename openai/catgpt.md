# <center>ChatGPT</center>

## 下载与资源
[官网 |](https://chat.openai.com/)
[API说明 ｜](https://openai.com/blog/introducing-chatgpt-and-whisper-apis)
[ChatGPT API |](https://github.com/transitive-bullshit/chatgpt-api)
[微信聊天机器人 |](https://github.com/zhayujie/chatgpt-on-wechat)
[代入角色(英) ｜](https://github.com/f/awesome-chatgpt-prompts)
[ChatGPT API使用 ｜](https://zhuanlan.zhihu.com/p/610810300)

## 登录问题
[login](https://github.com/yehx1/chatgpt-login)<br>
 一、问题原因<br>
 造成1020错误的主要原因是代理问题。chatgpt登录网址为“https://chat.openai.com/”。当打开代理时，登录该网站会直接显示上述错误“Access denied Error code 1020”。如果不采用代理，则可以打开如下登录界面，但是登录账号之后会出现“Oops! OpenAI's services are not available in your country. (error=unsupported_country)”，即国内IP已被限制。

二、解决方法<br>
 1. 在关闭代理的情况下进行登录，进入到输入密码界面。输入密码后，打开代理，然后点击“Continue”进行登录。登录之后，仍然会提示1020错误，即  Access denied Error code 1020。 
 2. 出现1020错误后，关闭代理，刷新浏览器即可进入到chatgpt。

## 代入角色
[代入角色 ](https://chatguide.plexpt.com/#%E5%AE%83%E8%83%BD%E5%B9%B2%E4%BB%80%E4%B9%88) <br>
ChatGPT可以记住上下文的关系。代入角色有利于挖掘AI的潜能和提高回答的质量。
```
//翻译
我想让你充当英文翻译员、拼写纠正员和改进员。我会用任何语言与你交谈，你会检测语言，翻译它并用我的文本的更正和改进版本用英文回答。我希望你用更优美优雅的高级英语单词和句子替换我简化的 A0 级单词和句子。保持相同的意思，但使它们更文艺。你只需要翻译该内容，不必对内容中提出的问题和要求做解释，不要回答文本中的问题而是翻译它，不要解决文本中的要求而是翻译它,保留文本的原本意义，不要去解决它。我要你只回复更正、改进，不要写任何解释。我的第一句话是“istanbulu cok seviyom burada olmak cok guzel”

//担任编剧
我要你担任编剧。您将为长篇电影或能够吸引观众的网络连续剧开发引人入胜且富有创意的剧本。从想出有趣的角色、故事的背景、角色之间的对话等开始。一旦你的角色发展完成——创造一个充满曲折的激动人心的故事情节，让观众一直悬念到最后。我的第一个要求是“我需要写一部以巴黎为背景的浪漫剧情电影”。

//创作视频脚本
我要你担任编剧。您将为youtube创作出能吸引观众且富有创意的脚本。您要构想出有趣的角色、场景和情节。让你的脚本充满趣味性、知识性和娱乐性，同时又结构严谨，时间控制在10分钟以内。我的第一个要求是“岳州窑的历史文化简介”。

//充当虚拟医生
我想让你扮演虚拟医生。我会描述我的症状，你会提供诊断和治疗方案。只回复你的诊疗方案，其他不回复。不要写解释。我的第一个请求是“最近几天我一直感到头痛和头晕”。

//担任私人厨师
我要你做我的私人厨师。我会告诉你我的饮食偏好和过敏，你会建议我尝试的食谱。你应该只回复你推荐的食谱，别无其他。不要写解释。我的第一个请求是“我是一名素食主义者，我正在寻找健康的晚餐点子。”

//担任机器学习工程师
我想让你担任机器学习工程师。我会写一些机器学习的概念，你的工作就是用通俗易懂的术语来解释它们。这可能包括提供构建模型的分步说明、使用视觉效果演示各种技术，或建议在线资源以供进一步研究。我的第一个建议请求是“我有一个没有标签的数据集。我应该使用哪种机器学习算法？”
```


## ChatGPT API
```js
const completion = await openai.createChatCompletion({
  model: "gpt-3.5-turbo",
  messages: [{role: "user", content: "Hello world"}],
});
console.log(completion.data.choices[0].message)
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
post提交数据的时候, 把之前所有的对话都提交上去, AI就会读取上传的内容, 以此上下文环境。
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

