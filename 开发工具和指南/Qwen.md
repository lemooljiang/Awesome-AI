## Qwen 通义千问

[通义千问](https://bailian.console.aliyun.com)

阿里云百炼是一站式大模型开发与应用平台，集成了通义千问及主流第三方模型。它为开发者提供了兼容OpenAI的API及全链路模型服务；同时，也提供可视化应用构建能力，让业务人员能快速创建智能体、知识库问答等AI应用。


## 案例js
```js
import OpenAI from "openai"

const Qwen = new OpenAI({
  apiKey: qwen_key,
  baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
})

async function main(content) {
  const response = await Qwen.chat.completions.create({
    model: "qwen-plus",
    messages: [
      {"role": "user", "content": content}
    ],
  });
  console.log(6352, response.choices[0].message.content)
}
main("AI技术为何先表现出破坏性")
```