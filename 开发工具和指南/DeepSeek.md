## DeepSeek

[DeepSeek](https://www.deepseek.com)

基于Transformer结构，具有670亿参数的规模，性价比很高。


## 案例js
```js
import OpenAI from "openai"

const Deepseek = new OpenAI({
  apiKey: deepseek_key,
  baseURL: "https://api.deepseek.com/v1"
})

async function main(content) {
  const response = await Deepseek.chat.completions.create({
    model: "deepseek-chat",  //deepseek-reasoner
    messages: [
      {"role": "user", "content": content}
    ],
  });
  console.log(6352, response.choices[0].message.content)
}
main("AI技术为何先表现出破坏性")
```