## Claude


[应用](https://claude.ai)
[开发平台](https://platform.claude.com)
[文档](https://platform.claude.com/docs/en/api/sdks/typescript)




## 案例js
```js
npm install @anthropic-ai/sdk
const client = new Anthropic({
  apiKey: process.env["ANTHROPIC_API_KEY"] // This is the default and can be omitted
});

const message = await client.messages.create({
  max_tokens: 1024,
  messages: [{ role: "user", content: "Hello, Claude" }],
  model: "claude-opus-4-7"
});

console.log(message.content);

// 兼容OpenAI
import OpenAI from "openai"

const openai = new OpenAI({
  apiKey: "ANTHROPIC_API_KEY", // Your Claude API key
  baseURL: "https://api.anthropic.com/v1/" // Claude API endpoint
})

const response = await openai.chat.completions.create({
  messages: [{ role: "user", content: "Who are you?" }],
  model: "claude-opus-4-7" // Claude model name
})

console.log(response.choices[0].message.content)
```