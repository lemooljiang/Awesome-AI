## openrouter
[OpenRouter](https://openrouter.ai/docs)

OpenRouter包含了多个大语言模型，国内可以访问。

## 案例python
```py
import openai
from dotenv import dotenv_values
import sys


env_vars = dotenv_values('.env')
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = env_vars['OPENROUTER_API_KEY']

def routerChat(query, temperature):
    completion = openai.ChatCompletion.create(
        model = "openai/gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
           ],
        temperature = temperature,
        headers={ "HTTP-Referer": "https://test.com",
                  "X-Title": "test" }
    )
    # print(6662, completion)
    return completion.choices[0].message.content


def routerStreaming(query, temperature):
    # print(253, "openRouter")
    response = openai.ChatCompletion.create(
        model = "openai/gpt-3.5-turbo-16k",
        messages = query,
        temperature = temperature,
        max_tokens = 2000,
        stream = True,
        headers={ "HTTP-Referer": "https://test.com",
                  "X-Title": "test" }
    )
    for trunk in response:
        if trunk['choices'][0]['finish_reason'] is not None:
            data = '[DONE]'
            return 'ok', 200
        else:
            data = trunk['choices'][0]['delta'].get('content','')
        yield data
```

## 案例js
```js
import fetch from "node-fetch"

const apiKey = "sk-xxxxxxx"

async function test(){
	const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
	  method: 'POST',
	  headers: {
	    'Authorization': 'Bearer ' + apiKey,
	    'HTTP-Referer': "https://test.com", 
	    'X-Title': "test"
	  },
	  body: JSON.stringify({
	    model: "openai/gpt-3.5-turbo", 
	    messages: [
	      {"role": "system", "content": "You are a helpful assistant."},
	      {"role": "user", "content": "hello"}
	    ],
	    stream:true
	  })
	})

	try {
		for await (const chunk of response.body) {
		console.log(566,chunk.toString())
		}
	} catch (err) {
		console.error(err.stack)
	}

}
test()
```