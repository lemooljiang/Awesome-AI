# <center>HuggingFace</center>

## 下载与资源
[官网 ｜](https://huggingface.co)
[API ](https://huggingface.co/docs/api-inference/index)
[Dashboard ](https://api-inference.huggingface.co/dashboard/usage)

## API
```js
cnpm install node-fetch --save

import fetch from "node-fetch"

//https://api-inference.huggingface.co/models/<MODEL_ID>
//stabilityai/stable-diffusion-2-1
//succinctly/text2image-prompt-generator
//prompthero/openjourney
async function query(data) {
  const response = await fetch(
      'https://api-inference.huggingface.co/models/succinctly/text2image-prompt-generator',
      {
          headers: { Authorization: `Bearer ${API_TOKEN}` },
          method: "POST",
          body: JSON.stringify(data),
      }
  )
  const result = await response.json();
  console.log(22, result[0].generated_text)
  // return result;
}
query(prompt)

//axios的写法
async function query(prompt){
  let res = await axios.request({
    url: 'https://api-inference.huggingface.co/models/succinctly/text2image-prompt-generator',
    headers: { Authorization: `Bearer ${API_TOKEN}` },
    method: "POST",
    data: JSON.stringify(prompt),
  })

  // console.log(123, res) 
  console.log(123, res.data[0].generated_text) 
}

query(prompt)
```