# <center>Stability</center>

## 下载与资源
[官网 ｜](https://stability.ai/) 
[文档 ｜](https://platform.stability.ai)
[API参数 ｝](./parameter.md)
[REST-API |](https://api.stability.ai/docs)

## REST API
[文档](https://platform.stability.ai/rest-api)
```js
import fetch from "node-fetch"

let url = 'https://api.stability.ai'
let API_KEY = "sk-lx8xxxxxxxxxx"
const response = await fetch(
    url+'/v1/engines/list',
    {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Authorization': `Bearer ${API_KEY}`,
        },
        // method: "POST",
        method: 'GET',
        // body: JSON.stringify(queryData),
    }
)
const result = await response.json()  //result.artifacts[0].base64
console.log(33, result)
}

// 用户
const url = `${apiHost}/v1/user/account`
const url = `${apiHost}/v1/user/balance`
//模型
const url = `${apiHost}/v1/engines/list`

//生成
`${apiHost}/v1/generation/${engineId}/text-to-image`,
https://api.stability.ai/v1/generation/{engine_id}/text-to-image

let engineId = 'stable-diffusion-512-v2-0'
let prompt = "A Yuezhou kiln dish designed with Plum Blossom abstract patterns,Rococo "

async function query() {
  let engineId = 'stable-diffusion-768-v2-1'
  const response = await fetch(
      `${url}/v1/generation/${engineId}/text-to-image`,
      {
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': `Bearer ${API_KEY}`,
          },
          method: "POST",
          // method: 'GET',
          body: JSON.stringify({
            text_prompts: [
              {
                text: prompt,
              },
            ],
            cfg_scale: 7,
            clip_guidance_preset: 'FAST_BLUE',
            height: 768,
            width: 768,
            samples: 1,
            steps: 30,
          })
      }
  )
  const result = await response.json()  //result.artifacts[0].base64
  // console.log(33, result)

  result.artifacts.forEach((image, index) => {
    let d = new Date()
    let time = d.getTime()
    fs.writeFileSync(
      `./out/v1_txt2img_${index}_${time}.png`,
      Buffer.from(image.base64, 'base64')
    )
    console.log(166, index, "generate ok")
  })
}

//参数：
engine_id： string Example: stable-diffusion-v1-5
  stable-diffusion-v1
  stable-diffusion-v1-5
  stable-diffusion-512-v2-0
  stable-diffusion-768-v2-0
  stable-diffusion-512-v2-1
  stable-diffusion-768-v2-1
  stable-inpainting-v1-0
  stable-inpainting-512-v2-0
height：Default: 512
width： Default: 512  
  For 768 engines:   589,824 ≤ height * width ≤ 1,048,576
  All other engines: 262,144 ≤ height * width ≤ 1,048,576
text_prompts: An array of text prompts to use for generation.
samples: nteger (Samples) [ 1 .. 10 ], Default: 1,  Number of images to generate  
seed: 	integer (Seed) [ 0 .. 4294967295 ], Default: 0, Random noise seed (omit this option or use 0 for a random seed)
```

## 与IPFS结合
```js
import fs from 'fs'
import fetch from "node-fetch"
import {create} from 'ipfs-http-client'

const ipfs = create({ host: 'example.io', port: '5002', protocol: 'https' })
const API_KEY = process.env.STABILITY_API_KEY
const stability_url = 'https://api.stability.ai'
const engineId = 'stable-diffusion-768-v2-1'
const ipfs_host = "https://example.io/ipfs/"

async function stability(prompt, id, username, num){
  const response = await fetch(
    `${stability_url}/v1/generation/${engineId}/text-to-image`,
    {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Authorization': `Bearer ${API_KEY}`,
        },
        method: "POST",
        body: JSON.stringify({
          text_prompts: [
            {
              text: prompt,
            },
          ],
          cfg_scale: 7,
          clip_guidance_preset: 'FAST_BLUE',
          height: 768,
          width: 768,
          samples: num,
          steps: 30,
        })
    }
  )
  const result = await response.json()  //result.artifacts[0].base64
  //多张图片
  let imgurls = []
  let img_length = result.artifacts.length
  for(let i = 0; i < img_length; i++){
    let content = Buffer.from(result.artifacts[i].base64, 'base64')
    let resX = await ipfs.add(content)
    let imgHash = resX.path
    // console.log(88, resX, 456, imgHash)
    imgurls.push(ipfs_host+imgHash)
  }
  return imgurls
}
```




