# <center>Stability</center>

## 下载与资源
[官网 ｜](https://stability.ai/) 
[文档 ｜](https://platform.stability.ai)
[API参数 ｝](./parameter.md)
[REST-API |](https://api.stability.ai/docs)

## API
```js
import pkg from 'stability-client'
const { generateAsync } = pkg
// import fs from 'fs'

let API_KEY = "sk-xxxxxxsA42f"  

// let prompt = `In a cyberpunk city, modern art`  只能英文
let prompt = "A Yuezhou kiln dish designed with Plum Blossom abstract patterns,Rococo "

async function main(){
    try {
        const { res, images } = await generateAsync({
          prompt: prompt,
          apiKey: API_KEY,
          engine: 'stable-diffusion-768-v2-1', 
          outDir: './images',
        })
        console.log(111, res)
        console.log(123, images)
      } catch (e) {
        console.log(444, e)
      }

}
main()
```



