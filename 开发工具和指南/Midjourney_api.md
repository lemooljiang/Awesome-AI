# <center>Midjourney API</center>

## 下载与资源
[npm |](https://www.npmjs.com/package/midjourney)
[github |](https://github.com/erictik/midjourney-api)


## 获取参数
因为midjourney只提供了discord服务，所以这款API也是通过discord来模拟调用的。在使用前要做好前期准备：注册好discord，订阅midjourney，将midjourney bot注册到单独的服务器中。

1. 服务器ID、频道ID
在discord服务器的url地址栏中找到， 如下：
https://discord.com/channels/1073xxx/10738xxx
channels后是服务器ID、频道ID这样的顺序。

2. 获取用户Token
[参考](https://github.com/novicezk/midjourney-proxy/blob/main/docs/discord-params.md)
进入频道，打开network（F12），刷新页面(F5)，找到 messages 的请求，这里的 Authorization 即用户Token

## 安装
npm install midjourney --save  //^4.3.13  4.3.17
//npm update midjourney --save
//或者
yarn add midjourney


## 基本使用
```js
import { Midjourney } from "midjourney"

const client = new Midjourney({
  ServerId: "xxxx",   //服务器ID
  ChannelId: "xxxx",  //频道ID
  SalaiToken: "xxxx", //用户Token
  Debug: false,
  Ws: true,
})

await client.init()
// console.log(266, client)


async function main(){
  try {
    const prompt = "taken with Canon EOS 5D Mark IV, Canon EF 85mm f/1. 2L II USM, ISO100, f/1. 2, shutter speed 1/100; a flaming woman with wings, in the style of realisticyetstylized, dark orange and red, made of all of the above, i can't believe howbeautifulthis is, exaggerated poses, warmcore --ar 2:3"
    //imagine
    const Imagine = await client.Imagine( prompt )
    console.log(669, Imagine)
    if (!Imagine || !Imagine.options) {
      console.log("no message")
      return
    }

    let imgarr = []
    for (let i = 0; i < Imagine.options.length; i++) { 
      let labels = ["U1", "U2", "U3", "U4"]
      if(labels.includes(Imagine.options[i].label)){
        let CustomID = Imagine.options[i].custom
        let Upscale = await client.Custom({
          msgId: Imagine.id,
          flags: Imagine.flags,
          customId: CustomID
        })
        if (!Upscale) {
          console.log("no Upscale")
          continue
        }
        console.log(336,i, "upscale", Upscale.uri)
        imgarr.push(Upscale.uri)
      }
    }

    console.log(866, "imgarr", imgarr)

  } catch (error) {
    console.log(444, "error", error)
  }  
}
main()
```
