
```js
//"stability-client": "^1.8.0"
//必要参数
type RequiredStabilityOptions = {
  apiKey: string
  prompt: string
}
//可选参数
type DraftStabilityOptions = Partial<{
  outDir: string
  debug: boolean
  requestId: string
  samples: number
  engine: string
  host: string
  seed: number | Array<number>
  width: number
  height: number
  diffusion: keyof typeof diffusionMap
  steps: number
  cfgScale: number
  noStore: boolean
  imagePrompt: {
    mime: string
    content: Buffer
    mask?: { mime: string; content: Buffer }
  } | null
  stepSchedule: { start?: number; end?: number }
}>

//不保存在本地
noStore: true in the options

//图生图
let imageStrength = 0.2
imagePrompt: {
    mime: "",
    content: fs.readFileSync('./images/test.png')
},
stepSchedule: {
    start: imageStrength,
    end: 1 - imageStrength
}    

engine:
// stable-diffusion-v1
// stable-diffusion-v1-5
// stable-diffusion-512-v2-0
// stable-diffusion-768-v2-0
// stable-diffusion-512-v2-1
// stable-diffusion-768-v2-1
// stable-inpainting-v1-0
// stable-inpainting-512-v2-0
```