# <center>Midjourney prompt</center>

## 下载与资源
[官网 ｜](https://www.midjourney.com/)
[文档 ｜](https://docs.midjourney.com/docs/quick-start)
[client ｜](https://www.npmjs.com/package/midjourney-client)
[openjourney ｜](https://replicate.com/prompthero/openjourney/api#run)
[openjourney doc ｜](https://replicate.com/docs)
[PromptHero ｜](https://prompthero.com/) 
[Midjourney Prompt ｜](https://prompt.noonshot.com/midjourney) 
[Midjourney Prompt2 ｜](https://github.com/willwulfken/MidJourney-Styles-and-Keywords-Reference)
[Huggingface prompt ｜](huggingface.co/spaces/doevent/prompt-generator)
[prompt-generator ](https://www.howtoleverageai.com/midjourney-prompt-generator)


## 关键词格式
![基本格式](https://cdn.document360.io/3040c2b6-fead-4744-a3a9-d56d621c6c7e/Images/Documentation/MJ_Prompt_basic.png)
/imagine prompt 

![关键词格式 ](https://cdn.document360.io/3040c2b6-fead-4744-a3a9-d56d621c6c7e/Images/Documentation/MJ%20Prompt.png)
/imagine url prompt parameters 

## 参数
/imagine prompt <br>
/blend 组合，如对两张图片的组合 <br>
--ar 16:9  5:4  3:2 纵横比<br>
--chaos 0-100 值越高越有想像力<br>
--seed 连续人物，引用特定画作<br>
  查找种子值： 添加反应， 咖啡杯， 上方输入env,点击信封<br>
--niji 动漫模式<br>

可以使用图片作为关键词  Image Prompting<br>
  先帖图片，再写上关键词<br>

--panels 连续动作 <br>
--niji   动漫风<br>
--ar 5:5 图像比例<br>
--chaos 随机值   <br>
--style 4a /4b<br>

## 关键词格式
1.Sticker Design --- 贴纸风格 eg: sticker design of cute girl <br>
  Graphic Design of robot and flowers <br>
  Graphic Design, A phone is rising, surrounded by lights and flowers<br>
  Industrial Design, Hard edge style phone<br>
  Graphic Logo Design,Star-shaped, Simple, line construction<br>
2.“A物体”As“B人物”  --- 角色替换 eg: elon mask as a commander<br>
3.Symmetrical,flat icon design --- 简洁，对称LOGO设计 eg: lemon, Symmetrical,flat icon design <br>
4.Game sheet of --- 游戏装备列表 eg: game sheet of gens <br>
5.Knolling --- 将相关联的物品以平行或是 90 度排放的组织方式  eg: knolling tool set <br>
6.8-bit, 16-bit  --- 怀旧游戏，像素风 eg: 8-bit game pixel art, star war <br>
7._ out of [material ]  --- 被材质覆盖的物体 eg: castle out of flowers <br>
8.Layered Paper  --- 折纸艺术画风 eg: layered paper sea wave<br>
9.Isometric art  ---  等轴艺术画风 eg: Isometric tower<br>
10.Blacklight ---  黑暗灯光特效 eg: blacklight bridge<br>
11.Naïve art ---  纯朴艺术画风 eg: naive art ant<br>
12.Mascot Logo --- 吉祥物设计 eg: Mascot Logo carribage<br>
13.T-shirt vector --- 服装设计 eg: T-shirt vector dogs and flowers<br>
14.Pattern --- 图案设计 eg: chinese native pattern<br>
15.Tattoo --- 纹身设计 eg: rose tatton design<br>
16.Interior Design，architecture --- 建筑设计 eg: Interior Design， a warm chinese house<br>
17.Photorealistic --- 照片级真实设计 eg: a red car in forest, Photorealistic <br>
18.Stained glass window --- 被物体覆盖的窗户 eg: flower Stained glass window <br>
19.Blender 3D --- 3D效果 eg: a wood horse Blender 3D<br>
20.Explode_____by Nychos -- 爆炸性街头艺术 eg: Explode planet by Nychos

## 分割关键词
用 :: 分割关键词，例如： hot:: dog both concepts are considered separately, creating a picture of a dog that is warm. 


## 艺术家风格
style of __ <br>
  宫崎骏（日语：宮﨑 駿／みやざき はやお Miyazaki Hayao<br>
  村上隆(日语 村上 隆 平假名 むらかみ たかし 罗马字 Murakami Takashi<br>
  草间弥生（Yayoi Kusama）<br>
eg:<br>
A kite flying in the sky, a boy and a girl chasing on the ground, a lot of <br>flowers on the grass, low shrubs,style of Yayoi Kusama<br>
A few shrimps chased by a crab，ink painting style of Murakami Takashi <br>
A few shrimps chased by a crab:: picasso::1.4 --v 4<br>
A few shrimps chased by a crab, Painted By Andre Masson<br>

## client API
```js
npm install midjourney-client --save
import midjourney from 'midjourney-client'

//从关键词生成图片
async function image(){
  let name = "crab"
  // 只能英文
  let prompt = "A few shrimps chased by a crab, Painted By Andre Masson"
  console.log(prompt)
  let response = await midjourney(prompt)
  //可以加上参数
  // let response = await midjourney(prompt, {width: 1024})

  let image_url = response[0]
  console.log(168, "req:", response)
  // downImage(image_url, name)
}

//其它参数
guidance_scale: '7',
//Maximum size is 1024x768 or 768x1024 because of memory limits
// Allowed values:128, 256, 512, 768, 1024
// Default value: 512
width: 512,
height: 512,  
num_inference_steps: 50,
num_outputs: 1, 
seed: null,
num_outputs 1 // Allowed values:1, 4, Default value: 1
```


