# <center>Midjourney prompt</center>

## 下载与资源
[官网 ｜](https://www.midjourney.com/)
[文档 ｜](https://docs.midjourney.com/docs/quick-start)
[样式参考 ｜](https://github.com/willwulfken/MidJourney-Styles-and-Keywords-Reference)
[openjourney ｜](https://replicate.com/prompthero/openjourney/api#run)
[openjourney doc ｜](https://replicate.com/docs)
[PromptHero ｜](https://prompthero.com/) 
[关键词助手｜](https://prompt.noonshot.com/midjourney) 
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
  先帖图片，再写上关键词, <br>
  eg:https://s.mj.run/4t043-II22c subway station 

--panels 连续动作 <br>
--niji   动漫风<br>
--ar 5:5 图像比例<br>
--chaos 随机值   <br>
--style 4a /4b<br>

## 图片参数技巧
1. 先生成一个酷炫的背景， abstract tron legacy light rays
2. 然后图片+主体，https://s.mj.run/Yft37s2rrN0 cyberpunk adorable kitten
  也可以使用两张图片融合（blend）

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
12.Mascot Logo --- 吉祥物设计 <br>
  eg: Mascot Logo carribage<br>
      A Yuezhou kiln, Mascot Logo<br>
13.T-shirt vector --- 服装设计 eg: T-shirt vector dogs and flowers<br>
14.Pattern --- 图案设计  <br>
  eg: chinese native pattern<br>
  A Yuezhou kiln dish designed with Chinese cloud abstract patterns <br>
  A Yuezhou kiln dish designed with fish abstract patterns
15.Tattoo --- 纹身设计 eg: rose tatton design<br>
16.Interior Design，architecture --- 建筑设计 eg: Interior Design， a warm chinese house<br>
17.Photorealistic --- 照片级真实设计 eg: a red car in forest, Photorealistic <br>
18.Stained glass window --- 被物体覆盖的窗户 eg: flower Stained glass window <br>
19.Blender 3D --- 3D效果 eg: a wood horse Blender 3D<br>
20.Explode_____by Nychos -- 爆炸性街头艺术 eg: Explode planet by Nychos
21.logo for __
  Logo design for a copany called H.M modern design
  Design a logo for travel agency, in the style of paul rand
  Piza, symmertical flat icon design
  Yuezhou kiln, simple, as a logo
22.Long Exposure 长时间曝光
23.POV （point of view）第一人称视角
24.Elegant, 5000s 
 eg: a beautiful girl, Elegant, 5000s,front view --ar 16:9 
25.Art Nouveau 新艺术   Rococo洛可可


## 分割关键词
用 :: 分割关键词，例如： hot:: dog 
它会接受为是两个关键词，而不是理解来一个词"hot dog" .

比重：
还可以设置关键词的比重，比如： 
  hot::2 dog 
  shopping mall::12  by teamlab::30

## Remixing
偏重随机的艺术效果
/prefer remix  或 /setting Remix setting打开或关闭


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


## 超级写实主义
[写实主义](https://www.reddit.com/r/midjourney/comments/119mwu2/i_entered_every_fancy_keyword_that_i_knew_of_to/)
```  
prompt:
portrait of an indian village woman in forest in Himachal pradesh, clear facial features, Cinematic, 35mm lens, f/1.8, accent lighting, global illumination --uplight --v 4

portrait of an peking girl, clear facial features, Cinematic, 35mm lens, f/1.8, accent lighting, global illumination --uplight --v 4
```

## 灯光
1. 双重曝光: double exposure 
   eg: a beautiful girl and flowers, double exposure
2. 电影级灯光: cinematic lighting
3. 黑暗灯光: dark moody lighting 
   eg: blacklight bridge, dark moody lighting 
4. 童话灯光: fairy light
5. 全息摄影: Holography

## 设计风格
1. 字符风格: ASCII art <br>
  eg: ASCII art, a girl face ,front view
2. 拼接艺术: collage art
3. 欧普艺术(视幻艺术): Op art 
  eg:a beautiful girl face, Op art 
4. 怀旧风格：synthwave
 eg: a beautiful woman, Smiling, front view, synthwave
5. 80年代风格: Retrowave  eg: a beautiful woman, Smiling, Retrowave 
6. 水彩画风格: Watercolor sketch of a boy
 eg: Watercolor sketch of peking street
7. 橡皮泥风格: Plasticine
  eg:Plasticine of a cat
8. 引用灵感: inspired
9. 拟人化: Anthropomorphize
10. 景泰蓝: Cloisonnism
   eg:Cloisonnism style porcelain of a bottle
      Yuezhou Kiln of a bottle
11. 仙女风: Fairy Kei fashion  <br> 
  eg: a beautiful girl in red, Fairy Kei fashion
eg:<br> 
masterpiece, anime girl in the rainy day, aesthetic, transparent colorful vinyl jacket, highly detailed, reflections --ar 2:3<br> 
deep purple nissan gtr in a neon city at night, shot with a sony mirrorless, 35mm, photography, cinematic, anti-aliasing, CGI --ar 3:2 <br> 
cinematic shot from cyberpunk movie by Pedro Almodovar --ar 2:1

