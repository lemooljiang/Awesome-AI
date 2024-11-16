# <center>Midjourney prompt</center>

## 下载与资源
[官网 ｜](https://www.midjourney.com/)
[文档 ｜](https://docs.midjourney.com/docs/quick-start)
[样式参考 ｜](https://github.com/willwulfken/MidJourney-Styles-and-Keywords-Reference)
[艺术风格 |](https://midlibrary.io)
[艺术风格 ｜](https://lib.kalos.art/)
[openjourney ｜](https://replicate.com/prompthero/openjourney/api#run)
[replicate doc ｜](https://replicate.com/docs)
[PromptHero ｜](https://prompthero.com/) 
[关键词助手｜](https://prompt.noonshot.com/midjourney) 
[Huggingface prompt ｜](huggingface.co/spaces/doevent/prompt-generator)
[prompt-generator ｜](https://www.howtoleverageai.com/midjourney-prompt-generator)
[关键词集锦 ｜](https://zhuanlan.zhihu.com/p/614050846)
[商业设计完全教程 ｜](https://u0gp5ergxyk.feishu.cn/wiki/WMPkwaYybi5X1ykqyZHcFPeAnue)


## 关键词格式
![基本格式](https://cdn.document360.io/3040c2b6-fead-4744-a3a9-d56d621c6c7e/Images/Documentation/MJ_Prompt_basic.png)
/imagine prompt 

![关键词格式 ](https://cdn.document360.io/3040c2b6-fead-4744-a3a9-d56d621c6c7e/Images/Documentation/MJ%20Prompt.png)
/imagine url prompt parameters   //可以使用图片作为关键词
eg:  https://s.mj.run/4t043-II22c subway station 

## 基本命令
/imagine prompt 生成图像<br>
/blend 组合，如对两张图片的组合 <br>
/info 用户信息<br>
/settings 设置
/describe 从图片生成提示词（逆向工程）
/subscribe 管理订阅

## 提示词格式
主体 + 视角 + 距离 + 情绪 + 细节 + 光线 + 风格 + 参数
eg: character of monkey in style of nothing see nothing hear nothing speak wearingheadphones listening rap, citylights background, Hyper detailed, hyper realistic, 8k, --ar 9:16 --s 950
猴子的角色，什么都看不见，什么都听不见，戴着耳机听说唱，城市灯光背景，超细节，超逼真，8k-v 5.1-ar 9:16-s 950

主体内容+气氛灯光色彩+构图+风格参考
eg: A spaceship surrounded by a swarm of small red craft, foggy, top-lit, strongly reflective, wide-angle, ultra-high-definition detail, concept art
一艘宇宙飞船，周围是一群红色的的小型飞行器，雾蒙蒙的，顶光的，强反射的，广角的，超清细节的，概念艺术

## 常用参数
--ar(aspect) 图片的宽高比，eg: --ar 16:9  --ar 5:4<br>
--q (quality) 0.25-5 质量参数，默认为1， 越大质量越高<br>
--s(stylize) 100-1000, 图像的艺术化程度化，默认为100<br>
--c(chaos) 0-100 默认为0, 初始图像的差异化<br>
--v(version) 版本号，在settings中设置了就无需再设置<br>
--seed 连续人物，引用特定画作<br>
  查找种子值： 添加反应， 咖啡杯， 上方输入env,点击信封<br>
--niji 5 动漫模式<br>
--iw 0-2  参考原图的权重，在垫图中使用<br>
--no 要去除的因素，比如 --no white 画面中没有白色<br>
--stop 0-100  在哪个位置时停止出图<br>
--panels 连续动作 <br>
--tile 可重复拼帖的图像

## 图片参数技巧
1. 先生成一个酷炫的背景， abstract tron legacy light rays
2. 然后图片+主体，https://s.mj.run/Yft37s2rrN0 cyberpunk adorable kitten
  也可以使用两张图片融合（blend）

## 增加文字
现只支持英文，以 "" 括起来，加关键词 --style raw ， eg: "harry Potter --style raw"

## 常用关键词
1.Sticker Design --- 贴纸风格 eg: sticker design of cute girl <br>
  Graphic Design of robot and flowers <br>
  Graphic Design, A phone is rising, surrounded by lights and flowers<br>
  Industrial Design, Hard edge style phone<br>
  Graphic Logo Design,Star-shaped, Simple, line construction<br>
2.“A物体”As“B人物”  --- 角色替换 eg: elon mask as a commander<br>
3.Symmetrical,flat icon design --- 简洁，对称LOGO设计 eg: lemon, Symmetrical,flat icon design <br>
4.Game sheet of --- 游戏装备列表 eg: game sheet of gens <br>
5.Knolling --- 将相关联的物品以平行或是 90 度排放的组织方式  eg: knolling tool set， knolling tool fruits<br>
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
17.Photorealistic --- 照片级真实设计 eg: a red car in forest, Photorealistic 
18.Stained glass window --- 被物体覆盖的窗户 eg: flower Stained glass window 
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
用 `|`、`::` 分割关键词，例如： hot ｜ dog， hot:: dog
它会接受为是两个关键词，而不是理解来一个词"hot dog" .

比重：
还可以设置关键词的比重，1-4，默认是1，比如： 
  hot::2 dog 
  shopping mall::12  by teamlab::30

## Remixing
偏重随机的艺术效果
/prefer remix  或 /settings Remix 打开或关闭


## 艺术家风格
style of __ 
  凡高 van gogh
  张小刚 zhang xiaogang
  宫崎骏（日语：宮﨑 駿／みやざき はやお Miyazaki Hayao
  村上隆(日语 村上 隆 平假名 むらかみ たかし 罗马字 Murakami Takashi<br>
  草间弥生（Yayoi Kusama）
eg:<br>
A kite flying in the sky, a boy and a girl chasing on the ground, a lot of <br>flowers on the grass, low shrubs,style of Yayoi Kusama<br>
A few shrimps chased by a crab，ink painting style of Murakami Takashi <br>
A few shrimps chased by a crab:: picasso::1.4<br>
A few shrimps chased by a crab, Painted By Andre Masson<br>


## 超级写实主义
[写实主义](https://www.reddit.com/r/midjourney/comments/119mwu2/i_entered_every_fancy_keyword_that_i_knew_of_to/)
```  
prompt:
portrait of an indian village woman in forest in Himachal pradesh, clear facial features, Cinematic, 35mm lens, f/1.8, accent lighting, global illumination --uplight

portrait of an peking girl, clear facial features, Cinematic, 35mm lens, f/1.8, accent lighting, global illumination --uplight 
```

## 灯光
1. 双重曝光: double exposure 
   eg: a beautiful girl and flowers, double exposure
2. 电影级灯光: cinematic lighting
3. 黑暗灯光: dark moody lighting 
   eg: blacklight bridge, dark moody lighting 
4. 童话灯光: fairy light
5. 全息摄影: Holography
6. Rembrandt light	伦勃朗光
7. mood lighting	情绪照明
8. Soft illuminaotion/ soft lights	柔和的照明/柔光

## 视图
Aerial view	鸟瞰图
andala	曼茶罗构图
ultrawide shot	超广角
extreme closeup	极端特写
macroshot	微距拍摄
an expansive view of	广阔的视野
busts	半身像
profile	侧面
symmetrical body	对称的身体
symmetrical face	对称的脸
wide view	广角
bird view 俯	视/鸟瞰
up view	俯视图
front view	正视图
symmetrical	对称
Center the composition	居中构图
symmetrical the composition	对称构图



## 设计风格
1. 字符风格: ASCII art 
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
11. 仙女风: Fairy Kei fashion  
  eg: a beautiful girl in red, Fairy Kei fashion

## 风格头像
图片 + 风格 + 参数， eg:
https://ipfs.ilark.io/ipfs/QmRobehozeey31UW7cCDazoikqTXi1a7aZd2vERvXx3jsm portrait, style of Van Gogh --iw 1.1


## 标志设计
[LOGO设计全案](https://mp.weixin.qq.com/s/8DUXnEy0EYxvx0rw3p9ZXw)

 Logo类型+主体物+风格+艺术家+视角+行业+颜色 

单色黑白字母LOGO
Company logo, flat, clean, simplicity modern, minimalist, vintage, cartoon, geometric, lettermark logo of letter MS
公司LOGO，扁平、干净、简约、现代、极简主义、复古、卡通、几何、字母标记LOGO，字母为MS。

字母主题LOGO
A company logo, stylized capital letter B, playground theme, overall purple color scheme
公司标志，以大写字母B为主题，运动场景，整体紫色调

letter A Logo, Logo design, flat vector, minimalist --s 200
字母A标志，标志设计，平面矢量，极简主义

吉祥物logo
提示词：red fox, side view, Logo design, minimalist, color block, black background
翻译：赤狐，侧视图，标志设计，极简主义，色块，黑色背景

线条logo
提示词：fox, front view, Logo design, minimalist, line, soft, white background
翻译：狐狸，前视图，标志设计，极简主义，线条，柔和，白色背景

手绘logo
提示词：a coffee shop handwriting style Logo, there’s a little bird, Simplicity
翻译：一个咖啡店 手写风格的标志，有一只小鸟，简单

具体图形
提示词： Cloud, Logo design, minimalist Logo, dazzling light, sparkling feeling, gradient colors, psychedelic
翻译：云，标志设计，极简主义标志，耀眼的光线，闪闪发光的感觉，渐变色，迷幻色，明亮的颜色

多个图形结合
Logo composed of cloud and game controller, Logo design, minimalist logo, gradient colors
由云朵和游戏手柄组成的标志，标志设计，极简主义标志，渐变色

抽象图形
提示词：simple linear wavy Logo, minimalistic
简单的线性波浪形标志，极简

徽章式
提示词：red fox, front view, Logo design, badge, black background
翻译：赤狐，前视图，标志设计，徽章，黑色背景
提示词： （图像提示）elephant, front view, Logo design, badge,
翻译：大象，正面视图，标志设计，徽章，黑色背景，圆形

行业logo
提示词：The Logo of a coffee shop
提示词：Logo of a technology company

提示词： a logo for a children’s toys brand, simple,vector,by Pablo Picasso
儿童玩具品牌的标志，简单，矢量，毕加索风格


黑白图形LOGO
A simple black and white creative logo drawn with ONLY ONE SINGLE LINE representing a stylised Wolf from profil in a minimalistic oneline tattoo style, 2D, Harmonious, full body 
一个简单的黑白创意标志，用一条线条勾勒出了一个符号化的侧面狼形象，在极简主义的单线纹身风格中，2D，和谐，全身

多色图形LOGO
create very simple logo  for company named MANTE with many colors 
为名为“MANTE”的公司设计非常简单的标志，使用多种颜色。

平面图形LOGO
Logo design, a logo that combines coffee related elements, vector, flat and simple design, with a white background, designed by Alvin Lustig 
标志设计，一个结合咖啡相关元素的标志，矢量图，扁平而简约的设计，白色背景，由阿尔文·拉斯蒂格设计
eg: Logo design, a logo that combines yuezhou kiln related elements, Mainly celadon design and development,vector, flat and simple design, with a white background, designed by Paul Rand 
eg2: Logo design, a logo that combines Hunan Tourism Development Conference related elements, reflecting Yueyang's culture and elements,vector, flat and simple design, with a white background, designed by Paul Rand 


设计师：
保罗·兰德（Paul Rand，1914-1996）
阿尔文·勒斯蒂格（Alvin Lustig，1915-1955）
亚历山大·亚历克斯·斯坦维斯（Alexander Alex Steinweiss，1917-2011）
布拉德伯里·汤普森（Bradbury Thompson，1911-1995）
乔治·契尼（George Tscherny，1924- ）
毕加索风格： Pablo Picasso

## 吉祥物设计
形象生成公式： 类型+主体物+风格+颜色+视角景别+质感灯光  

//动物形象
提示词： three views, front view, side view, back view, cartoon IP, orange cat, full body, standing, clean background, pixar, IP blind box clay material, studio light, octane render, 3D, C4D, blender, hyper quality, UHD, 8K --ar 16:9 --niji 5
翻译： 三视图，前视图，侧视图，后视图，卡通IP，橙色猫，全身，站立，干净的背景，皮克斯，IP, 盲盒黏土材料，工作室灯光，辛烷值渲染，3D，C4D，blender（一款3D软件），超高质量，超高清, 8K

提示词： 3D animation style character design, a cute polar dog cartoon character, --niji 5 --s 120
翻译： 3D动画风格的人物设计，一个可爱的北极犬卡通人物

//动物拟人化
提示词： 3D, chibi, a cute tiger cartoon character, holding a book, wearing an overcoat, front view, blue and pink, pure white background, POP MART style, IP image, advanced natural color matching, cute and colorful, exquisite details, C4D, octane renderer, ultra high definition, perfect lighting, cartoon realism,fun character settings, ray tracing --niji 5 --ar 16:9 --s 120
翻译： 3D，chibi，一个可爱的老虎卡通人物，拿着一本书，穿着大衣，正视图，蓝色和粉红色，纯白背景，泡泡玛特风格， IP形象，高级自然配色，可爱多彩，精致的细节， C4D，辛烷值渲染，超高清，完美照明，卡通现实主义，有趣的角色设置，光线追踪

//古风人物
提示词： a super cute girl, wearing traditional Chinese Hanfu, chibi, dreamy, IP character design, full body,white background, bright color, Pixar style, 3D render, front lighting, high detail, C4D, 8K
翻译： 一个超级可爱的女孩，穿着中国传统汉服，chibi，梦幻，IP人物设计，全身，白底，明亮的颜色，皮克斯风格，3D渲染，正面照明，高细节，C4D，8K


## 表情包
[img url] a young man,emoji pack, multiple poses and expressions,[happy,sad,expectant,laughing,disappointed,surprised,pitiful,aggrieved,despised,embarrassed,unhappy] 2d,meme art,white background --niji 5 --iw 1.1

## 角色一致性
跟之前的风格一致性--sref命名基本一致，--cref。

我们可以使用多个 URL 来混合多张图片中的信息/字符，例如 --cref URL1 URL2（这类似于多张图片或样式提示）。
eg: Someone sitting in the middle of a concert --cref https://s.mj.run/rvDu3RXchC8

eg2: You are sitting in a chair, sipping coffee, with a beautiful robot serving you beside you. Other robots are busy working in the background. The scene is captured from a first-person perspective, showcasing the futuristic environment filled with advanced technology and the elegance of the robot, --cref https://s.mj.run/V2kCoCMTPbo --cw 0

--cw 参数的作用
Midjourney 能够从参考图像中识别的角色属性将与提示混合，从而创建出新的角色。你可以通过使用 --cw N 参数（cref 权重）来控制，其中 N 的值可以从 1 到 100。默认值为 100。
--cw 参数不会改变角色参考图像的强度/影响力，--cw 参数的作用：
--cw 100 的值（默认值）将捕捉整个角色；
--cw 0 的值将仅捕捉面部，大致类似于面部替换，而且你无法关闭面部的参考

## 艺术风格
watercolor， 水彩画
Psychedelic，迷幻风格
Chinese ink style, ink drop，水墨画
graphite sketch，素描
tie dye illustration，扎染风格
Cubist screen print illustration style，立体拼贴风格
black line art work,black and white，黑白色调
in pop art retro comic style，波普艺术
Charcoal drawing，炭笔画
Pointillism,in style of Georges Seurat，点彩画
woodcu print，木刻画
oil painting,brush strokes，油画

