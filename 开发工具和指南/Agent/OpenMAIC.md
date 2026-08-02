# <center> OpenMAIC </center>

## 下载与资源
[github |](https://github.com/THU-MAIC/OpenMAIC)
[文档 |](https://open.maic.chat/docs/zh-cn/getting-started)
[体验入口 |](http://open.maic.chat)


## 简介
OpenMAIC（Open Multi-Agent Interactive Classroom）是清华团队开源的多智能体互动课堂平台。它以多智能体技术为核心，打造出AI教师、AI同学协同的全自动沉浸式课堂，用科技重构在线学习体验，让枯燥的知识学习变得高效又有趣。

OpenMAIC把任何主题或文档变成一场沉浸式课堂。幻灯、测验、可交互模拟，再加会讲、会板书、会讨论的 AI 教师和同学。快试试OpenMAIC,一起探索未来教学的新范式。


## 安装
Node.js 20 或更高
pnpm 10 或更高（如果没装，先 npm install -g pnpm）
```js
git clone https://github.com/THU-MAIC/OpenMAIC.git
cd OpenMAIC
pnpm install

// 配置
cp .env.example .env.local
打开 .env.local，填入你有 key 的提供方
// 启动开发服务器 http://localhost:3000
pnpm dev

// 生产构建
pnpm build
pnpm start
```


