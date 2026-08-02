# <center> OpenClaw</center>


## 下载与资源
[官网](https://openclaw.ai/)
[文档](https://docs.openclaw.ai/zh-CN)
[github](https://github.com/openclaw/openclaw)
[clawhub](https://clawhub.ai/)
[openclaw10](https://openclaw101.dev/zh#resources)
[使用指南](https://vercel-deploy-livid-nine.vercel.app/)
[菜鸟](https://www.runoob.com/ai-agent/openclaw-clawdbot-tutorial.html)
[案例](https://github.com/hesamsheikh/awesome-openclaw-usecases) 
[ClawHub](https://github.com/openclaw/clawhub)
[ClawWork](https://github.com/HKUDS/ClawWork) AI员工
[ClawRouter](https://github.com/BlockRunAI/ClawRouter) 降低费用


## 简介
OpenClaw 是 2025 年 11 月发布的开源 AI Agent 框架，由奥地利程序员 Peter Steinberger 创建。2026 年 1 月底开始爆火。在短短几个月内，OpenClaw 星标数量就超过了 Linux，成为了 Github 上获星最多的软件项目。

OpenClaw 的作用，就是通过自然语言控制电脑，完成自动化操作。以前，你需要多种工具和专业技能，才能达到同样的效果。可以预料，几年之内，这类工具将会普及到每个人的手机。OpenClaw 目标是让 AI 不只是给建议，而是直接完成完整工程任务。

而「养龙虾」这个爆梗其实就是指OpenClaw。OpenClaw 正在让 Agent 的创业，进入到 “人人皆可参与” 的时代。当无数的 “养虾人” 带着自己的脑洞和创意，在 OpenClaw 这个开源框架里不断探索，一个个有趣的 Demo开始出现，全新的智能时代的序幕也在拉开。而这个时代的主角，不再是只有大厂和顶尖的技术团队，每一个有想法、敢尝试的人，都能在这场浪潮里，找到属于自己的机会。


## 快速开始
```sh
npm i -g openclaw
openclaw -v //2026.6.11 (e085fa1) 2026.7.1-2 (0790d9f)

#/ubuntu
curl -fsSL https://openclaw.ai/install.sh | bash

#主要入口
openclaw onboard

# 停止
openclaw gateway stop
openclaw gateway --port 18789 --verbose

# web ui
openclaw dashboard
http://127.0.0.1:18789
# 如果是远程服务器：
ssh -N -L 18789:127.0.0.1:18789 <user>@<hostip>
Then open:
http://localhost:18789

后续重新配置：
openclaw configure
如果 ~/.openclaw/openclaw.json 存在，选择保留 / 修改 / 重置。

重启 OpenClaw 服务
openclaw gateway restart

更新
openclaw update

openclaw tui 命令行
ctrl + c 退出
```


## 社区项目
Nanobot：https://github.com/HKUDS/nanobot  简化版， python
NanoClaw：https://github.com/qwibitai/nanoclaw
IronClaw：https://github.com/nearai/ironclaw
Spacebot：https://github.com/spacedriveapp/spacebot
CoPaw：https://github.com/agentscope-ai/CoPaw
TinyClaw：https://github.com/TinyAGI/tinyclaw
ZeroClaw：https://github.com/zeroclaw-labs/zeroclaw
PicoClaw：https://github.com/sipeed/picoclaw
zclaw：https://github.com/tnm/zclaw
RosClaw：https://github.com/PlaiPin/rosclaw    


## 远程连接
如果OpenClaw装在远程服务器上，本地访问可以通过ssh连接
```sh
ssh -N -L 18789:127.0.0.1:18789 user@hostip
然后打开： http://127.0.0.1:18789

隧道建立后，openclaw health 和 openclaw status --deep 会通过 ws://127.0.0.1:18789 访问远程 Gateway 网关。
openclaw gateway status、openclaw gateway health、openclaw gateway probe 和 openclaw gateway call 也可以通过 --url 指定转发后的 URL。
```

## 自定义域名访问
通过ssh连接使用有些麻烦，每次都要登录。那么，可以直接域名访问、像个网站一样随时使用么？答案是肯定的。

需要三步来完成域名访问OpenClaw。
1. 第一步 域名解析和Nginx反向代理
a. 域名解析
域名（比如是 example.com）解析到远程服务器上，使用certbot安装ssl证书（`certbot --nginx `）
b. Nginx反向代理
修改 example.conf 文件（/etc/nginx/conf.d）
```sh
server_name  example.com;

gzip on;
gzip_buffers 4 16k;
gzip_comp_level 6;
gzip_vary on;
# 限制上传文件大小 如果需要上传大文件或模型 建议调大
client_max_body_size 50m;

location / {
  proxy_pass http://127.0.0.1:18789;
  proxy_set_header   Host             $host;
  proxy_set_header   X-Real-IP        $remote_addr;
  proxy_set_header   X-Forwarded-For  $proxy_add_x_forwarded_for;
  proxy_http_version 1.1;

  # 【核心】支持 WebSocket 连接 防止AI对话时流式传输或长连接中断
  proxy_set_header Upgrade $http_upgrade;
  proxy_set_header Connection "upgrade";

  # 超时设置（防止大模型生成回答时间过长导致 Nginx 报 504 超时）
  proxy_read_timeout 600s;
  proxy_send_timeout 600s;
}
```
2. 第二步 修改openclaw.json
在远程服务器上打开 OpenClaw 配置文件（通常位于 ~/.openclaw/openclaw.json），有两个位置需要修改。
```sh
Loopback 监听模式（即仅绑定在 127.0.0.1 / localhost 上）修改成 lan
"gateway": {
    "bind": "lan"
 }

controlUi 添加allowedOrigins：
"controlUi": {
  "allowedOrigins": [
    "http://<hostip>:18789",
    "https://example.com"
  ]
}
```
配置好后重启 OpenClaw 服务， `openclaw gateway restart`
3. 设备授权（首次连接）
首次连接OpenClaw时需要对其授权
```sh
openclaw devices list (查看pending状态的)
openclaw devices approve <requestId> 
eg: openclaw devices approve bbd82409-f177-4dc2-xxxx-axxxxx
```
附： 最主要的问题
`origin not allowed (open the Control UI from the gateway host or allow it in gateway.controlUi.allowedOrigins)`
把它丢给gemini，按它的提示一步步修改直到成功。


## Skill加载顺序
OpenClaw 按以下来源加载，优先级从高到低。当多个位置出现同名 Skill 时，优先级最高的来源生效。
```js
1 — 最高 	工作区Skills 	<workspace>/skills
2 	项目智能体Skills 	<workspace>/.agents/skills
3 	个人智能体Skills 	~/.agents/skills
4 	托管/本 Skills 	~/.openclaw/skills
5 	内置Skills 	随安装包提供
6 — 最低 	额外目录 	skills.load.extraDirs + 插件 Skills
Skill 根目录支持分组布局。只要 SKILL.md 出现在已配置根目录下的任意位置（最多深入 6 层），OpenClaw 就会发现该 Skill

在对话框中 /skill （或skill关键词）即可显示已安装的skills，选中后即可指定特定的skill干活！
```