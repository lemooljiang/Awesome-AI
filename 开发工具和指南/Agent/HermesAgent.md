Hermes Agent

[github](https://github.com/nousresearch/hermes-agent)
[文档](https://hermes-agent.nousresearch.com/docs/)
[中文文档](https://hermesagent.org.cn/)

官方英文文档：https://hermes-agent.nousresearch.com/docs/
中文社区文档：https://hermes-doc.aigc.green/
Awesome 资源：https://github.com/0xNyk/awesome-hermes-agent
官方技能商店：https://hermes-agent.nousresearch.com/docs/skills
生态地图（80+ 工具可视化）：https://hermes-ecosystem.vercel.app
橙皮书（花叔深度解读）：https://www.huasheng.ai/orange-books/hermes-agent/


Harness Engineering
https://walkinglabs.github.io/learn-harness-engineering/zh/

Hermes Agent 是一个开源、自托管、支持长期记忆与 Skills 的 AI Agent。它既能在终端中完成复杂任务，也能通过消息网关在微信、飞书、电报等平台上持续工作。

## 安装
```py
# macOS / Linux / WSL2
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

## docker
docker pull nousresearch/hermes-agent:latest
docker run -v ~/.hermes:/opt/data nousresearch/hermes-agent:latest

# VPS 云服务器部署（适合 24 小时不间断运行）
推荐低配 Ubuntu 22.04 LTS 服务器（单核 CPU + 1GB 内存就够），Hetzner、DigitalOcean 等平台月费约 5 美元。
SSH 登录后，直接粘贴执行上面的本地一键安装脚本即可。
额外 Tips ：部署后用 screen 或 tmux 后台运行，避免 SSH 断开后进程退出。
```

## 常用命令速查（入门必备）：
hermes —— 开启对话模式
hermes model —— 切换/查看模型
hermes tools —— 配置可用工具
hermes gateway —— 启动消息网关
hermes setup —— 重新配置向导
hermes doctor —— 系统诊断
hermes migrate openclaw —— 从 OpenClaw 迁移数据（支持自动导入历史技能和配置）


## 变现路径
代安装与排障服务（闲鱼/小红书接单，远程帮小白部署，一单几十到上百）
工作流定制（帮电商/投研团队写自动抓取、排版脚本，收费几百到上千）
卖现成技能包（打包常用 Markdown Skill，虚拟产品零成本多卖）
做自动化内容账号（定时任务自动搜集新闻、改写发帖，赚流量分成）
做付费数据社群（24h 监控行业数据，自动推送，收包月费）
技能分发赚提成（上传到技能平台，按调用次数分成）
自动化任务脚本（监控打折、重复点击等省时省力）
企业内部工具开发（接入公司数据库/代码库，做外包项目）
做培训和社群陪跑（开小班教安装+高阶用法，提供长期答疑）


