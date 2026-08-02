# <center> OpenCode </center>

## 下载与资源
[官网 |](https://opencode.ai/)
[github |](https://github.com/anomalyco/opencode)



## 简介
OpenCode 是一个开源的 AI 编码代理。它提供终端界面、桌面应用和 IDE 扩展等多种使用方式。


## 安装
```js
npm i -g opencode-ai

// 基本配置
opencode
/connect  配置llm
// 使用 /connect 命令添加提供商的 API 密钥后，凭据会存储在 ~/.local/share/opencode/auth.json 中。
cd /home/opencode
/init
```

## 本地配置Skills
```js
cd /home/opencode
创建目录 .opencode -> skills
将skill包放入此目录即可

//全局目录
~/.config/opencode/skills
```


