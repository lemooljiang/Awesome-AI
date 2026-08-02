# <center> Agent Skills </center>

## 下载与资源
[AgentSkill](https://agentskills.io) Anthropic Skills 的开放标准
[claude skills](https://github.com/ComposioHQ/awesome-claude-skills)
[Anthropics Skills](https://github.com/anthropics/skills)
[ClawHub](https://clawhub.ai) OpenClaw 的公共 Skills 注册中心
[Skillsmp](https://skillsmp.com) Skills Marketplace
[skill0](https://skill0.io) 国内特赞团队（atypica.ai）推出的 Skills 市场，目前收录了 423 个 skills
[女娲.skill](https://github.com/alchaincyf/nuwa-skill) 女娲帮你蒸馏任何人的思维方式，让乔布斯、马斯克、芒格、费曼都给你打工
[dot-skill](https://github.com/titanwings/colleague-skill) 不止同事，任何人都能蒸馏成 Skill


## Skills 本质：Agent 的「工作手册」
如果把 Agent 比作一个聪明但缺乏上下文的实习生，那么 Skill 就是交给它的 标准作业流程（SOP）——固化编码规范、统一输出风格、限制行为逻辑。
Skills 本质上就是教 AI 按固定流程做事的操作说明书，一旦写好，就能像函数一样反复调用。
核心定义：Skill 是 Agent 原生内置的强制约束系统，将「流程配方 + 工具 + 材料」（instruction、references、scripts、assets）打包成一个技能文件夹。唯一合法载体是 SKILL.md。


## 与其他概念的关系
Prompt、Tool、MCP, 它们本质上是不同层次的能力：
- Prompt：单次对话指令，用完即弃
- Tool：单一功能调用（如搜索、计算器）
- MCP：外部工具调用协议，关注「怎么连接」
- Skill：封装完整任务执行方法论，关注「怎么做好一件事」

## 三层渐进式披露架构
Skill 最精妙的设计是 渐进式披露（Progressive Disclosure）——信息分层加载，极大节省 Token 消耗。
Level 1 — 元信息层（Metadata）：name + description，始终在上下文中，约 1-3 行。
Level 2 — 指令层（Instruction）：SKILL.md 正文，触发技能时加载。决定「什么时候该用这个技能」「具体怎么做」。
Level 3 — 资源层（Resources）：references/、scripts/、assets/ 等文件，Agent 判断需要时才读取。Scripts 只执行不读取内容，完全不计入 Token。

## 强制目录结构规范
目录结构极其严格，差一个字符都不生效

![skill2.jpg](https://img.ilark.io/img/202601/1784013517495-x6e9x.jpg)


## 结构详解
每个 Skill 就是一个包含 SKILL.md 的文件夹。SKILL.md 由两部分组成：
YAML 元数据区（必填）：文件顶部的 --- 分隔块，包含 name（技能标识）和 description（一句话描述）。这两个字段决定了 Agent 何时触发此 Skill。
正文区（Markdown）：YAML 之后的全部内容，完整的 Markdown 格式，支持标题、列表、代码块、表格等所有 Markdown 语法，描述具体执行规则。
```js
---
name: my-skill
description: "需要做 XXX 时使用此技能"
---
　
# 技能名称
## 触发条件
- 用户提到 XXX
## 执行流程
1. 第一步：检查环境
2. 第二步：执行核心逻辑
3. 第三步：输出结果
## 输出规范
- 格式要求
- 风格要求
## 可用资源
- scripts/generate.py — 生成脚本
- references/api-docs.md — API 参考文档
- assets/logo.png — 品牌 Logo
```

## 使用
一般Agent可以直接召唤使用，相当方便简洁。这里以OpenClaw WebUI为例简单演示下。
