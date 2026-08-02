# <center> Deep Agents </center>


## 下载与资源
[官网 |](https://www.langchain.com/deep-agents)
[文档 |](https://docs.langchain.com/oss/python/deepagents/overview) 
[Github |](https://github.com/langchain-ai/deepagents)
[中文手册 |](https://langchain-doc.cn/v1/python/deepagents/overview.html)
[Deep Agents CLI |](https://docs.langchain.com/oss/python/deepagents/cli/overview)
[案例 |](https://github.com/langchain-ai/deepagents/tree/main/examples)
[Deep Agents 实战 |](https://github.com/datawhalechina/deepagents-in-action)

## 简介
Deep Agents 是一个基于 LangChain 代理核心构建模块构建的独立库。它使用 LangGraph 运行时来实现持久化执行、流式处理、人机协同等特性。

它是一个独立的库，用于构建能够处理复杂、多步骤任务的Agent，具备规划能力、用于上下文管理的文件系统以及派生子Agent的能力。

## 简介
Deep Agents 是一个基于 LangChain 代理核心构建模块构建的独立库。它使用 LangGraph 运行时来实现持久化执行、流式处理、人机协同等特性。

它是一个独立的库，用于构建能够处理复杂、多步骤任务的Agent，具备规划能力、用于上下文管理的文件系统以及派生子Agent的能力。

![deepagent21.jpg](https://img.ilark.io/img/202601/1775911017965-0r218.jpg)

![deepagent22.jpg](https://img.ilark.io/img/202601/1775911073474-xa8nc.jpg)

## 优点
- 处理需要规划和分解的复杂、多步骤任务
- 通过文件系统工具管理大量上下文
- 将工作委托给专门的子Agent以实现上下文隔离
- 在对话和线程之间持久化内存

## 核心功能
- 规划和任务分解
深度Agent包含一个内置的 write_todos 工具，使Agent能够将复杂任务分解为离散的步骤，跟踪进度，并在新信息出现时调整计划。

- 上下文管理
文件系统工具（ls、read_file、write_file、edit_file）允许Agent将大量上下文卸载到内存中，防止上下文窗口溢出，并能够处理可变长度的工具结果。
- 子Agent派生
内置的 task 工具使Agent能够为上下文隔离派生专门的子Agent。这可以保持主Agent的上下文清洁，同时仍能深入处理特定的子任务。
- 长期记忆
使用 LangGraph 的 Store 扩展Agent，使其具有跨线程的持久内存。Agent可以从以前的对话中保存和检索信息。

## 安装
```py
virtualenv deep_env
source deep_env/Scripts/activate   //windows
pip install deepagents -i https://pypi.tuna.tsinghua.edu.cn/simple  // 0.5.2/0.6.12  langchain 1.2.15/ 1.3.14
pip install langchain-openai python-dotenv -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install tavily-python markdownify -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -U langchain deepagents -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 快速开始
```py
from deepagents import create_deep_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_deep_agent(
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

## 深度研究助手
```py
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from typing import Literal
from deepagents import create_deep_agent
import  requests, json


env_vars = dotenv_values('.env')
OPENAI_KEY = env_vars['OPENAI_API_KEY'] 
OPENAI_BASE_URL = env_vars['OPENAI_API_BASE'] 
SERPER_KEY = env_vars['SERPER_KEY'] 

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY,base_url=OPENAI_BASE_URL)


def internet_search(query):
    """Get real-time Internet information"""
    url = "https://google.serper.dev/search"
    payload = json.dumps({
      "q": query,
      "num": 1,
    })
    headers = {
      'X-API-KEY': SERPER_KEY,
      'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, data=payload)
    data = json.loads(response.text)
    print(668,"fetch_real_time_info", data)
    if 'organic' in data:
        return json.dumps(data['organic'],  ensure_ascii=False) 
    else:
        return json.dumps({"error": "No organic results found"},  ensure_ascii=False) 


# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(model=llm, tools=[internet_search], system_prompt=research_instructions)

result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})

# Print the agent's response
print(111, result["messages"][-1].content)
```

## 集成skills
在 Deep Agents 中引入 Anthropic Skills 非常简单，因为 Deep Agents 原生支持 Agent Skills 规范（由 Anthropic 提出并开源的标准）。你不需要任何适配层，直接把符合规范的 Skill 目录传给 create_deep_agent 即可。
```py
my_project/
└── skillsRepo
    └── skills/
        └── pptx/
            ├── SKILL.md    # 技能的核心说明与工作流定义
            └── scripts     # 该技能配套使用的辅助脚本
└── main.py 

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

env_vars = dotenv_values('.env')
OPENAI_KEY = env_vars['OPENAI_API_KEY'] 
OPENAI_BASE_URL = env_vars['OPENAI_API_BASE'] 

llm = ChatOpenAI(model="gpt-5.4-mini", api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)
# 创建后端（从本地文件系统加载 skills）
backend = FilesystemBackend(root_dir="./skillsRepo") 
agent = create_deep_agent(model=llm, backend=backend, skills=["./skills"]) 
# 指向skills目录，或直接指定 ./skills/pptx 

result = agent.invoke({"messages": [{"role": "user", "content": "有哪些skills"}]})
print(111, result["messages"][-1].content)
```

## 文件权限
```py
from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends.filesystem import FilesystemBackend

backend = FilesystemBackend(root_dir="./skillsRepo")
agent = create_deep_agent(
    model=llm, 
    backend=backend, 
    skills=["./skills/frontend-design"],
    permissions=[
        FilesystemPermission(
            operations=["write"],           # 限制的操作类型
            paths=["/skills/**"],           # 限制的路径模式
            mode="deny",                    # 模式：deny（禁止）/ interrupt（审批）
        ),
        FilesystemPermission(
            operations=["write"],
            paths=["/output/**"],
            mode="allow",                   # 允许写入 output 目录
        ),
    ]
)
```