# <center> MCP </center>

## 下载与资源
[文档](https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro)
[Servers](https://github.com/modelcontextprotocol/servers)
[awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)
[mcpservers](https://mcpservers.org/zh-CN/)
[LangChain MCP](https://docs.langchain.com/oss/python/langchain/mcp)
[Figwright](https://github.com/awdr74100/figwright)  Free, two-way Figma MCP server


## 简介
MCP起源于2024年11月25日 Anthropic 发布的文章：Introducing the Model Context Protocol。

MCP （Model Context Protocol，模型上下文协议）定义了应用程序和 AI 模型之间交换上下文信息的方式。这使得开发者能够以一致的方式将各种数据源、工具和功能连接到 AI 模型（一个中间协议层），就像 USB-C 让不同设备能够通过相同的接口连接一样。MCP 的目标是创建一个通用标准，使 AI 应用程序的开发和集成变得更加简单和统一。

![mcp.jpg](https://img.ilark.io/img/202601/1786020546069-upbj3.jpg)
MCP 就是以更标准的方式让 LLM Chat 使用不同工具。这样你应该更容易理解“中间协议层”的概念了。Anthropic 旨在实现 LLM Tool Call 的标准。


## uv安装环境
[教程](https://www.runoob.com/python3/uv-tutorial.html)
uv 是 Astral 公司推出的一款基于 Rust 编写的 Python 包管理工具，旨在成为 “Python 的 Cargo”。它提供了快速、可靠且易用的包管理体验，在性能、兼容性和功能上都有出色表现，为 Python 项目的开发和管理带来了新的选择。
```py
1. 安装
# 通过PIP安装（推荐）
直接使用Python自带的pip安装，兼容性最佳： - 0.7.16
pip install uv   
uv --version  //uv 0.12.1

2. 初始化项目
uv init my_project
cd my_project
# 这会创建以下基本项目结构：
my_project/
└──src
|   └──my_project
|        └──__init__.py      
├── pyproject.toml    # 项目配置和依赖声明
├── .python-version   # 固定 Python 版本
└── README.md

3. 创建虚拟环境
uv venv
# 激活虚拟环境
source .venv/Scripts/activate

4. 添加生产依赖
uv add mcp
uv add openai python-dotenv
uv add httpx
# 设置国内镜像，如果安装速度慢，可以在 pyproject.toml 中设置国内镜像源
[tool.uv]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
# 查看已安装的包：
uv pip list

5. 运行
uv run 是 uv 中非常实用的命令，可以无需手动激活虚拟环境直接运行脚本或命令，uv 会自动找到并使用正确的环境
# 直接运行 Python 脚本
uv run main.py
# 运行任意命令（在虚拟环境的上下文中执行）
uv run python -c "import requests; print(requests.__version__)"
```

## MCP Server
根据MCP协议定义，Server可以提供三种类型的标准能力，Resources、Tools、Prompts，每个Server可同时提供者三种类型能力或其中一种。
**Resources**：资源，类似于文件数据读取，可以是文件资源或是API响应返回的内容。
**Tools**：工具，第三方服务、功能函数，通过此可控制LLM可调用哪些函数。
**Prompts**：提示词，为用户预先定义好的完成特定任务的模板。

Model Context Protocol（MCP）由Anthropic开源，用于将大型语言模型直接连接数据源。它分为本地 stdio / 远程 SSE/HTTP，支持**标准输入输出（stdio）** 和 **基于HTTP的服务器推送事件（SSE）** 两种传输方式。Stdio模式适用于本地通信，通过启动服务器作为子进程实现高效低延迟的数据交换，适合快速响应的本地应用。而基于HTTP和SSE的方式则适用于分布式或远程场景，实现客户端与服务器间实时数据推送。
1. 本地通讯：文件读写、Git、本地数据库、计算器。使用了stdio传输数据，具体流程Client启动Server程序作为子进程，其消息通讯是通过stdin/stdout进行的，消息格式为JSON-RPC 2.0。   
2. 远程通讯：联网搜索、企业 API、CRM、向量库、第三方服务。 Server 对外只暴露三类标准化基元（Primitives），模型可自动发现、动态调用。Client与Server可以部署在任何地方，Client使用SSE与Server进行通讯，消息的格式为JSON-RPC 2.0，Server定义了/see与/messages接口用于推送与接收数据。


## Server端设计
把工具（tool）封装进特定的方法中，以提供统一的调用方法和方式，这样可以增加规范性和可复用性。
```py
uv add mcp

# server.py
from mcp.server.mcpserver import MCPServer

# 初始化 MCP 服务器
mcp = MCPServer("WeatherServer")

# 给函数加上装饰器即可
@mcp.tool()
async def userful_tool(input: str):
    """
    工具的说明和使用
    """
    pass
            
if __name__ == "__main__":
    mcp.run(transport='stdio')

# 可使用inspector做测试
npx -y @modelcontextprotocol/inspector uv run server.py
```

## Server案例
本地python文件，以stdio传输。以查询天气的工具来做简单的案例
```py
# server.py
import json
import httpx
import asyncio
from typing import Any
from mcp.server.mcpserver import MCPServer

# 初始化 MCP 服务器
mcp = MCPServer("WeatherServer")
# OpenWeather API 配置
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "1b63xxxxx...." # 请替换为你自己的 OpenWeather API Key
USER_AGENT = "weather-app/1.0"

async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    从 OpenWeather API 获取天气信息。
    :param city: 城市名称（需使用英文，如 Beijing）
    :return: 天气数据字典；若出错返回包含 error 信息的字典
    """
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "zh_cn"
     }
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(OPENWEATHER_API_BASE, params=params,
            headers=headers, timeout=30)
            response.raise_for_status()
            return response.json() # 返回字典类型
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP 错误: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}
            
def format_weather(data: dict[str, Any] | str) -> str:
    """
    将天气数据格式化为易读文本。
    :param data: 天气数据（可以是字典或 JSON 字符串）
    :return: 格式化后的天气信息字符串
    """
    # 如果传入的是字符串，则先转换为字典
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return f"无法解析天气数据: {e}"
            # 如果数据中包含错误信息，直接返回错误提示
    if "error" in data:
        return f"{data['error']}"
    # 提取数据时做容错处理
    city = data.get("name", "未知")
    country = data.get("sys", {}).get("country", "未知")
    temp = data.get("main", {}).get("temp", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    # weather 可能为空列表，因此用 [0] 前先提供默认字典
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "未知")
    return (
        f"城市{city}, {country}\n"
        f"温度: {temp}°C\n"
        f"湿度: {humidity}%\n"
        f"风速: {wind_speed} m/s\n"
        f"天气: {description}\n")

@mcp.tool()
async def query_weather(city: str) -> str:
    """
    输入指定城市的英文名称，返回今日天气查询结果。
    :param city: 城市名称（需使用英文）
    :return: 格式化后的天气信息
    """
    data = await fetch_weather(city)
    print(123, "weather data:", data)
    return format_weather(data)

if __name__ == "__main__":
    mcp.run(transport='stdio')

# Inspector测试
npx -y @modelcontextprotocol/inspector uv run server.py
```

![mcp2.jpg](https://img.ilark.io/img/202601/1786020632244-5nkll.jpg)
测试界面

## 服务器测试 MCP Inspector
在实际开发MCP服务器的过程中，Anthropic提供了一个非常便捷的debug工具：Inspector。借助Inspector，我们能够非常快捷的调用各类server，并测试其功能
`npx -y @modelcontextprotocol/inspector uv run server.py`


## streamable-http server
远程mcp和本地版的在写法上其实差别不大，只是在传输方式上有差别。它采用Fastapi来提供网络服务，这样其它的客户端就可以使用它啰！
```py
# 安装包
uv add mcp fastapi

import contextlib
from mcp.server import MCPServer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 1. 初始化 MCP 2.0 Server 实例
mcp = MCPServer("Streamable-HTTP-MCP-Server")

# 2. 注册 MCP 工具 (Tools)
@mcp.tool()
async def calculate_sum(a: float, b: float) -> str:
    """计算两个数的和"""
    result = a + b
    return f"计算结果为: {result}"

@mcp.tool()
async def get_weather(city: str) -> dict:
    """获取指定城市的天气状况"""
    return {"city": city, "status": "晴朗", "temperature": 25.5}


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    关键：必须在 lifespan 中启动 MCP 的 session_manager，
    否则 transport 会报 RuntimeError: Task group is not initialized
    """
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(mcp.session_manager.run())
        yield

# 3. 创建主 FastAPI 应用
app = FastAPI(
    title="FastAPI MCP Server",
    lifespan=lifespan,
    version="1.0.0"
)

# CORS：必须暴露 Mcp-Session-Id，否则客户端无法维持会话
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id", "WWW-Authenticate"],
)

# 4. 生成 Streamable HTTP 模式的 ASGI 子应用并挂载
# 协议类型指定为 streamable-http，将单端点流式 HTTP 挂载至 /mcp
# 注意：streamable_http_app() 内部默认端点是 /mcp
# 所以最终访问路径是 /mcp/mcp
mcp_http_app = mcp.streamable_http_app()
app.mount("/mcp", mcp_http_app)

# 路由测试
@app.get("/")
async def root():
    return {
        "message": "MCP HTTP Server is running",
        "mcp_endpoint": "/mcp",
        "docs": "/docs",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)

# 在浏览器连接mcp时输入: http://localhost:8005/mcp/mcp
```

## 测试
`uv run server.py` 即可拉起mcp服务

客户端方面: `npx -y @modelcontextprotocol/inspector` 启动客户端，连接测试

![mcpinspector.jpg](https://img.ilark.io/img/202601/1786107572013-vefw1.jpg)
**注意URL的填写**： http://localhost:8005/mcp/mcp

连上后，就可点开`tool`进行测试了。


## LangChain MCP client
LangChain agents可以很方便的集成MCP的功能，有个专门的库来处理，它就是`langchain-mcp-adapters`library。导入的`MultiServerMCPClient`是官方提供的多服务器管理客户端，可以同时连接多个 MCP 服务器。

### 主要能力
- 工具调用与管理：能够将各种 MCP 服务器提供的工具转换为Agent可直接使用的标准化工具。
- 多协议通信支持：支持两种主要的通信方式 – stdio 模式（将 MCP 服务器作为本地子进程运行）和 HTTP 模式（通过 SSE 通道和 POST 请求进行通信）。
- 资源与提示词操作：可以访问和管理 MCP 服务器提供的资源文件以及预定义的提示词模板。
- 多服务器协同：通过 MultiServerMCPClient 能够同时连接和管理多个 MCP 服务器，从它们加载工具、提示和资源。

### 应用场景
- 数据集成：连接内部的知识库、数据库和业务系统，让 AI 能够访问最新的数据。例如，可以连接 SalesForce、Google Drive 等系统。
- 文件操作自动化：通过文件系统 MCP 服务器，AI 智能体可以自动执行文件读写、目录管理等操作。
- 多工具协同工作流：通过连接多个专用 MCP 服务器，构建能够处理复杂任务的 AI 智能体系统。

```py
# 主要是`MultiServerMCPClient`连接 MCP 服务器，以提供工具调用。
from langchain_mcp_adapters.client import MultiServerMCPClient  

client = MultiServerMCPClient(
    {
        "fetch_weather": {
            "transport": "stdio",  # Local subprocess communication
            "command": "python",
            # path to your server.py file
            "args": ["weather_server.py"],
        },
        "httpserver": {
            "transport": "http",  # HTTP-based remote server
            # Ensure you start your weather server on port 8000
            "url": "http://localhost:8005/mcp/mcp",
        }
    }
)
tools = await client.get_tools()
print(123, "tools", tools)
```


## MCPClient完整案例
```py
uv add "mcp<2"   #  注意：mcp2.0暂不与LangChain包兼容，要装低版本的
uv add langchain-mcp-adapters
uv add langchain langchain-openai python-dotenv

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import dotenv_values

env_vars = dotenv_values('.env')
DEEPSEEK_API_KEY = env_vars['DEEPSEEK_API_KEY'] 
DEEPSEEK_BASE_URL = env_vars['DEEPSEEK_BASE'] 

llm = ChatOpenAI(model="deepseek-v4-flash", api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

async def main():
    client = MultiServerMCPClient(
        {
            "httpserver": {
                "transport": "http",  # HTTP-based remote server
                # Ensure you start your weather server on port 8005
                "url": "http://localhost:8005/mcp/mcp",
            }
        }
    )

    tools = await client.get_tools()
    print(123, "tools", tools) # get_weather
    agent = create_agent(model=llm, tools=tools)
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in beijing?"}]}
    )
    print(996, weather_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
```

