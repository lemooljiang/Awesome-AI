# <center> LangGraph  </center>

## 简介
`LangGraph` 解决线性序列的局限性问题，而解决的方法就是循环图。在`LangGraph`框架中，用图来管理代理的生命周期并在其状态内将暂存器作为消息进行跟踪，增加了以循环方式跨各种计算步骤协调多个链或参与者的功能。

`LangGraph`通过组合`Nodes`和`Edges`去创建复杂的循环工作流程，通过消息传递的方式串联所有的节点形成一个通路。那么维持消息能够及时的更新并向该去的地方传递，则依赖`langGraph`构建的`State`概念。 

`LangGraph`的优势则是：
- **循环和分支**：在应用程序中实现循环和条件。
- **持久性**：在图中的每个步骤之后自动保存状态。随时暂停和恢复图形执行，以支持错误恢复、人机交互工作流程等。
- **人机交互**：中断图形执行以批准或编辑代理计划的下一个操作。
- **流支持**：流输出由每个节点生成（包括令牌流）。
- **与LangChain集成**：LangGraph 与LangChain和LangSmith无缝集成。

![调用示意图](https://ipfs.ilark.io/ipfs/QmQuzkAzvJLRTmb3Dgs82UEaHcf6vpgUBywFtTAmqZbLe7)

## 下载与资源
[LangGraph文档 |](https://langchain-ai.github.io/langgraph)
[LangGraph Github |](https://github.com/langchain-ai/langgraph)
[LangChain github |](https://github.com/langchain-ai/langchain)
[LangChain |](https://python.langchain.com/docs/introduction/)


## 安装
```py
virtualenv lang_env
source lang_env/Scripts/activate   //windows
pip install langgraph -i https://pypi.tuna.tsinghua.edu.cn/simple （清华镜像） 0.4.8
pip install langchain langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple  0.3.25 / 0.3.22

pip install langchain-experimental -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install langgraph-supervisor -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install bs4 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## GraphState
在图中提到了节点、边、状态和路由四个概念。

定义图时要做的第一件事是定义图的`State`。状态表示会随着图计算的进行而维护和更新的上下文或记忆。它用来确保图中的每个步骤都可以访问先前步骤的相关信息，从而可以根据整个过程中积累的数据进行动态决策。这个过程通过状态图`StateGraph`类实现，它继承自 `Graph` 类，这意味着 `StateGraph` 会使用或扩展基类的属性和方法。
```py
# 构建图
builder = StateGraph(dict) 
```

## Nodes
在 `LangGraph` 中，节点是一个 `python` 函数（sync或async），接收当前`State`作为输入，执行自定义的计算，并返回更新的`State`。所以其中第一个位置参数是`state` 。
```py
def agent_node(state:InputState):
    print("我是一个AI Agent。")
    return 
```

定义好了节点以后，我们需要使用`add_node`方法将这些节点添加到图中。在将节点添加到图中的时候，可以自定义节点的名称。而如果不指定名称，则会为自动指定一个与函数名称等效的默认名称。代码如下：
```py
builder.add_node("agent_node", agent_node)
builder.add_node("action_node", action_node)
```

## Edges
Edges（边）用来定义逻辑如何路由以及图何时开始与停止。这是代理工作以及不同节点如何相互通信的重要组成部分。有几种关键的边类型：
- 普通边：直接从一个节点到下一个节点。
- 条件边：调用函数来确定下一个要转到的节点。
- 入口点：当用户输入到达时首先调用哪个节点。
- 条件入口点：调用函数来确定当用户输入到达时首先调用哪个节点。

同样，我们先看普通边。如果直接想从节点`A`到节点`B`，可以直接使用`add_edge`方法。注意：`LangGraph`有两个特殊的节点：`START`和`END`。`START`表示将用户输入发送到图的节点。使用该节点的主要目的是确定应该首先调用哪些节点。`END`节点是代表终端节点的特殊节点。当想要指示哪些边完成后没有任何操作时，将使用该节点。因此，一个完整的图就可以使用如下代码进行定义：
```py
from langgraph.graph import START, END

builder.add_edge(START, "agent_node")
builder.add_edge("agent_node", "action_node")
builder.add_edge("action_node", END)
# 最后，通过`compile`编译图。在编译过程中，会对图结构执行一些基本检查（如有没有孤立节点等）。
graph = builder.compile()

graph.invoke({"question":"hello，你好"})
```

## State
对于`LangGraph`的底层图算法是利用消息传递机制来定义程序的运行这一结论，接下来，我们将详细探讨消息（Messages）是如何通过`State`进行传递的，其中包含了什么传递模式和内容。

`State`实际上是一个共享的数据结构。如上图所示，状态表现为一个简单的字典。通过对这个字典进行读写操作，可以实现自左而右的数据流动，从而构建一个可运行的图结构。那么根据前面学习的内容，我们可以利用这个流程来复现并理解图中的动态数据交换

`LangGraph`内部原理是：`State`中的每个`key`都有自己独立的`Reducer`函数，通过指定的`reducer`函数应用状态值的更新。
`Reducer` 函数用来根据当前的状态（state）和一个操作（action）来计算并返回新的状态。它是一种设计模式，用于将业务逻辑与状态变更解耦，使得状态的变更预测性更强并且容易追踪。这样的函数通常接收两个参数：当前的状态（state）和一个描述应用了什么操作的对象（action）， 根据 `action` 类型来决定如何修改状态。比如，在一个购物车应用中，可能会有添加商品、删除商品、修改商品数量等操作。返回一个新的状态对象，而不是修改原始状态对象。简单理解，`Reducer`函数做的就是根据给定的输入（当前状态和操作）生成新的状态。

掌握`State`的定义模式和消息传递是`LangGraph`中的关键，也是构建应用最核心的部分，所有的高阶功能，如工具调用、上下文记忆，人机交互等依赖`State`的管理和使用，所以大家务必理解并掌握上述相关内容。


## 一个完整案例
```py
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.graph import START, END
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


env_vars = dotenv_values('.env')
OPENAI_KEY = env_vars['OPENAI_API_KEY'] 
OPENAI_BASE_URL = env_vars['OPENAI_API_BASE'] 


# 定义输入的模式
class InputState(TypedDict):
    question: str


# 定义输出的模式
class OutputState(TypedDict):
    answer: str


# 将 InputState 和 OutputState 这两个 TypedDict 类型合并成一个更全面的字典类型。
class OverallState(InputState, OutputState):
    pass


def agent_node(state: InputState):
    print("I am AI Agent ", InputState, state["question"])
    return {"question": state["question"]}


def action_node(state: InputState):
    print("agent action", InputState, state["question"])
    step = state["question"]
    return {"answer": f"input question {step}，succeed！"}


def llm_node(state: InputState):
    messages = [
        # ("system","You are a helpful assistant"),
        ("user", state["question"])
    ]
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_KEY,base_url=OPENAI_BASE_URL)
    # gpt-3.5-turbo o1-mini
    response = llm.invoke(messages) 
    print(123, "response", response)
    return {"answer": response.content}

# 明确指定它的输入和输出数据的结构或模式
builder = StateGraph(OverallState, input=InputState, output=OutputState)

# 添加节点
builder.add_node("llm_node", llm_node)

# 添加边
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

# 编译图
graph = builder.compile()

final_answer = graph.invoke({"question":"how are you"})
print(final_answer["answer"])
```

## 单代理架构应用
在代理架构模式下，大模型能通过多种方式进行操作控制。最基本的功能是在两个潜在路径之间做出选择。进而在每条路径上，如果存在多个可用工具，大模型能够自主决定调用哪一个。更复杂的情况下，它还能评估生成的答案是否满足问题的需求。如有必要进行额外工作，它将自行执行，直到得到一个充分满足条件的答案为止。`LangGraph`框架则是从这个角度出发，接入了路由代理，工具代理，自主循环代理以及多代理这四大类代理架构，以支持不同的场景需求。


## Router Agent
`LangGraph`中`Router`的常用使用形式，通过预定义的分支结构，可以根据用户的输入请求灵活适配不同的场景，在这个过程中，结构化输出对于路由至关重要，因为它们确保系统可以可靠地解释大模型的决定并采取行动。这种`Router Agent`（路由代理）的优势就是可以精准的控制程序链路中的每一个细节，但同时也表现出来了这是一种相对有限的控制级别的代理架构，因为大模型通常只能控制单个决策。想象一下上面的场景中，如果我们希望定义的`insert_db`不仅仅只是包含插入数据库，而是有一堆各式各样的工具，比如网络搜索，RAG等等，应该如何进一步的扩展呢？ 难道要做对每一个工具在`insert_db`节点下再通过`Router Function`做分支判断吗？虽然可行，但总归并不是高效的做法。


## Tool Calling Agent
`Tool Calling Agent`（工具调用代理）是`LangGraph`支持的一种`AI Agent`代理架构。这个代理架构是在`Router Agent`的基础上，大模型可以自主选择并使用多种工具来完成某个条件分支中的任务。工具调用大家应该非常熟悉，当我们希望代理与外部系统交互时，工具就非常有用。大模型能根据用户的自然语言输入选择调用工具，并将返回符合该工具架构的输出。

在`LangGraph`框架中，可以直接使用预构建`ToolNode`进行工具调用，其内部实现原理和我们之前介绍的手动实现的`Function Calling`流程思路基本一致.

经过`ToolNode`工具后，其返回的是一个`LangChain Runnable`对象，会**将图形状态（带有消息列表）作为输入并输出状态更新以及工具调用的结果**，通过这种设计去适配`LangGraph`中其他的功能组件。比如我们后续要介绍的`LangGraph`预构建的更高级`AI Agent`架构 - `ReAct`，两者搭配起来可以开箱即用，同时通过`ToolNode`构建的工具对象也能与任何`StateGraph`一起使用，只要其状态中具有带有适当`Reducer`的`messages`键。由此，对于`ToolNode`的使用，有三个必要的点需要满足，即：

1. **状态必须包含消息列表。**
2. **最后一条消息必须是AIMessage。**
3. **AIMessage必须填充tool_calls。**

`ToolNode`使用消息列表对图状态进行操作。所以它要求消息列表中的最后一条消息是带有`tool_calls`参数的`AIMessage` ，比如我们可以手动调用工具节点
```py
from langchain_core.messages import AIMessage

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "fetch_real_time_info",
            "args": {"query": "小米汽车"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

tool_node.invoke({"messages": [message_with_single_tool_call]})
```

通过`ToolNode(tools)`可以根据参数来执行函数，并返回结果。而其前一步，根据自然语言生成执行具体某个函数必要参数的过程，则由大模型决定，所以一个完整的基于大模型的工具调用过程应该是，在实例化大模型的时候，就告诉大模型你都有哪些工具可以使用。这个过程可以通过`bind_tools`函数来实现.


## 案例：联网查找的代理 
```py
from typing import Union, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import operator, requests, json
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


env_vars = dotenv_values('.env')
OPENAI_KEY = env_vars['OPENAI_API_KEY'] 
OPENAI_BASE_URL = env_vars['OPENAI_API_BASE'] 
SERPER_KEY = env_vars['SERPER_KEY'] 

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY,base_url=OPENAI_BASE_URL)  


class SearchQuery(BaseModel):
    query: str = Field(description="Questions for networking queries")

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

@tool(args_schema = SearchQuery)
def fetch_real_time_info(query):
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
    if 'organic' in data:
        return json.dumps(data['organic'],  ensure_ascii=False) 
    else:
        return json.dumps({"error": "No organic results found"},  ensure_ascii=False)  


def chat_with_model(state):
    """generate structured output"""
    messages = state['messages']
    response = llm.invoke(messages) 
    return {"messages": [response]}

# 判断是否要工具调用
def exists_function_calling(state: AgentState):
    result = state['messages'][-1]
    print(563, "exists_function_calling")
    return len(result.tool_calls) > 0

# 不调用工具
def final_answer(state):
    """generate natural language responses"""
    messages = state['messages'][-1]
    return {"messages": [messages]}

# 调用工具
def execute_function(state: AgentState):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    tools = [fetch_real_time_info]
    tools = {t.name: t for t in tools}
    for t in tool_calls:
        if not t['name'] in tools:     
            result = "bad tool name, retry" 
        else:
            result = tools[t['name']].invoke(t['args'])
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {'messages': results}


# 请你基于现在得到的信息，进行总结，生成专业的回复
SYSTEM_PROMPT = """
Please summarize the information obtained so far and generate a professional response.
"""

# 拼接查找的信息后再最终生成结果
def natural_response(state):
    """generate final language responses"""
    messages = state['messages'][-1]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=messages.content)]
    response = llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(AgentState)

graph.add_node("chat_with_model", chat_with_model)
graph.add_node("execute_function", execute_function)
graph.add_node("final_answer", final_answer)
graph.add_node("natural_response", natural_response)

# 设置图的启动节点
graph.set_entry_point("chat_with_model")
graph.add_conditional_edges(
    "chat_with_model",
    exists_function_calling,
    {True: "execute_function", False: "final_answer"}
    )
graph.add_edge("execute_function", "natural_response")
graph.set_finish_point("final_answer")
graph.set_finish_point("natural_response")
graph = graph.compile()


tools = [fetch_real_time_info]
llm = llm.bind_tools(tools)

messages = [HumanMessage(content="what is labubu")]  #测试
result = graph.invoke({"messages": messages})
res = result["messages"][-1].content
print(896, res)
```

![toolcall.jpg](https://ipfs.ilark.io/ipfs/QmcyWaFj9eAbv8E6Efj9PMgbSPoaYs1p396VwUtfB4vzcm)
整体流程如上所示


## Full Autonomous
`Tool Calling Agent` 的局限性又在于：虽然它可以自主选择工具，但在其架构中，每次仅能执行一次函数调用（无论是单个外部函数还是多个外部函数）**。因此，当任务需要依次执行 A 工具、B 工具和 C 工具时，它无法支持这种自主控制的过程。因此，面对这种更复杂的需求，就需要引入了 `Full Autonomous`（自治循环代理）架构。
`Full Autonmonous` 以两种主要的方式去扩展了`Agent`对工作流的控制，分别是：
- 多步骤决策： `Agent`可以控制一系列决策，而不仅仅是一个决策。
- 工具访问： `Agent`可以选择并使用多种工具来完成任务。


## ReAct
当有了工具列表和模型后，就可以通过`create_react_agent`这个`LangGraph`框架中预构建的方法来创建自治循环代理（ReAct）的工作流，其必要的参数如下：
- model： 支持工具调用的LangChain聊天模型。
- tools： 工具列表、ToolExecutor 或 ToolNode 实例。
- state_schema：图的状态模式。必须有`messages`和`is_last_step`键。默认为定义这两个键的`Agent State`。


## ReAct案例
```py
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Union, Optional
from pydantic import BaseModel, Field
import requests, json, asyncio
from langgraph.prebuilt import create_react_agent


env_vars = dotenv_values('.env')
OPENAI_KEY = env_vars['OPENAI_API_KEY'] 
OPENAI_BASE_URL = env_vars['OPENAI_API_BASE'] 
SERPER_KEY = env_vars['SERPER_KEY'] 
WEATHER_KEY = env_vars['WEATHER_KEY']


## 第一个工具
class WeatherLoc(BaseModel):
    location: str = Field(description="The location name of the city")


@tool(args_schema=WeatherLoc)
def get_weather(location):
    """
    Function to query current weather.
    :param loc: Required parameter, of type string, representing the specific city name for the weather query. \
    Note that for cities in China, the corresponding English city name should be used. For example, to query the weather for Beijing, \
    the loc parameter should be input as 'Beijing'.
    :return: The result of the OpenWeather API query for current weather, with the specific URL request address being: https://api.openweathermap.org/data/2.5/weather. \
    The return type is a JSON-formatted object after parsing, represented as a string, containing all important weather information.
    """
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2.设置查询参数
    params = {
        "q": location,               
        "appid": WEATHER_KEY,    
        "units": "metric",      
        "lang":"zh_cn"        
    }

    # Step 3.发送GET请求
    response = requests.get(url, params=params)
    
    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)


# 第二个工具
class SearchQuery(BaseModel):
    query: str = Field(description="Questions for networking queries")


@tool(args_schema = SearchQuery)
def fetch_real_time_info(query):
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
    if 'organic' in data:
        return json.dumps(data['organic'],  ensure_ascii=False) 
    else:
        return json.dumps({"error": "No organic results found"},  ensure_ascii=False)  


llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY,base_url=OPENAI_BASE_URL)  
tools = [fetch_real_time_info, get_weather]
graph = create_react_agent(llm, tools=tools)


# 可以自动处理成 HumanMessage 的消息格式
finan_response = graph.invoke({"messages":["what is labubu"]})
print(569, finan_response)  
# finan_response["messages"][-1].content
```

以上案例中使用了两个外部工具`[fetch_real_time_info, get_weather]`, 图的生成也只用了一行代码`create_react_agent(llm, tools=tools)` ,确实比之前两个案例简单直接了很多。



## 事件流
在实际应用中，流式输出尤其适用于需要快速反馈的业务场景，如聊天机器人，因为大语言模型可能需要几秒钟才能生成对查询的完整响应，这远远慢于应用程序对最终用户的响应速度约为 200-300 毫秒的阈值，如果是涉及多个大模型调用的复杂应用程序，这种延时会变得更加明显。让应用程序感觉响应更快的关键策略是显示中间进度；即，通过 `token` 流式传输大模型`Token`的输出，以此来显著提升用户体验。而在开发阶段，利用流式输出功能可以准确追踪到事件的具体执行阶段，并捕获相关数据，从而接入不同逻辑的数据处理和决策流程。是我们在应用开发中必须理解和掌握的技术点。

流式输出功能在`LangGraph` 框架中的实现方式比较简单，因为`LangGraph`底层是基于 `LangChain` 构建的，所有就直接把`LangChain`中的回调系统拿过来使用了。在`LangChain`中的流式输出是：以块的形式传输最终输出，即一旦监测到有可用的块，就直接生成它。最常见和最关键的流数据是大模型本身生成的输出。 大模型通常需要时间才能生成完整的响应，通过实时流式传输输出，用户可以在生成时看到部分结果，这可以提供即时反馈并有助于减少用户的等待时间。

`LangGraph`框架中的工作流中由各个步骤的节点和边组成。这里的流式传输涉及在各个节点请求更新时跟踪图状态的变化。这样可以更精细地监控工作流中当前处于活动状态的节点，并在工作流经过不同阶段时提供有关工作流状态的实时更新。其实现方式也是和`LangChain`一样通过`.stream`和`.astream`方法执行流式输出，只不过适配到了图结构中。调用`.stream`和`.astream`方法时可以指定几种不同的模式，即：

- "values" ：在图中的每个步骤之后流式传输**状态**的完整值。
- "updates" ：在图中的每个步骤之后将更新流式传输到状态。如果在同一步骤中进行多个更新（例如运行多个节点），则这些更新将单独流式传输。
- "debug" ：在整个图的执行过程中流式传输尽可能多的信息，主要用于调试程序。
- "messages"：记录每个`messages`中的增量`token`。
- "custom"：自定义流，通过`LangGraph 的 StreamWriter`方法

```py
async def main():
    async for event in graph.astream_events({"messages": ["what is labubu"]}):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            print(event["data"]["chunk"].content, flush=True)

asyncio.run(main())
```


##  多代理系统
多代理系统其实就可以非常简单的理解为：将原本的应用程序拆分成多个较小的独立代理，从而组合而成的系统。这些小的独立代理可以是简单的大模型交互代理，也可以是复杂的 `ReAct` 代理。举个比较热门的案例，假设我们需要建立一个用于数据分析的`Agent`，则可以设计代理配置：`Agent 1`作为用户意图识别代理，集成大模型用来解析用户的查询和指令，理解其意图和需求，并将用户输入转化为具体的任务。`Agent 2`作为数据分析代理，集成大模型并绑定若干个处理不同数据和需求的工具，提供统计分析、趋势预测和数据可视化服务。当任务涉及到代码生成时，`Agent 3`，即代码执行代理，会接收用户输入的代码，在安全的Python环境中执行这些代码，并返回运行结果，用于代码测试、执行特定算法或自动化任务。

由此能感受到的是多智能体系统 （MAS） 是通过多个单代理之间的协作来解决复杂的任务，其中多代理系统中集成的每个单代理，都有特定的背景身份和独有的技能。其显著的优势则包含如下三个方面：
- 专业化：当一个系统中可以创建多个专注于特定领域的专家代理，能实现处理更复杂的应用的`AI`系统。
- 模块化：单独的代理开发模式对于开发、测试和维护完整代理系统是更加容易的。
- 控制度：显式地控制代理的通信方式，而不仅仅是依赖函数调用。

`LangGraph`利用基于图的结构来定义代理并在它们之间建立连接。在此框架中，每个代理都表示为图中的一个节点，并通过边链接到其它代理。每个代理通过接收来自其他代理的输入并将控制权传递给下一个代理来执行其指定的操作。在`LangGraph` 框架的设计中，主要通过如下几种方法来建立各个子代理之间的通信连接：
- NetWork（网络）：每个代理都可以与其他每个代理通信。任何代理都可以决定接下来要呼叫哪个其他代理。
- Supervisor（主管）：每个代理都与一个 `Supervisor` 代理通信。由 `Supervisor` 代理决定接下来应调用哪个代理。
- Supervisor （tool-calling）： `Supervisor` 架构的一个特例。每个代理都是一个工具。由`Supervisor`代理通过工具调用的方式来决定调用哪些子代理执行任务，以及要传递给这些代理程序的参数
- Hierarchical（分层）：定义具有 `supervisor` 嵌套 `supervisor`多代理系统。这是 `Supervisor` 架构的一种泛化，允许更复杂的控制流。

-- Subgraphs
`Subgraphs`（子图）指的是能够用作另一个图中的节点的图。**简单理解就是：把一个已经编译好的图，嵌入到另一个已经编译好的图中，并且两个独立图的中的状态可以信息共享**。一个典型的应用就是构建多代理系统架构。它所做的事情是：当把每个独立的`Agent`图结构定义为一个子图时，只要遵守子图的接口（输入和输出模式）规范，那么子图中定义的共享状态就可以在父图中进行使用

添加子图主要解决的问题就是解决各`Single-Agent`之间的通信问题，即它们如何在图执行期间在彼此之间传递状态。这主要有两种情况：
- 父、子图的状态模式中有共同的键（通道）。
- 父、子图的状态模式中没有共同的键。（通道）


## Supervisor
`LangGraph` 中的 `Supervisor`充当多代理工作流程中的中央控制器，协调各个代理之间的通信和任务分配。它的工作原理是接收一个代理的输出，解释这些消息，然后相应地指导任务流程。它在`LangGraph` 中基于图结构中的节点实现，允许随着任务的发展或新代理的集成而动态交互和灵活调整工作流程，从而优化流程的有效性和速度。

实现的思路是：将代理定义为节点，并添加一个 `supervisor` 节点来决定接下来应该调用哪些代理节点。使用条件边根据 `supervisor` 的决策将执行路由到适当的代理节点。

![supervisor.jpg](https://ipfs.ilark.io/ipfs/QmdUBsWsR7R6GiuJ23PTFg4hpK2dBQcXKCg4mQ19dPtDbX)
Supervisor案例示意图

```py
from typing import Union, Optional, Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState, END
import operator, requests, json
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import create_react_agent

env_vars = dotenv_values('.env')
OPENAI_KEY = env_vars['OPENAI_API_KEY'] 
OPENAI_BASE_URL = env_vars['OPENAI_API_BASE'] 
SERPER_KEY = env_vars['SERPER_KEY'] 

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY,base_url=OPENAI_BASE_URL)  

class AgentState(MessagesState):
    next: str

# 第一个代理
class SearchQuery(BaseModel):
    query: str = Field(description="Questions for networking queries")

@tool(args_schema = SearchQuery)
def fetch_real_time_info(query):
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
    if 'organic' in data:
        return json.dumps(data['organic'],  ensure_ascii=False) 
    else:
        return json.dumps({"error": "No organic results found"},  ensure_ascii=False)  

# 第二个代理
repl = PythonREPL()
@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return result_str


# 使用 `create_react_agent` 构建成两个`ReAct`代理。
search_agent = create_react_agent(
    llm, 
    tools=[fetch_real_time_info]
)

code_agent = create_react_agent(
    llm, 
    tools=[python_repl]
)


# 分别将两个`ReAct Agent` 构造成节点，并添加代理名称标识。
def search_node(state: AgentState):
    result = search_agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="searcher")]
    }


def code_node(state: AgentState):
    result = code_agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="coder")]
    }

# 直接交互的节点
def chat(state: AgentState):
    messages = state["messages"][-1]
    model_response = llm.invoke(messages.content)
    final_response = [HumanMessage(content=model_response.content, name="chat")]
    return {"messages": final_response}


# 设置代理主管可以管理的子代理
members = ["chat", "searcher", "coder"]
options = members + ["FINISH"]

# 定义路由
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH"""
    next: Literal[*options]


def supervisor(state: AgentState):
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}.\n\n"
        "Each worker has a specific role:\n"
        "- chat: Responds directly to user inputs using natural language.\n"
        "- coder: Activated for tasks that require mathematical calculations or specific coding needs.\n"
        "- searcher: Get real-time Internet information.\n"
        "Given the following user request, respond with the worker to act next."
        " Each worker will perform a task and respond with their results and status."
        " When finished, respond with FINISH."
    )
    messages = [{"role": "system", "content": system_prompt},] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_ = response["next"]
    if next_ == "FINISH":
        next_ = END
    return {"next": next_}  


builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("chat", chat)
builder.add_node("searcher", search_node)
builder.add_node("coder", code_node)


for member in members:
    # 每个子代理在完成工作后总是向主管“汇报”
    builder.add_edge(member, "supervisor")
builder.add_conditional_edges("supervisor", lambda state: state["next"])
builder.add_edge(START, "supervisor")
graph = builder.compile()

finan_response = graph.invoke({"messages":["What are the latest movies in 2025"]})
print(569, finan_response)  # finan_response["messages"][-1].content
```


## token计算
```py
from langchain_community.callbacks.manager import get_openai_callback

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    stream_usage=True,
)

with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    print(cb)
//Tokens Used: 27
	Prompt Tokens: 11
	Completion Tokens: 16
Successful Requests: 1
Total Cost (USD): $2.95e-05


with get_openai_callback() as cb:
    result = graph.invoke({"messages":["How's the weather in Beijing"]})
    print(668, result)
    print(236, cb)
    print(569, cb.total_tokens)
//236 Tokens Used: 3715
        Prompt Tokens: 3560
                Prompt Tokens Cached: 0
        Completion Tokens: 155
                Reasoning Tokens: 0
Successful Requests: 6
Total Cost (USD): $0.0006269999999999998
569 3715

``` 

## 短期记忆
`LangGraph`框架中的`checkpointer`，通过一些数据结构来存储`State`状态中产生的信息，并且在每个`task`开始时去读取全局的状态。主要通过以下四种方式来实现：
- MemorySaver： 用于实验性质的记忆检查点。
- SqliteSaver / AsyncSqliteSaver： 使用 `SQLite` 数据库 实现的记忆检查点，适合实验性质和本地工作流程。
- PostgresSaver / AsyncPostgresSaver： 使用 `Postgres` 数据库实现的高级检查点，适合在生产系统中使用。
- 支持自定义检查点。

`checkpointer`是`memory`的一种特定实现，它在执行期间保存图在各个点的状态，使系统能够在中断时从该点恢复。这与 `LangGraph` 中状态的一般概念不同，后者表示应用程序在任何给定时刻的当前快照。虽然状态是动态的并且随着图形的执行而变化，但`checkpointer`提供了一种存储和检索历史状态的方法，从而促进更复杂的工作流程和人机交互。

以 `MemorySaver` 这个实现`checkpointer`的方法为例，帮助大家理解这个过程。
```py
# 导入检查点
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(model="gpt-4o", api_key=key,base_url=base_url,temperature=0,)


class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": response}

def translate_message(state: State):
    system_prompt = """
    Please translate the received text in any language into English as output
    """
    messages = state['messages'][-1]
    messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=messages.content)]
    response = llm.invoke(messages)
    return {"messages": response}

builder = StateGraph(State)

builder.add_node("call_model", call_model)
builder.add_node("translate_message", translate_message)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", "translate_message")
builder.add_edge("translate_message", END)


memory = MemorySaver()
graph_with_memory = builder.compile(checkpointer=memory)   # 在编译图的时候添加检查点
# 当添加了`checkpointer`后，在该图执行的每个超级步骤中会自动创建检查点。即每个节点处理其输入并更新状态后，会当前状态将保存为检查点。但如果像普通图一样，仅传入输入的问题是会报错的

# **当增加了`checkpointer`后，需要`Thread`来作为`checkpointer`保存图中每个检查点的唯一标识，而`Thread`（线程）又是通过`thread_id`来标识某个特定执行线程，所以在使用`checkpointer`调用图时，必须指定`thread_id`，指定的方式是作为配置`configurable`的一部分进行声明。** 正确调用的代码就如下所示：

# 这个 thread_id 可以取任意数值
config = {"configurable": {"thread_id": "1"}}

for chunk in graph_with_memory.stream({"messages": ["你好，我叫西山老师"]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


for chunk in graph_with_memory.stream({"messages": ["请问我叫什么？"]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

**短期记忆可让应用程序记住单个线程或对话中先前的交互，并且可以随时找到某个对话线程中继续之前的问答。**`LangGraph` 将短期记忆作为代理状态的一部分进行管理，并通过线程范围的检查点进行持久化。此状态通常可以包括对话历史记录以及其他状态数据，例如上传的文件、检索的文档或生成的工件。通过将这些存储在图的状态中，程序可以访问给定对话的完整上下文，同时保持不同线程之间的分离。这就是其现实应用价值的体现。

那么接下来要考虑的是： 既然所实际进行存储的是 `Checkpointer`， 那么`Checkpointer`如何去做持久化的存储呢？正如我们上面使用的 `MemorySaver`， 虽然在当前的代码运行环境下可以去指定线程ID，获取到具体的历史信息，但是，一旦我们重启代码环境，则所有的数据都将被抹除。那么一种持久化的方法就是把每个`checkpointer`存储到本地的数据库中。


## 长期记忆
有效的记忆管理可以增强代理维护上下文、从过去的经验中学习以及随着时间的推移做出更明智决策的能力。大多数`AI Agent`构建的应用程序都需要记忆来在多个交互中共享上下文。在 `LangGraph` 中，这种记忆就是通过`checkpointer` 和 `store` 来做持久性，从而添加到任何`StateGraph`中。最常见的用例之一是用它来跟踪对话历史记录，但是也有很大的优化空间，因为随着对话变得越来越长，历史记录会累积并占用越来越多的上下文窗口，导致对大模型的调用更加昂贵和耗时，并且可能会出错。为了防止这种情况发生，我们一般是需要借助一些优化手段去管理对话历史记录，同时更加适配生产环境的` PostgresSaver / AsyncPostgresSaver ）`高级检查点，我们也将随着知识点的进一步补充后，再结合实际的案例进行详细的讲解。

```py
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()
store = InMemoryStore()

model = ...
research_agent = ...
math_agent = ...

workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt="You are a team supervisor managing a research expert and a math expert.",
)

# Compile with checkpointer/store
graph = workflow.compile(
    checkpointer=checkpointer,
    store=store
)

config = {"configurable": {"thread_id": "111"}, "user_id": "8"}
async for chunk in graph.astream({"messages": ["你好，介绍一个你自己"]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```