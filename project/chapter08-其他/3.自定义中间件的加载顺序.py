import tools
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from util.ModelUtil import get_llm

# 注意填写中间件时，before类型的是按顺序执行的，after类型的是逆序执行的
@before_model
def before_middleware_1(state: AgentState[int], runtime: Runtime):
    print("执行---before_middleware_1")

@before_model
def before_middleware_2(state: AgentState[int], runtime: Runtime):
    print("执行---before_middleware_2")

@before_model
def before_middleware_3(state: AgentState[int], runtime: Runtime):
    print("执行---before_middleware_3")

@after_model
def after_middleware_1(state: AgentState[int], runtime: Runtime):
    print("执行---after_middleware_1")

@after_model
def after_middleware_2(state: AgentState[int], runtime: Runtime):
    print("执行---after_middleware_2")

@after_model
def after_middleware_3(state: AgentState[int], runtime: Runtime):
    print("执行---after_middleware_3")

@tool(description="查询国家或者城市天气")
def get_weather(location: str) -> str:
    print("调用天气工具---get_weather")
    return f"It's always sunny in {location}!"


agent = create_agent(
    model=get_llm(),
    middleware=[before_middleware_1, before_middleware_2, before_middleware_3, after_middleware_1, after_middleware_2, after_middleware_3],
    tools=[get_weather],
)

res = agent.invoke({"messages":[{"role": "user", "content": "What's the weather in China?"}]})
print(res)
