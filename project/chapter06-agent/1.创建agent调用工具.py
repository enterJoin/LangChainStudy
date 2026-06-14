from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel

from util.ModelUtil import get_llm


@tool(description="求平方的工具")
def cal_square(num):
    return num * num

class Answer(BaseModel):
    summary: str
    confidence: float

tools=[cal_square]
agent = create_agent(get_llm(), tools=tools, response_format=Answer)

promptTemplate = ChatPromptTemplate.from_messages([
    ("human", "Summarize AI trends")
])

promptValue = promptTemplate.invoke({})
print(promptValue)
# 使用messages: 即可替换使用ChatPromptTemplate格式
result = agent.invoke({"messages": promptValue.to_messages()})
# result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
print(result["structured_response"])