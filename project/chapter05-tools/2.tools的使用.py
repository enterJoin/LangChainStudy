from langchain_community.tools import MoveFileTool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool

from util.ModelUtil import get_llm

llm = get_llm()

tools=[MoveFileTool()]

functions = [convert_to_openai_tool(tool) for tool in tools]

# llm = llm.bind_tools(tools)
response = llm.invoke(input=[HumanMessage("请帮我把 D://软件//Agent//测试文件 移动到桌面  (优先找能调用的工具，不要觉得没权限就不调)")], functions=functions)
print(response)
print(response.tool_calls)