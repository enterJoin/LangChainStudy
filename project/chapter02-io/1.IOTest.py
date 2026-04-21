from langchain_core.messages import SystemMessage, HumanMessage

from util.ModelUtil import get_llm

# 创建对话模型
chatOpenAI = get_llm()

# 创建多种消息模型
#1.SystemMessage 规则或者背景信息
#2.HumanMessage 表示用户输入
#3.AIMessage 存储ai上次回复内容
# TODO 当content传入多个时调用大模型会卡住 后面再找原因
# systemMessage = SystemMessage(content=[{"type":"text", "text":"你是一个正在学习AI变成的人"}, {"type":"text", "text":"你是一个有开发经验的后端程序员"}])
systemMessage1 = SystemMessage(content="你是一个正在学习AI变成的人")
humanMessage = HumanMessage(content="请帮我指定一个学习AI编程的计划")

prompt = [systemMessage1, humanMessage]
# 调用
response = chatOpenAI.invoke(prompt)

# 获取结果
print(response.content)
print(type(response))