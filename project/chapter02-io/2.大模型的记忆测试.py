from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from util.ModelUtil import get_llm

# 结论：大模型没有记忆功能  需要将systemPrompt和AIMessage手动拼接，可以组成上下文

chatOpenAI = get_llm()

systemPrompt = "你的名字叫小智，有一只很可爱的宠物叫皮卡丘"

systemPrompt = SystemMessage(content=systemPrompt)
# humanMessage = HumanMessage(content="你叫什么名字?")
humanMessage = HumanMessage(content="你的宠物叫什么名字?")

messages = [systemPrompt, humanMessage]

response = chatOpenAI.invoke(messages)

print(response.content)
aiMessage = AIMessage(content=response.content)

humanMessage2 = HumanMessage(content="你的宠物叫什么名字?他可爱吗")
message2 = [systemPrompt, aiMessage, humanMessage2]

response2 = chatOpenAI.invoke(message2)
print(response2.content)