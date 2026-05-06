from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from util.ModelUtil import get_llm

memory = ChatMessageHistory()
llm = get_llm()

while True:
    question = input("请输入你的问题：")
    if question == "Q":
        break
    memory.add_user_message(question)
    response = llm.invoke(memory.messages)

    print(response.content)
    memory.add_ai_message(response.content)
