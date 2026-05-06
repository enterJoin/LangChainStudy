from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.sql.operators import in_op

from util.ModelUtil import get_llm

llm = get_llm()

chatPromptTemplate = ChatPromptTemplate.from_messages(messages=[
    ("system", "你是一个古诗词助手"),
    ("human", "你好，我的问题是: {question}")
])

chain = chatPromptTemplate | llm
while True:
    question = input("请输入你的问题")
    if question == "Q":
        break
    response = chain.invoke({"question": question})

    print(response.content)
    chatPromptTemplate.messages.append(AIMessage(content=response.content))
    chatPromptTemplate.messages.append(HumanMessage(content=question))