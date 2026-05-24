from langchain_classic.chains.llm import LLMChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from util.ModelUtil import get_llm

template = """以下是人类与AI之间的友好对话描述。AI表现得很健谈，并提供了大量来自其上下文的
具体细节。如果AI不知道问题的答案，它会表示不知道。
当前对话：
{history}
Human: {question}
AI:"""

promptTemplate = PromptTemplate.from_template(template=template)

llm = LLMChain(llm=get_llm(), prompt=promptTemplate, memory=ConversationBufferWindowMemory(k=1), verbose=True)


response1 = llm.invoke({"question":"你好，我是孙小空"})
print(response1)
response2 = llm.invoke({"question":"你好，今天是晴天，天气真不错"})
print(response2)

response3 = llm.invoke({"question":"你好，这次考试我考的很好，快表扬一下我"})
print(response3)

response4 = llm.invoke({"question":"你好，我叫什么名字？"})
print(response4)
