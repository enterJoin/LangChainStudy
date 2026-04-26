from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.sequential import SimpleSequentialChain
from langchain_core.prompts import ChatPromptTemplate

from util.ModelUtil import get_llm

llm = get_llm()

# chatPromptTemplate = ChatPromptTemplate.from_messages([
#     ("system", "你是一位优秀的{role}"),
#     ("human", "我的问题是{question}")
# ])
# # verbose打印过程！！！
# llmChain = LLMChain(prompt=chatPromptTemplate, llm=llm, verbose=True)
# response = llmChain.invoke({"role": "程序员", "question": "怎么转型AI-Agent开发"})
# print(response)

#唯一输入输出 多次调用 chain的使用
chatPrompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一位技术专家"),
    ("human", "请详细解释什么是: {keyword}")
])
chain1 = LLMChain(llm=llm, prompt=chatPrompt1, verbose=True)
response = chain1.invoke(input={"keyword", "什么是langchain"})
print(response)
chatPrompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个总结高手"),
    ("human", "这是一个经过提问的回答: {answer}"),
    ("human", "请用50字以内总结该文本")
])
chain2 = LLMChain(llm=llm, prompt=chatPrompt2, verbose=True)
simpleSequentialChain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
print(simpleSequentialChain.invoke(input={"keyword", "什么是langchain"}))

#多次输入 多次输出（调用大模型次数不同）