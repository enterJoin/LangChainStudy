# 多chain的使用
from langchain_classic.chains.sequential import SequentialChain
from langchain_core.prompts import ChatPromptTemplate
from util.ModelUtil import get_llm
from langchain_classic.chains.llm import LLMChain

llm = get_llm()

chain_a_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个运动新闻实时关注者"),
    ("human", "请详细解释什么是: {keyword}，并且为什么{why}?")
])
chain_a = LLMChain(llm = llm, prompt = chain_a_template, verbose = True, output_key = "a_output")

chain_b_template = ChatPromptTemplate.from_messages([
    ("system", "你一个文本总结高手"),
    ("human", "请总结文本: {a_output}"),
    ("human", "请用100字以内总结该文本")
])

chain_b = LLMChain(llm = llm, prompt = chain_b_template, verbose = True, output_key = "b_output")

all_chain = SequentialChain(chains = [chain_a, chain_b]
                            , verbose = True
                            , input_variables = ["keyword", "why"]
                            , output_variables = ["a_output", "b_output"])

res = all_chain.invoke({"keyword": "为什么国足打的这么烂？", "why": "还那么挣钱"})
print(res)
print(res.a_output)
print(res.b_output)

