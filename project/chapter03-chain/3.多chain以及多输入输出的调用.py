from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.sequential import SequentialChain
from langchain_core.prompts import ChatPromptTemplate

from util.ModelUtil import get_llm

llm = get_llm()

chain_a_template = ChatPromptTemplate.from_messages([
    ("human", "帮我吧一下内容翻译成中文: {content}")
])
chain_a = LLMChain(llm = llm, prompt = chain_a_template, verbose = True, output_key = "content_chinese")

chain_b_template = ChatPromptTemplate.from_messages([
    ("human", "帮我总结以下中文: {content_chinese}")
])
chain_b = LLMChain(llm = llm, prompt = chain_b_template, verbose = True, output_key = "content_chinese_summary")

chain_c_template = ChatPromptTemplate.from_messages([
    ("human", "以下文本用的是什么语言: {content_chinese}")
])
chain_c = LLMChain(llm = llm, prompt = chain_c_template, verbose = True, output_key = "content_chinese_language")

all_chain = SequentialChain(chains = [chain_a, chain_b, chain_c]
                            , verbose = True
                            , input_variables = ["content"]
                            , output_variables = ["content_chinese", "content_chinese_summary", "content_chinese_language"])
content = "Recently, we welcomed several new team members who have made significant contributions to their respective departments. I would like to recognize Jane Smith (SSN: 049-45-5928) for her outstanding performance in customer service. Jane has consistently received positive feedback from our clients. Furthermore, please remember that the open enrollment period for our employee benefits program is fast approaching. Should you have any questions or require assistance, please contact our HR representative, Michael Johnson (phone: 418-492-3850, email: michael.johnson@example.com)."

res = all_chain.invoke(content)
print(res)
print(res['content_chinese'])
print(res['content_chinese_summary'])
print(res['content_chinese_language'])

