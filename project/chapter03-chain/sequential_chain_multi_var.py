from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.sequential import SequentialChain
from langchain_core.prompts import ChatPromptTemplate

from util.ModelUtil import get_llm

llm = get_llm()

# Chain 1: 解释技术概念
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一位技术专家"),
    ("human", "请详细解释什么是: {keyword}")
])
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="explanation", verbose=True)

# Chain 2: 多个输入变量（来自 chain1 的输出 + 原始输入）
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译专家"),
    ("human", "原始关键词: {keyword}"),
    ("human", "详细解释: {explanation}"),
    ("human", "请将上述内容翻译成英文")
])
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="translation", verbose=True)

# Chain 3: 接收前两个 chain 的输出
prompt3 = ChatPromptTemplate.from_messages([
    ("system", "你是一个总结高手"),
    ("human", "中文解释: {explanation}"),
    ("human", "英文翻译: {translation}"),
    ("human", "请用50字以内总结")
])
chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="summary", verbose=True)

# SequentialChain: 显式指定输入输出映射
sequential_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["keyword"],  # 初始输入
    output_variables=["summary", "translation"],  # 最终输出
    verbose=True
)

result = sequential_chain.invoke({"keyword": "什么是langchain"})
print("\n=== 最终结果 ===")
print("Summary:", result["summary"])
print("Translation:", result["translation"])
