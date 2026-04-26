from langchain_core.prompts import ChatPromptTemplate

from util.ModelUtil import get_llm
from langchain_core.output_parsers import StrOutputParser, XMLOutputParser
from langchain_core.output_parsers import JsonOutputParser

llm = get_llm()

response = llm.invoke("你好")
print("common type:", type(response))
print("common result:", response)
print("common result:", response.content)

# StrOutputParser
strParser = StrOutputParser()
strResponse = strParser.invoke(response)
print("StrOutputParser type:", type(strResponse))
print("StrOutputParser result:", strResponse)

# JsonOutputParser
jsonLLMResponse = llm.invoke("你好，人工智能用英文怎么说？问题用q表示，答案用a表示，返回一个JSON格式")
jsonParser = JsonOutputParser()
jsonResponse = jsonParser.invoke(jsonLLMResponse)
print("JsonOutputParser type:", type(jsonResponse))
print("JsonOutputParser result:", jsonResponse)

# XMLOutputParser - 需要让LLM返回XML格式
xMLOutputParser = XMLOutputParser()
xmlPromptTemplate = ChatPromptTemplate.from_template(
    template="回答用户的查询，必须严格按照XML格式返回.\n{format_instructions}\n{query}",
).partial(format_instructions=xMLOutputParser.get_format_instructions())
xmlChain = xmlPromptTemplate | llm | xMLOutputParser
xmlResponse = xmlChain.invoke({"query": "告诉我你的名字和功能"})
print("XMLOutputParser type:", type(xmlResponse))
print("XMLOutputParser result:", xmlResponse)   #dict

# chain调用
chainJsonOutputParser = JsonOutputParser()
chatPromptTemplate = ChatPromptTemplate.from_template(
    template="回答用户的查询.\n{format_instructions}\n{query}\n",
).partial(format_instructions=chainJsonOutputParser.get_format_instructions())
chain = chatPromptTemplate | llm | chainJsonOutputParser
response = chain.invoke({"query": "明天是多少号？天气怎么样？"})
print("chain response:", response)
print("chain type:", type(response))  #dict