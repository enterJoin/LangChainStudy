from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    MessagesPlaceholder, FewShotPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate, \
    ChatMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings

from util.ModelUtil import get_llm, get_config

# 注意ChatPromptTemplate创建模板message数组的每一个元素是一个元组
# chatPromptTemplate = ChatPromptTemplate(messages=[
#     ("system", "你是一名{role}"),
#     ("human", "问题是：{question}")
# ])

# chatPromptTemplate = ChatPromptTemplate.from_messages([
#     ("system", "你是一名{role}"),
#     ("human", "问题是：{question}")
# ])
# 只有invoke方法才需要传字典  其他都是需要什么参数声明即可
# response = chatPromptTemplate.invoke(input={"role": "后端开发工程师", "question": "怎么才能不写出bug"})
# ## ChatPromptValue
# print(type(response))
#
# formatResponse = chatPromptTemplate.format(role="后端开发工程师", question="怎么才能不写出bug")
# # str
# print(type(formatResponse))
# print(formatResponse)

# formatMessageResponse = chatPromptTemplate.format_messages(role="后端开发工程师", question="怎么才能不写出bug")
# print(formatMessageResponse)
# print(type(formatMessageResponse))  #list  数组放着两种消息System,Human

# formatPromptMessage = chatPromptTemplate.format_prompt(role="后端开发工程师", question="怎么才能不写出bug")
# print(formatPromptMessage)
# print(type(formatPromptMessage))   #ChatPromptValue 与invoke一样
#
# #ChatPromptValue 如何转化为消息数组和字符串
# print(formatPromptMessage.to_messages())
# print(formatPromptMessage.to_string())


# ChatPromptTemplate = ChatPromptTemplate.from_template()

# 更多方式创建消息提示词模板
# 1.如果数组中只有一个字符串则默认是HumanMessage
# chatPromptTemplate = ChatPromptTemplate.from_messages(["你好，我的问题时{question}"])
# promptValue = chatPromptTemplate.invoke(input={"question": "你叫什么名字？"})
# print(promptValue)
# print(chatPromptTemplate)

# 2.使用字典
# chatPromptTemplate = ChatPromptTemplate.from_messages(messages=[
#     {"role": "system", "content": "你的一名{role}"},
#     {"role": "human", "content": "我的问题时{question}"}
# ])
#
# print(chatPromptTemplate)
# promptValue = chatPromptTemplate.invoke(input={"role":"警察", "question":"警察每天都在干嘛?"})
# print(promptValue)

# 3.使用BaseMassage的构造函数
# chatPromptTemplate = ChatPromptTemplate.from_messages(messages=[
#     SystemMessage("你是一名{role}"),
#     HumanMessage("我的问题时{question}")
# ])
# print(chatPromptTemplate)

# 4.分开
# systemChatPromptTemplate = ChatPromptTemplate.from_messages(messages=[("system", "你是一名{role}")])
# humanChatPromptTemplate = ChatPromptTemplate.from_messages(messages=[("human", "我的问题时{question}")])
#
# chatPromptTemplate = ChatPromptTemplate.from_messages(messages=[systemChatPromptTemplate, humanChatPromptTemplate])
# print(chatPromptTemplate)
# promptValue = chatPromptTemplate.invoke(input={"role":"警察", "question":"警察每天都在干嘛?"})
# print(promptValue)

# 5.使用System与Human的消息模板  多个消息模板组合
####注意：如果有变量的情况一定要使用模板（不管是System还是Human的消息！！！！！！！）
# systemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(template="你是一名{role}")
# humanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(template="我的问题时{question}")
# chatPromptTemplate = ChatPromptTemplate.from_messages(messages=[systemMessagePromptTemplate, humanMessagePromptTemplate])
# print(chatPromptTemplate)
# promptValue = chatPromptTemplate.invoke(input={"role":"警察", "question":"警察每天都在干嘛?"})
# print(promptValue)


# 关于MessagePlaceHolder的使用！！！（可动态记录AIMessage）
# systemMessagePromptTemplate = SystemMessagePromptTemplate.from_template("你是一名{role}")
# humanMessagePromptTemplate = HumanMessagePromptTemplate.from_template("我的问题是：{question}")
#
# chatPromptTemplate = ChatPromptTemplate.from_messages([
#     systemMessagePromptTemplate,
#     MessagesPlaceholder("history"),
#     humanMessagePromptTemplate
# ])
#
# promptValue = chatPromptTemplate.invoke(input = {
#     "role": "资深后端开发",
#     "history": [HumanMessage("程序员越变越强的表现是什么？"), AIMessage("头发变得越来越少")],
#     "question": "我刚才问了什么问题？"
# })
#
# chatAIModel = get_llm()
# aIMessage = chatAIModel.invoke(input=promptValue)
# print(aIMessage.content)

# 关于FewShotPromptTemplate  创建提示词模板，让AI根据提示词模板形式回答问题
# template = "根据一位优秀的数学老师的回答: input: {input}, output: {output}, 世界上最满意的description: {description}"
# promptTemplate = PromptTemplate.from_template(template)
# mathExamples = [
#     {"input": "1 + 1", "output": " = 2", "description": "这是一个普遍的加法运算"},
#     {"input": "1 - 1", "output": " = 0", "description": "这是一个稍微有难度的减法运算"}
# ]
# fewShotPromptTemplate = FewShotPromptTemplate(
#     example_prompt=promptTemplate,
#     examples=mathExamples,
#     suffix="input: {input}, output: ",
#     input_variables=["input"],
# )
# promptValue = fewShotPromptTemplate.invoke({"input": "100 * 20"})
# print(promptValue)


# 关于FewShotChatMessagePromptTemplate的使用
# examples = [
#     {"input": "2---2", "output": "4"},
#     {"input": "2---3", "output": "8"},
# ]
# example_prompt = ChatPromptTemplate(messages = [
#     ("system", "你是一个找规律专家"),
#     ("human", "{input}是多少？"),
#     ("ai", "输出结果时：{output} 规律找对了吗？")
# ])
# #使用FewShotChatMessagePromptTemplate时 可以校准example_prompt对应的参数在examples中的值
# fewShotChatMessagePromptTemplate = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
#
# #将FewShotChatMessagePromptTemplate传入ChatPromptTemplate,ChatPromptTemplate里面设置的消息类型对应的值还生效
# ChatPromptTemplate = ChatPromptTemplate.from_messages(messages=[
#     ("system", "你是一个数学奇才"),
#     fewShotChatMessagePromptTemplate,
#     ("human", "{input}"),
# ])
# promptValue = ChatPromptTemplate.invoke(input={"input": "2---4"})
# chatAIModel = get_llm()
# aIMessage = chatAIModel.invoke(promptValue)
# print(aIMessage.content)


# 示例选择器（当样本数量且类型比较多，不可能全都抛给大模型，而是找相似度比较高的 余弦相似度，长度，最大边际相关示例选择（优先选择语义相似，同时惩罚避免返回同质化内容））
example_prompt = ChatPromptTemplate.from_template(
    template="Input: {input}\nOutput: {output}",
)
# 3.创建一个示例提示词模版
examples = [
    {"input": "高兴", "output": "悲伤"},
    {"input": "高", "output": "矮"},
    {"input": "长", "output": "短"},
    {"input": "精力充沛", "output": "无精打采"},
    {"input": "阳光", "output": "阴暗"},
    {"input": "粗糙", "output": "光滑"},
    {"input": "干燥", "output": "潮湿"},
    {"input": "富裕", "output": "贫穷"},
]
# 4.定义嵌入模型(需要单独购买)
embeddings = OpenAIEmbeddings(
    model="embedding-2",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=get_config("API_KEY")
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    FAISS,
    k=2,
)
# 或者
# example_selector = SemanticSimilarityExampleSelector(
# examples,
# embeddings,
# FAISS,
# k=2
# )
fewShotChatMessagePromptTemplate = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    examples=examples
)

promptValue = fewShotChatMessagePromptTemplate.invoke(input={"input": "精神抖擞"})
print(promptValue)

#最后还可以从文件中读取模板！！！