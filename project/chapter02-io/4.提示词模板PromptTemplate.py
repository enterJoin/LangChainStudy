from langchain_core.prompts import PromptTemplate

from util.ModelUtil import get_llm

promptTemplate = PromptTemplate(template="你是一位{role}，你的名字是{name}", input_variables=["role", "name"])
print(promptTemplate)

res = promptTemplate.format(role="程序员", name="小智")
print(res)


#推荐使用PromptTemplate.from_template创建提示词模板
promptTemplate2 = PromptTemplate.from_template(template="你是一位{role}，你的名字是{name}")
res2 = promptTemplate2.format(role="程序员2", name="小智2")
print(res2)


# 没有参数则format的时候不需要
promptTemplate3 = PromptTemplate.from_template(template="今天天气真好！")
res3 = promptTemplate3.format()
print(res3)


# 可给予默认提示词模板变量  但是后面仍可给该变量赋值
promptTemplate4 = PromptTemplate.from_template(template="你是一位{role}，你的名字是{name}", partial_variables={"role": "学生"})
res4 = promptTemplate4.format(name="小智2")
# res4 = promptTemplate4.format(role="程序员2", name="小智2")
print(res4)

# 可以在创建模板后赋予默认值  后面可直接跟字符串
promptTemplate5 = (PromptTemplate.from_template(template="你是一位{role}，你的名字是{name}").partial(role="高中生").partial(name="小高")
                   + "，学习成绩还不错")
print(promptTemplate5)

## 使用invoke赋值
promptTemplate6 = PromptTemplate.from_template(template="你是一位{role}，你的名字是{name}") + "专业技能很强，会{skill},算是一名优秀的{role}吗"
promptValue = promptTemplate6.invoke({"role": "电工", "name": "小强", "skill": "修理，创造，拼接"})

chatOpenAI = get_llm()
response = chatOpenAI.invoke(promptValue)
print("答复: ", response.content)

#流式复习
for content in chatOpenAI.stream(promptValue):
    print(content.content, end="", flush=True)