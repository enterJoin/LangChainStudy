import bs4
import langchain
import openai
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


from util.ModelUtil import invoke, get_llm, get_config

def hello_world_test():
    print(langchain.__version__)
    print(openai.__version__)

    messages = [{"role": "user", "content": "你好 世界"}]
    print(invoke(messages))


def output_parser_test():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个资深大模型应用开发者，请用JSON格式回答"),
        ('user', '{input}')
    ])
    # output_parser = StrOutputParser()
    output_parser = JsonOutputParser()

    chain = prompt | llm | output_parser
    message = chain.invoke({'input': '请简单介绍一下langchain和他的用法'})
    print(message)


def embedding_test():
    import os
    from dotenv import load_dotenv
    load_dotenv("../config/llm.env")

    embeddings = OpenAIEmbeddings(
        model="embedding-2",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/embeddings",
        openai_api_key=get_config("API_KEY")
    )

    loader = WebBaseLoader(
        web_path="https://www.taptap.cn/moment/736556486750111273"    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    print(len(documents))

    # 向量存储  embeddings 会将 documents 中的每个文本片段转换为向量，并将这些向量存储在 FAISS 向量数据库中
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    retriever.search_kwargs = {
        "k"
        : 3}
    docs = retriever.invoke(
        "火花是一个什么宠物？"
    )
    # for i,doc in enumerate(docs):
    #     print(f"⭐第{i+1}条规定：")
    #     print(doc)

    # 6.定义提示词模版
    prompt_template = """
    你是一个问答机器人。
    你的任务是根据下述给定的已知信息回答用户问题。
    确保你的回复完全依据下述已知信息。不要编造答案。
    如果下述已知信息不足以回答用户的问题，请直接回复'我无法回答您的问题'。
    请用中文回答用户问题。
    
    已知信息:
    {info}
    用户问：
    {question}
    """
    # 7.得到提示词模版对象
    template = PromptTemplate.from_template(prompt_template)
    # 8.得到提示词对象
    prompt = template.format(info=docs, question='火花是一个什么宠物？')
    ## 9. 调用LLM
    response = get_llm().invoke(prompt)
    print(response.content)

if __name__ == '__main__':
    embedding_test()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
