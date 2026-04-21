import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from zai import ZhipuAiClient
import time

_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'llm.env')
load_dotenv(_env_path)


def get_config(key):
    return os.getenv(key)


# 异步调用（实验组）
async def async_test(chatModelAI, message):
    start_time = time.time()
    response = await chatModelAI.ainvoke(message)  # 异步调用
    duration = time.time() - start_time
    print(f"异步调用耗时：{duration:.2f}秒")
    return response

async def sync_test(chatModelAI, message):
    start_time = time.time()
    response = chatModelAI.invoke(message)  # 异步调用
    duration = time.time() - start_time
    print(f"异步调用耗时：{duration:.2f}秒")
    return response


def invoke(message):
    # 初始化客户端
    client = ZhipuAiClient(api_key=get_config('API_KEY'))

    # 创建聊天完成请求
    return client.chat.completions.create(
        model=get_config('model'),
        messages=message,
        temperature=0.6
    )


def get_llm():
    # streaming表示开启流式输出
    return ChatOpenAI(
        api_key=get_config('API_KEY'),
        base_url=get_config('URL'),
        model=get_config('model'),
        timeout=60,
        temperature=1,
        max_tokens=500,
        streaming=True
    )
