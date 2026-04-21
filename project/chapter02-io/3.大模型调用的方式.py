# 阻塞式invoke(或者直接chatOpenAI(message) 底层调用的也是invoke)  流式stream  批量batch  异步形式ainvoke
import time
import asyncio

from langchain_core.messages import SystemMessage, HumanMessage

from util.ModelUtil import get_llm, async_test, sync_test


async def main():

    chatOpenAI = get_llm()

    # 流式调用
    # for chunk in chatOpenAI.stream("请帮我设计一个AI应用开发的学习计划"):
    #     # 一定要加end加以限制，否则会打印很多空字符串
    #     print(chunk.content, end="", flush=True)

    # 批量调用
    print("主函数开始时间: ", time.time())
    message1 = [SystemMessage("你的名字叫小智"), HumanMessage("你的名字是？")]
    message2 = [SystemMessage("你的职业是一名程序员"), HumanMessage("你的职业是？你的名字是？")]
    # response = chatOpenAI.batch([message1, message2])

    # for message in response:
    #     print(message.content)


    # 异步调用
    response1, response2 = await asyncio.gather(sync_test(chatOpenAI, message1),sync_test(chatOpenAI, message2))

    print("主函数结束时间: ", time.time())
    print(response1.content)
    print(response2.content)

if __name__ == '__main__':
    asyncio.run(main())
