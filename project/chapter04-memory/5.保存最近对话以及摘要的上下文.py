# ConversationTokenBufferMemory    保存最近token数量的上下文（对话内容，不够就直接砍掉）
# ConversationSummaryMemory        摘要形式保存完整对话
# ConversationSummaryBufferMemory  近多少条保存记录记录，之前就都是保存摘要
from langchain_classic.memory import ConversationSummaryBufferMemory

from util.ModelUtil import get_llm

llm = get_llm()


memory = ConversationSummaryBufferMemory(llm=llm, return_messages=True)

memory.save_context({"input": "你好，我的名字叫小明"}, {"output": "很高兴认识你，小明"})
memory.save_context({"input": "李白是哪个朝代的诗人"}, {"output": "李白是唐朝诗人"})
memory.save_context({"input": "唐宋八大家里有苏轼吗？"}, {"output": "有"})

print(memory.load_memory_variables({}))
print(memory.chat_memory.messages)
