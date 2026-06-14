from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.utils.uuid import uuid7
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from util.ModelUtil import get_llm


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model=get_llm(),
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
#     config={"configurable": {"thread_id": str(uuid7())}},
#     context=Context(user_id="user-123"),
# )

for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
}, stream_mode="values", config={"configurable": {"thread_id": str(uuid7())}}, context=Context(user_id="user-123")):
    # Each chunk contains the full state at that point
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
