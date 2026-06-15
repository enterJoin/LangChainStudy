from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar":[]}, config)


# get the latest state snapshot
config = {"configurable": {"thread_id": "1"}}
print(graph.get_state(config))

# get a state snapshot for a specific checkpoint_id
end_check_point_id = config.get("configurable").get("checkpoint_id")
config = {"configurable": {"thread_id": "1", "checkpoint_id": end_check_point_id}}
print(graph.get_state(config))

graph.update_state(config, {"foo": "aaa"})
print(graph.get_state(config))
print("update config: ", graph.get_state(config))

# 获取配置历史
print("history config: ")
for config in graph.get_state_history(config):
    print(config)