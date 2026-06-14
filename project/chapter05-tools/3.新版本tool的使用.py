from langchain.tools import tool

from util.ModelUtil import get_llm


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model_with_tools = get_llm().bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather like in abcde?")
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")