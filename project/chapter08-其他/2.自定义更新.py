from langchain.agents import create_agent

from util.ModelUtil import get_llm


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=get_llm(),
    tools=[get_weather],
    name="weather_agent",
)

stream = agent.stream_events(
    input,
    version="v3",
    transformers=[ToolActivityTransformer],
)

for activity in stream.extensions["tool_activity"]:
    print(activity)