from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"

# 所有会话，信息的快照
backend = StateBackend()

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),   # 文件读取中间件
        TodoListMiddleware(),                    #任务清单中间件
        SubAgentMiddleware(                      # 创建子agent中间件
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "tools": [search],
                    "model": "anthropic:claude-sonnet-4-6",
                    "middleware": [],
                }
            ],
        ),
    ],
)