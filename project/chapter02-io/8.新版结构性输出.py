from langchain.agents.structured_output import ToolStrategy, StructuredOutputValidationError, \
    MultipleStructuredOutputsError
from pydantic import BaseModel, Field
from langchain.agents import create_agent

from util.ModelUtil import get_llm

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"

## 注意ToolStrategy 还可以指定错误形式handle_errors=(ValueError, TypeError)/str/custom_error_handler(自定义异常)等等
class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent = create_agent(
    model=get_llm(),
    response_format=ToolStrategy(schema=ContactInfo, handle_errors="Please provide a valid rating between 1-5 and include a comment.")  # Auto-selects ProviderStrategy
)

# result = agent.invoke({
#     "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
# })
result = agent.invoke({
    "messages": [{"role": "user", "content": "你是谁？"}]
})

print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')