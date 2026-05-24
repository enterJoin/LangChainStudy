from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel,Field


class FieldInfo(BaseModel):
    a:int = Field(description="第1个要加的数")
    b:int = Field(description="第2个要加的数")


@tool(name_or_callable='add_num', description='两个数字相加',return_direct=True,
      args_schema=FieldInfo)
def add_num(a, b) -> int:
    return a + b

print(add_num.name)
print(add_num.description)
print(add_num.args)
print(add_num.return_direct)

print(add_num.invoke({"a": 1, "b": 2}))


def struct_add_num(a,b):
    return a+b

add_num_tool = StructuredTool.from_function(struct_add_num, name="struct_add_num", return_direct=True, description="两个数字相加")

print(add_num_tool.name)
print(add_num_tool.description)
print(add_num_tool.args)
print(add_num_tool.return_direct)