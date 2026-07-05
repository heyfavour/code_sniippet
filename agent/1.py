from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


# 节点间通讯的消息格式
class State(TypedDict):
    message: list[AnyMessage]
    extra_field: int


from langchain_core.messages import AIMessage


def node(state: State):
    message = state['message']
    new_message = AIMessage('这是一个新的节点')
    return {
        'message': message + [new_message],
        'extra_field': 1,
    }


from langgraph.graph import StateGraph

graph = StateGraph(State)
graph.add_node(node)
graph.set_entry_point("node")  # 设置入口
graph_builder = graph.compile()

# mermaid

from langchain_core.messages import HumanMessage

result = graph_builder.invoke(
    {
        "message": [HumanMessage("你好，我是wangzx"),]
    }
)
#print(result)

for message in result['message']:
    message.pretty_print()
