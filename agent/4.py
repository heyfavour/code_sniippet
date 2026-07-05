import operator
from typing import Any, Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph, END


# Annotated 允许为类型提供额外的元数据 不影响类型检查对类型本身的理解
class State(TypedDict):
    aggreate: Annotated[list, operator.add]


def a(state: State):  # Node
    print(f"添加A到{state['aggreate']}")
    return {"aggreate": ["A"]}


def b(state: State):  # Node
    print(f"添加B到{state['aggreate']}")
    return {"aggreate": ["B"]}


def route(state: State) -> Literal["b", END]:
    if len(state["aggreate"]) < 7:
        return "b"
    else:
        return END


graph_builder = StateGraph(State)
# node
graph_builder.add_node(a)
graph_builder.add_node(b)
# edge
graph_builder.add_edge(START, "a")
graph_builder.add_conditional_edges("a", route)
graph_builder.add_edge("b", "a")

# compile
graph = graph_builder.compile()
result = graph.invoke({"aggregate": []}, {'configurable': {"thread_id": "foo"}})
print(result)

# 递归限制 recursion_limit限制循环参数

from langgraph.errors import GraphRecursionError
# graph.invoke({"aggregate": []},{"recursion_limit":4})
try:
    graph.invoke({"aggregate": []},{"recursion_limit":4})
except GraphRecursionError:
    print("Recursion error")