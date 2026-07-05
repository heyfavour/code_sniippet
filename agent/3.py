import operator
from typing import Any, Annotated
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


def c(state: State):  # Node
    print(f"添加C到{state['aggreate']}")
    return {"aggreate": ["C"]}


def d(state: State):  # Node
    print(f"添加D到{state['aggreate']}")
    return {"aggreate": ["D"]}


graph_builder = StateGraph(State)
# node
graph_builder.add_node(a)
graph_builder.add_node(b)
graph_builder.add_node(c)
graph_builder.add_node(d)
# edge
graph_builder.add_edge(START, "a")
graph_builder.add_edge("a", "b")
graph_builder.add_edge("a", "c")
graph_builder.add_edge("b", "d")
graph_builder.add_edge("c", "d")
graph_builder.add_edge("c", END)
# compile
graph = graph_builder.compile()
result = graph.invoke({"aggregate": []}, {'configurable': {"thread_id": "foo"}})
print(result)
