from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph, state


class State(TypedDict):
    value1: str
    value2: str


def step_1(state: State):  # Node
    value1 = state["value1"]
    return {"value1": value1 + "a"}


def step_2(state: State):
    current_val1 = state["value1"]
    return {"value1": current_val1 + '_step2'}


def step_3(state: State):
    return {"value2": 10}


graph_builder = StateGraph(State)
# node
graph_builder.add_node(step_1)
graph_builder.add_node(step_2)
graph_builder.add_node(step_3)
# edge
graph_builder.add_edge(START, "step_1")
graph_builder.add_edge("step_1", "step_2")
graph_builder.add_edge("step_2", "step_3")
# compile
graph = graph_builder.compile()
result = graph.invoke({"value1": "c"})
print(result)
