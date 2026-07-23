import operator
import os
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig

from config import settings

# ---- LLM 初始化 ----
dp_v4 = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
    api_key=settings.deepseek_apikey.get_secret_value(),
    base_url="https://api.deepseek.com",
)

dp_flash = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
    api_key=settings.deepseek_apikey.get_secret_value(),
    base_url="https://api.deepseek.com",
)


# ---- State 定义 ----
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def _call_model(state: AgentState, config: RunnableConfig):
    model_name = config["configurable"].get("model", "flash")
    if model_name == "flash":
        model = dp_flash
    else:
        model = dp_v4
    response = model.invoke(state['messages'])
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("model", _call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="hi,你是谁?")]})
print(result)
config = {"configurable": {"model": "v4"}}
result = graph.invoke({"messages": [HumanMessage(content="hi,你是谁?")]}, config=config)
print(result)
