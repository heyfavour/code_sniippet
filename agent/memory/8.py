# 持久化
# 短期 长期 持久化
import os
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, MessagesState, START
from config import settings
from langgraph.checkpoint.memory import MemorySaver

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
    api_key=settings.deepseek_apikey.get_secret_value(),
    base_url="https://api.deepseek.com"
)


def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": response}


########################### 使用memorysaver保存中间状态
memory_saver = MemorySaver()

build_graph = StateGraph(MessagesState)
build_graph.add_node("call_model", call_model)
build_graph.add_edge(START, "call_model")
graph = build_graph.compile(checkpointer=memory_saver)
###################################### 线程隔离
config = {"configurable": {"thread_id": "1"}}

input_message = {"role": "user", "content": "hit,i'm wangzx"}
for chunk in graph.stream({"messages": input_message}, config=config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_message = {"role": "user", "content": "我叫什么"}
for chunk in graph.stream({"messages": input_message}, config=config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "2"}}
input_message = {"role": "user", "content": "我叫什么"}
for chunk in graph.stream({"messages": input_message}, config=config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "1"}}
input_message = {"role": "user", "content": "我叫什么"}
for chunk in graph.stream({"messages": input_message}, config=config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
