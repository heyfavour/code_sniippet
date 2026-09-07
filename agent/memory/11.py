from typing import Literal
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from config import settings

@tool
def get_weather(city:Literal["北京","深圳"]):
    """
    用于返回天气信息的工具函数
    """
    match city:
        case "北京":
            return "北京天气晴朗 大约22度 湿度30%"
        case "深圳":
            return "深圳暴雨 大约16度 湿度80%"


tools = [get_weather]

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
    api_key=settings.deepseek_apikey.get_secret_value(),
    base_url="https://api.deepseek.com"
)

from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = "mongodb://192.168.31.166:27017/"
with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    response = graph.invoke(
        {"messages": [("human", "北京天气如何？")]}, config
    )
    print(response)
    final_message = response["messages"][-1]
    print(final_message.content)

with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    response = graph.invoke(
        {"messages": [("human", "我刚刚问的哪里？")]}, config
    )
    print(response)
    final_message = response["messages"][-1]
    print(final_message.content)