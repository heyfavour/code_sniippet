# 跨线程查询 不同对话
# 不太理解为啥实现方式这么蠢，下次再看
import uuid

from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.store.memory import InMemoryStore

from config import settings

from langchain_openai import OpenAIEmbeddings

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
    api_key=settings.deepseek_apikey.get_secret_value(),
    base_url="https://api.deepseek.com"
)



###########################
# 内存堆栈 不同线程调用
in_memory_store = InMemoryStore(
    index={
        "embed":OpenAIEmbeddings(
            model = "Pro/BAAI/bge-m3",
            api_key = settings.deepseek_apikey.get_secret_value(),
            base_url = "https://api.deepseek.com"
        ),
        "dims":1024,
    }
)


def call_model(state:MessagesState,config:RunnableConfig,store:InMemoryStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories",user_id)
    memories = store.search(namespace,query=str(state.messages[-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"你是一个正在与用户交谈的助手，用户信息{info}"
    last_message = state.messages[-1]
    if "记住" in last_message.content or "remember" in last_message.content:
        # 硬编码一个记忆
        memory = "用户名字是xxx"
        store.put(namespace, str(uuid.uuid1()), {"data":memory})
    response =  model.invoke(
        [{"role":"system","content":system_msg}] + state["messages"]
    )
    return {"messages":response}

builder = StateGraph(MessagesState)
builder.add_node("call_model",call_model)
builder.add_edge(START,"call_model")

graph = builder.compile(checkpointer=MemorySaver(),store=in_memory_store)


config = {"configurable":{"thread_id":1,"user_id":"123456"}}
input_message = {"role":"user","content":"请记住我的名字叫xxx"}
for chunk in graph.stream({"messages":input_message},config,stream_mode="values"):
    chunk["messages"][-1].pretty_print()

