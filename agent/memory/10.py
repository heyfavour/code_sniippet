# reAct
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState,StateGraph,START,END
from langgraph.prebuilt import ToolNode
from config import settings

# 内存记忆
memory = MemorySaver()

@tool
def search(query:str):
    """调用此函数可以浏览网络"""
    return "北京天气晴朗 大约22度 湿度30%"

tools = [search]
tool_node = ToolNode(tools)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
    api_key=settings.deepseek_apikey.get_secret_value(),
    base_url="https://api.deepseek.com"
)
bound_model = model.bind_tools(tools)

def should_continue(state:MessagesState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "action"


def call_model(state:MessagesState):
    response = bound_model.invoke(state["messages"])
    return {"messages": response}

workflow = StateGraph(MessagesState)

workflow.add_node("agent",call_model)
workflow.add_node("action",tool_node)

workflow.add_edge(START,"agent")
workflow.add_conditional_edges("agent",should_continue,["action",END])
workflow.add_edge("action","agent")

app = workflow.compile(checkpointer=memory)


# http://192.168.31.166:27017/
# pip install pymongo langgrapgh langgraph-checkpoint-mongodb
app.invoke([
    {"role":"user","content":"北京天气如何"},
    {"role":"assistant","content":"请调用search函数"},
    {"role":"tool_call","function_call":{"name":"search","arguments":{"query":"北京天气"}}},
    {"role":"user","content":"北京天气晴朗 大约22度 湿度30%"},
])


