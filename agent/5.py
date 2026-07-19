"""
LangGraph 多 Agent 系统
- Supervisor：路由决策，决定由哪个专家处理
- Researcher：负责调研分析
- Writer：负责撰写总结
"""

from typing import Literal, Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from config import settings

# ---- LLM 初始化 ----
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=settings.deepseek_apikey.get_secret_value(),
    base_url="https://api.deepseek.com",
)


# ---- State 定义 ----
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str  # supervisor 决定下一步交给谁


# ---- Agent 节点 ----

def supervisor(state: AgentState) -> dict:
    """主管：根据用户问题决定交给 researcher 还是 writer"""
    system_prompt = SystemMessage(content=(
        "你是一个任务调度主管。根据用户的问题，决定交给哪个专家处理。\n"
        "如果问题需要调研、分析、解释原理 → 回复 researcher\n"
        "如果问题需要撰写、改写、总结 → 回复 writer\n"
        "只回复一个词：researcher 或 writer"
    ))
    response = llm.invoke([system_prompt] + state["messages"])
    choice = response.content.strip().lower()
    if "researcher" in choice:
        next_agent = "researcher"
    elif "writer" in choice:
        next_agent = "writer"
    else:
        next_agent = "researcher"  # 默认
    print(f"[Supervisor] 决策 → {next_agent}")
    return {"next_agent": next_agent}


def researcher(state: AgentState) -> dict:
    """调研专家：分析和解释问题"""
    system_prompt = SystemMessage(content=(
        "你是一位资深研究员。请对用户的问题进行深入分析和解释，"
        "提供有条理、有深度的回答。"
    ))
    response = llm.invoke([system_prompt] + state["messages"])
    print("[Researcher] 完成调研")
    return {"messages": [AIMessage(content=response.content)]}


def writer(state: AgentState) -> dict:
    """写作专家：撰写和总结内容"""
    system_prompt = SystemMessage(content=(
        "你是一位专业写手。请根据用户的问题，撰写清晰、简洁、有结构的回答。"
    ))
    response = llm.invoke([system_prompt] + state["messages"])
    print("[Writer] 完成撰写")
    return {"messages": [AIMessage(content=response.content)]}


# ---- 路由函数 ----

def route_supervisor(state: AgentState) -> Literal["researcher", "writer"]:
    """根据 supervisor 的决策路由到对应 agent"""
    return state["next_agent"]


# ---- 构建 Graph ----

graph = StateGraph(AgentState)

# 添加节点
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

# 添加边
graph.add_edge(START, "supervisor")            # 入口 → supervisor
graph.add_conditional_edges("supervisor", route_supervisor)  # supervisor → 专家
graph.add_edge("researcher", END)              # 专家完成后结束
graph.add_edge("writer", END)

# 编译
app = graph.compile()


# ---- 运行 ----

if __name__ == "__main__":
    print("=== LangGraph 多 Agent 系统 ===\n")

    # 测试1：调研类问题
    print("--- 测试1：调研类问题 ---")
    result = app.invoke({
        "messages": [HumanMessage(content="请解释一下什么是 RAG（检索增强生成）？")],
        "next_agent": "",
    })
    print(f"\n最终回复:\n{result['messages'][-1].content}\n")

    # 测试2：写作类问题
    print("--- 测试2：写作类问题 ---")
    result = app.invoke({
        "messages": [HumanMessage(content="帮我写一封给客户的感谢信，感谢他们的合作。")],
        "next_agent": "",
    })
    print(f"\n最终回复:\n{result['messages'][-1].content}")
