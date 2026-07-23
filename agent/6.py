# map reduce
import operator
import os
from typing import Annotated, Sequence

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_deepseek import ChatDeepSeek
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

from config import settings

subject_prompt = "请列出2到5个属于「{topic}」这个主题的具体子类别（例如：猫、狗、鸟等），每个子类别用逗号分隔，不要返回主题本身，只返回具体的子类别名称"
joke_prompt = "生成一个关于{subject}的笑话"
best_joke_prompt = "以下是关于{topic}的笑话，每个笑话有一个从0开始的编号（0, 1, 2...）。选出一个最好的一个！返回最佳笑话的编号   {jokes}"


class Subjects(BaseModel):
    subjects: list[str] = Field(description="2到5个具体的子类别名称，不能与主题本身相同", min_length=2, max_length=5)


class Joke(BaseModel):
    joke: str


class BestJoke(BaseModel):
    id: int = Field(description="Best joke ID", ge=0)

# chat才支持with_structured_output
# v4不支持 如果想用flash则自己解析
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=settings.deepseek_apikey.get_secret_value(),
    base_url="https://api.deepseek.com"
)


class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


class JokeState(TypedDict):
    subject: str


def generate_topics(state: OverallState):
    prompt = subject_prompt.format(topic=state["topic"])
    # 模型使用结构化输出
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}


def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}


def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": subject}) for subject in state["subjects"]]


def best_joke(state: OverallState):
    jokes_list = state["jokes"]
    jokes = "\n\n".join(jokes_list)
    prompt = best_joke_prompt.format(topic=state['topic'], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    # 防止模型返回的ID越界（如使用了1-based索引）
    return {"best_selected_joke": state["jokes"][response.id]}


graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)
app = graph.compile()


for subject in app.stream({"topic":"动物"}):
    print(subject)
