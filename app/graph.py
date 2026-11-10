"""LangGraph 状态图：计算器 agent(S1 D1,抄自官方 quickstart 2026-11-10版)。"""

import operator
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr
from typing_extensions import TypedDict

load_dotenv()

model = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    temperature=0,
)

# --------工具--------------


@tool
def multiply(a: int, b: int) -> int:
    """Multiply 'a' and 'b'.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds 'a' and 'b'.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide 'a' and 'b'.

    Args:
        a: First int
        b: Second int
    """
    return a / b


tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


# ----------状态--------------


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # reducer:追加，不覆盖
    llm_calls: int  # 无reducer 覆盖


# ---------节点-----------


def llm_call(state: MessagesState):
    """模型决定：调工具，还是直接回答。"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a helpful assistant tasked with performing arithmetic "
                            "on a set of inputs. 必须调用工具完成计算，禁止心算。"
                        )
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState):
    """执行模型要求的工具调用，把结果包成 ToolMessage 还回去。"""
    # last = state["messages"][-1]
    # if not isinstance(last, AIMessage):
    #     raise TypeError("tool_node 期望上一条消息是带 tool_calls 的 AIMessage")
    tool_calls = getattr(state["messages"][-1], "tool_calls", None) or []
    result = []
    for tool_call in tool_calls:
        t = tools_by_name[tool_call["name"]]
        observation = t.invoke(tool_call["args"])

        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": result}


def should_continue(state: MessagesState) -> str:
    """上一条消息里有 tool_calls 就回去执行，否则结束。"""
    tool_calls = getattr(state["messages"][-1], "tool_calls", None) or []
    return "tool_node" if tool_calls else END


# --------建图-------------
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()

# ------执行-------
if __name__ == "__main__":
    out = agent.invoke(
        {"messages": [HumanMessage(content="Add 3 and 4.")], "llm_calls": 0}
    )
    for m in out["messages"]:
        m.pretty_print()
