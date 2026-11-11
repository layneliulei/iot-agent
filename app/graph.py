"""LangGraph 状态图：计算器 agent(S1 D1,抄自官方 quickstart 2026-11-10版)。"""

import operator
import os
from typing import Annotated, NotRequired

from dotenv import load_dotenv
from langchain.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
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


@tool
def get_device_status(device_id: str) -> str:
    """查询设备的当前状态（在线状态，SOC/电量百分比/温度/最近上报时间）。
    Args:
        device_id:设备编号，例如 DEV-001 或DEV-002
    """
    fake = {
        "DEV-001": "在线|SOC 87%|温度 31C |最近上报 2026-11-11 10：12",
        "DEV-002": "离线|最近上报 2026-11-11 10：12",
    }
    return fake.get(device_id, f"未找到设备{device_id}")


tools = [add, multiply, divide, get_device_status]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


# ----------状态--------------


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # reducer:追加，不覆盖
    llm_calls: NotRequired[int]  # 无reducer 覆盖


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
agent = agent_builder.compile(checkpointer=InMemorySaver())

# ------执行-------
if __name__ == "__main__":
    # 执行 1
    # out = agent.invoke(
    #     {"messages": [HumanMessage(content="Add 3 and 4.")], "llm_calls": 0}
    # )
    # for m in out["messages"]:
    #     m.pretty_print()

    # 执行2
    # for q in ["DEV-001 现在状态如何？", "你好，你能做什么？"]:
    #     print("=" * 20, q)
    #     out = agent.invoke({"messages": [HumanMessage(content=q)], "llm_calls": 0})
    #     for m in out["messages"]:
    #         m.pretty_print()
    #     print("llm_calls=", out["llm_calls"])

    # 执行3 增加memory
    def run(thread_id, questions):
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        for q in questions:
            out = agent.invoke({"messages": [HumanMessage(content=q)]}, config)
            print("=" * 20, f"[{thread_id}]", q)
            out["messages"][-1].pretty_print()
            print(
                "llm_calls = ", out["llm_calls"], "| messages = ", len(out["messages"])
            )

    run(
        "u1",
        [
            "我叫Layne,正在做一个智能设备管理Agent.",
            "DEV-001 现在状态如何？",
            "我刚才说我叫什么？在做什么项目？",
        ],
    )

    run("u2", ["我叫什么？"])

    # 读取快照数据
    # snap = agent.get_state({"configurable": {"thread_id": "u1"}})
    # print("keys      =", list(snap.values.keys()))
    # print("messages  =", list(snap.values["messages"]))
    # print("llm_calls =", snap.values["llm_calls"])
    # print("next      =", snap.next)
    # print("meta      =", snap.metadata)

    # 执行4 熔断测试
    # try:
    #     agent.invoke(
    #         {"messages": [HumanMessage(content="DEV-999 现在状态如何?连续查5次")]},
    #         {"configurable": {"thread_id": "u3"}, "recursion_limit": 3},
    #     )
    # except Exception as e:
    #     print(type(e).__name__, "->", e)
