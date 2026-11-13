"""LangGraph 状态图：计算器 agent(S1 D1,抄自官方 quickstart 2026-11-10版)。"""

import asyncio
import operator
import os
from typing import Annotated, NotRequired

from dotenv import load_dotenv
from langchain.messages import (
    AIMessage,
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


_TICKETS: dict[str, dict] = {}


@tool
def create_ticket(idem_key: str, title: str, tenant_id: str = "t1") -> str:
    """为当前租户创建一条设备故障工单.同一幂等键重复提交只会生成一条工单.

    Args:
        idem_key:幂等键.同一件事重复提交必须用同一个键.例如 "inc-20261113-001".
        title:工单标题,例如 "DEV-001 离线".
        tenant_id:租户编号,由服务端注入,模型不要自己编.
    """
    key = f"{tenant_id}:{idem_key}"
    if key in _TICKETS:
        t = _TICKETS[key]
        return f"已存在工单{t['no']}(幂等命中,未重复创建)"
    no = f"T{len(_TICKETS) + 1001}"
    _TICKETS[key] = {"no": no, "tenant": tenant_id, "title": title}
    return f"已创建工单 {no}"


tools = [add, multiply, divide, get_device_status, create_ticket]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

# 设备 -> 所属租户(mock).真实系统来自设备台账,不在代码里写死
_DEVICE_TENANT = {"DEV-001": "t1", "DEV-002": "t1", "DEV-003": "t2"}

# 哪些工具要校验[调用者是否有权操作这个对象],以及对象在哪个参数里
_SCOPED = {"get_device_status": "device_id"}

_AUDIT: list[dict] = []  # 只够测试用;生成要logger/落库


def _audit(event: str, **kw) -> None:
    _AUDIT.append({"event": event, **kw})


def authorize(tool_name: str, args: dict, tenant_id: str | None) -> str | None:
    """返回None =放行;返回空字符串=拒绝原因(直接回灌给模型,不抛异常)."""
    field = _SCOPED.get(tool_name)
    if field is None:
        return None
    target = args.get(field)
    if not isinstance(target, str):
        return "设备编号缺失或类型非法,不要重试,如实告知用户无法查询"

    owner = _DEVICE_TENANT.get(target)
    if owner is None:
        return f"设备{target}不存在或编号非法,不要重试,如实告知用户无法查询."
    if owner != tenant_id:
        return (
            f"无权限:设备{target}不属于当前租户,这是业务拒绝,重试不会改变结果."
            f"请如实告知用户无权查询该设备,不要编造结果,也不要换设备绕过."
        )
    return None


async def run_tool_safely(
    tool, tool_call, timeout_s: float = 3.0, tenant_id: str | None = None
) -> ToolMessage:
    """任何工具都从这里进:超时不崩,异常不崩,错误信息回灌给模型.

    status字段是给代码看的(success/error),content是给模型看的
    重试规则(Day 2起):只重试超时/连接失败/5xx;写操作必须带幂等键;4xx与越权不重试

    """
    denied = authorize(tool.name, tool_call["args"], tenant_id)  # 越权在唯一入口拦
    if denied:
        _audit(
            "PERMISSION_DENIED",
            tenant_id=tenant_id,
            tool=tool.name,
            args=tool_call["args"],
        )
        return ToolMessage(content=denied, tool_call_id=tool_call["id"], status="error")
    try:
        async with asyncio.timeout(timeout_s):
            out = await tool.ainvoke(tool_call["args"])
        return ToolMessage(
            content=str(out), tool_call_id=tool_call["id"], status="success"
        )
    except TimeoutError:
        return ToolMessage(
            content=(
                f"工具{tool.name}超过{timeout_s}s 未返回."
                f"请告知用户稍后重试,或改用其他方式获取信息,不要编造结果."
            ),
            tool_call_id=tool_call["id"],
            status="error",
        )
    except Exception as e:
        return ToolMessage(
            content=(
                f"工具{tool.name}执行失败({type(e).__name__})."
                f"请向用户说明信息暂时不可用,不要编造结果."
            ),
            tool_call_id=tool_call["id"],
            status="error",
        )


# ----------状态--------------

MAX_SAME_TOOL = 3  # 同一工具连续3次没有可用结论 -> 强制收口
_NEEDS_OWNER = {"create_ticket"}  # 签名里带tenant_id的工具;其余靠authorize查表校验


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # reducer:追加，不覆盖
    llm_calls: NotRequired[int]  # 无reducer 覆盖
    tenant_id: NotRequired[str]  # 调用方(将来是/chat 的请求头)传入
    tool_streak: NotRequired[dict[str, int]]  # 工具名 -> 连续失败次数


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


async def tool_node(state: MessagesState):
    """执行模型要求的工具调用，把结果包成 ToolMessage 还回去。"""
    # last = state["messages"][-1]
    # if not isinstance(last, AIMessage):
    #     raise TypeError("tool_node 期望上一条消息是带 tool_calls 的 AIMessage")
    tool_calls = getattr(state["messages"][-1], "tool_calls", None) or []
    tenant_id = state.get("tenant_id", "t1")
    streak = dict(state.get("tool_streak") or {})
    result = []
    for tool_call in tool_calls:
        name = tool_call["name"]
        t = tools_by_name[name]
        args = dict(tool_call["args"])
        if name in _NEEDS_OWNER:
            args["tenant_id"] = tenant_id

        msg = await run_tool_safely(t, tool_call, timeout_s=3.0, tenant_id=tenant_id)

        # observation = t.invoke(tool_call["args"])
        # result.append(
        #     ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        # )
        result.append(msg)
        streak[name] = (streak.get(name, 0) + 1) if msg.status == "error" else 0
    return {"messages": result, "tool_streak": streak}


def after_tool_node(state: MessagesState) -> str:
    """连续失败到阈值就不再回模型,直接收口."""
    streak = state.get("tool_streak") or {}
    if any(v >= MAX_SAME_TOOL for v in streak.values()):
        return "force_stop"
    return "llm_call"


def force_stop(state: MessagesState):
    """熔断后的降级回答:不调用模型,零token,确定性."""
    worst = max(
        (state.get("tool_streak") or {}).items(), key=lambda kv: kv[1], default=("?", 0)
    )
    return {
        "messages": [
            AIMessage(
                content=(
                    f"我暂时无法完成:工具{worst[0]} 已连续{worst[1]}次没有返回可用结果."
                    f"建议稍后重试,或提供其他信息以便换一种方式处理."
                )
            )
        ],
        "tool_streak": {},
    }


def should_continue(state: MessagesState) -> str:
    """上一条消息里有 tool_calls 就回去执行，否则结束。"""
    tool_calls = getattr(state["messages"][-1], "tool_calls", None) or []
    return "tool_node" if tool_calls else END


# --------建图-------------
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("force_stop", force_stop)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_conditional_edges(
    "tool_node", after_tool_node, ["llm_call", "force_stop"]
)
agent_builder.add_edge("force_stop", END)
agent = agent_builder.compile(checkpointer=InMemorySaver())

# ------执行-------
if __name__ == "__main__":
    import asyncio

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
    async def run(thread_id, questions):
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        for q in questions:
            out = await agent.ainvoke({"messages": [HumanMessage(content=q)]}, config)
            print("=" * 20, f"[{thread_id}]", q)
            out["messages"][-1].pretty_print()
            print(
                "llm_calls = ", out["llm_calls"], "| messages = ", len(out["messages"])
            )

    async def main() -> None:
        await run(
            "u1",
            [
                "我叫Layne,正在做一个智能设备管理Agent.",
                "DEV-001 现在状态如何？",
                "我刚才说我叫什么？在做什么项目？",
            ],
        )

        await run("u2", ["我叫什么？"])

        # 越权:租户 t2 查 t1 的设备(本文件内直接用模块全局_AUDIT,不要写g._AUDIT)
        _AUDIT.clear()
        config: RunnableConfig = {"configurable": {"thread_id": "t2-x"}}
        out = await agent.ainvoke(
            {
                "messages": [HumanMessage(content="DEV-001现在状态如何?")],
                "tenant_id": "t2",
            },
            config,
        )
        print("llm_calls =", out["llm_calls"], "| messages =", len(out["messages"]))
        out["messages"][-1].pretty_print()
        print("audit =", _AUDIT)

    asyncio.run(main())

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
