import asyncio

from langchain.messages import AIMessage
from langchain.tools import tool

from app import graph


@tool
async def slow_tool() -> str:
    """一个永远来不及的工具"""
    await asyncio.sleep(5)
    return "done"


@tool
def boom_tool() -> str:
    """一个永远失败的工具"""
    raise RuntimeError("upstream 500")


def _call(name: str, call_id: str = "call_1") -> dict:
    return {"name": name, "args": {}, "id": call_id, "type": "toll_call"}


async def test_tool_timeout():
    msg = await graph.run_tool_safely(slow_tool, _call("slow_tool"), timeout_s=0.5)
    assert "超过0.5s" in msg.content
    assert msg.tool_call_id == "call_1"


async def test_tool_500():
    msg = await graph.run_tool_safely(boom_tool, _call("boom_tool"), timeout_s=3.0)
    assert "执行失败(RuntimeError)" in msg.content
    assert msg.tool_call_id == "call_1"


async def test_tool_node_survives_failure(monkeypatch):
    """工具炸了,tool_node仍返回一条ToolMessages --- 不抛异常,图就不会断."""
    monkeypatch.setitem(graph.tools_by_name, "boom_tool", boom_tool)
    state: graph.MessagesState = {
        "messages": [AIMessage(content="", tool_calls=[_call("boom_tool")])]
    }
    out = await graph.tool_node(state)
    assert "执行失败" in out["messages"][0].content
