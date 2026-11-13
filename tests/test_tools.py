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


def test_create_ticket_idempotent():
    """同一 tenant + idem_key 两次调用只产生一条工单。"""
    from app import graph as g

    g._TICKETS.clear()
    a = g.create_ticket.invoke(
        {"idem_key": "inc-1", "title": "DEV-001 离线", "tenant_id": "t1"}
    )
    b = g.create_ticket.invoke(
        {"idem_key": "inc-1", "title": "DEV-001 离线", "tenant_id": "t1"}
    )
    assert "已创建工单" in a
    assert "幂等命中" in b
    assert len(g._TICKETS) == 1


def test_idem_key_scoped_by_tenant():
    """幂等键按租户隔离:t2 的同一键不会被 t1 的工单吃掉。"""
    from app import graph as g

    g._TICKETS.clear()
    g.create_ticket.invoke({"idem_key": "inc-1", "title": "x", "tenant_id": "t1"})
    g.create_ticket.invoke({"idem_key": "inc-1", "title": "x", "tenant_id": "t2"})
    assert len(g._TICKETS) == 2


async def test_unauthorized():
    """t2 查 t1 的设备:返回拒绝原因,不抛异常,审计新增一条。"""
    from app import graph as g

    g._AUDIT.clear()
    msg = await g.run_tool_safely(
        g.tools_by_name["get_device_status"],
        {
            "name": "get_device_status",
            "args": {"device_id": "DEV-001"},
            "id": "call_p",
            "type": "tool_call",
        },
        tenant_id="t2",
    )
    assert "无权限" in msg.content
    assert "稍后重试" not in msg.content  # 业务拒绝不许教模型重试
    assert msg.status == "error"
    assert any(a["event"] == "PERMISSION_DENIED" for a in g._AUDIT)


def _state(tenant_id: str, streak: dict[str, int] | None = None) -> graph.MessagesState:
    """构造一个「模型要求查 DEV-001」的最小状态,供熔断测试用。"""
    call = {
        "name": "get_device_status",
        "args": {"device_id": "DEV-001"},
        "id": "c1",
        "type": "tool_call",
    }
    st: graph.MessagesState = {
        "messages": [AIMessage(content="", tool_calls=[call])],
        "tenant_id": tenant_id,
        "tool_streak": streak or {},
    }
    return st


async def test_breaker_routes_to_force_stop():
    """连续失败累计 1→2→3，第 3 次后强制收口 —— 连 tool_node 的计数接线一起验证，零 token。"""
    from app import graph as g

    call = {
        "name": "get_device_status",
        "args": {"device_id": "DEV-001"},
        "id": "c1",
        "type": "tool_call",
    }
    base = {"messages": [AIMessage(content="", tool_calls=[call])], "tenant_id": "t2"}
    streak = {}
    for expected in (1, 2, 3):
        out = await g.tool_node(_state("t2", streak))  # t2 查 t1 的设备 → 每次 error
        streak = out["tool_streak"]
        assert streak["get_device_status"] == expected
    assert g.after_tool_node(_state("t2", streak)) == "force_stop"

    out = g.force_stop(_state("t2", streak))
    assert "暂时无法完成" in out["messages"][0].content
    assert out["tool_streak"] == {}

    # 成功一次即清零:同一工具,换成有权限的 t1
    ok = {
        "messages": [AIMessage(content="", tool_calls=[call])],
        "tenant_id": "t1",
        "tool_streak": {"get_device_status": 2},
    }
    out2 = await g.tool_node(_state("t1", {"get_device_status": 2}))
    assert out2["tool_streak"]["get_device_status"] == 0
