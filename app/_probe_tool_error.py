"""验证:工具抛500时,图仍能跑到最终答案,跑完可删."""

import asyncio

from langchain.messages import HumanMessage
from langchain.tools import tool

from app import graph


@tool
def get_device_status(device_id: str) -> str:
    """查询设备的当前状态.

    Args:
        device_id:设备编号,例如 DEV-001.
    """
    raise RuntimeError("upstream 500")


async def main() -> None:
    graph.tools_by_name["get_device_status"] = get_device_status
    out = await graph.agent.ainvoke(
        {"messages": [HumanMessage(content="DEV-001现在状态如何?")]},
        {"configurable": {"thread_id": "probe-err"}},
    )

    print("llm_calls =", out["llm_calls"], "| messages =", len(out["messages"]))
    out["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
