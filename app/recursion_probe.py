"""确定性熔断探针：不调用LLM，只证明recursion_limli 会抛 Graph Recursion Error。"""

from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


class S(TypedDict):
    n: int


def a(state: S):
    return {"n": state["n"] + 1}


def b(state: S):
    return {"n": state["n"] + 1}


g = StateGraph(S)
g.add_node("a", a)
g.add_node("b", b)
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", "a")  # 环，永远进不了 END
probe = g.compile()

if __name__ == "__main__":
    try:
        probe.invoke({"n": 0}, {"recursion_limit": 3})
    except Exception as e:
        print("type = ", type(e).__name__)
        print("msg  =", e)
