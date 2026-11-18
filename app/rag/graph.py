"""RAG 三段图：retrieve -> rerank -> generate。独立于 app/graph.py 的工具 agent。"""

import os
from typing import NotRequired

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr
from typing_extensions import TypedDict

from app.rag.retrieve import RERANK_K, rerank, retrieve

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    temperature=0,
)

SYSTEM = (
    "你只能依据「材料」回答。材料中没有的信息，直接回答「材料中没有相关信息」，"
    "不要推测、不要补充常识。若材料按不同设备/型号分别给出答案，就分别列出，"
    "不要因为问题宽泛而整体拒答。回答末尾列出用到的出处（文件名#块号）。"
)


class RagState(TypedDict):
    question: str
    strategy: NotRequired[str]
    candidates: NotRequired[list[dict]]
    docs: NotRequired[list[dict]]
    answer: NotRequired[str]
    usage: NotRequired[dict]


def retrieve_node(state: RagState):
    """第一段：召回只看 question，不塞进 prompt。"""
    return {
        "candidates": retrieve(
            state["question"], strategy=state.get("strategy", "struct")
        )
    }


def rerank_node(state: RagState):
    """第二段：只对召回结果精排，成本花在少数候选上。"""
    return {"docs": rerank(state["question"], state.get("candidates") or [], RERANK_K)}


def build_context(docs: list[dict]) -> str:
    blocks = [f"[{d['source']}#{d['chunk_index']}]\n{d['text']}" for d in docs]
    return "\n\n---\n\n".join(blocks)


async def generate_node(state: RagState):
    docs = state.get("docs") or []
    if not docs:
        return {
            "answer": "材料中没有相关信息。",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    msg = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(
                content=f"材料：\n{build_context(docs)}\n\n问题：{state['question']}"
            ),
        ]
    )
    usage = getattr(msg, "usage_metadata", None) or {}
    if not usage:  # 某些版本把用量放在 response_metadata.token_usage
        usage = (getattr(msg, "response_metadata", None) or {}).get("token_usage") or {}
    return {"answer": msg.text, "usage": dict(usage)}


builder = StateGraph(RagState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("rerank", rerank_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate")
builder.add_edge("generate", END)
rag_graph = builder.compile()

if __name__ == "__main__":
    import asyncio

    QUERIES = [
        "设备离线了怎么处理?",
        "上报间隔是多少?",
        "这个设备的出厂价是多少?",
    ]

    async def _main() -> None:
        # 三条查询共用一个事件循环：asyncio.run 每次新建并关闭 loop，
        # 而 llm 的 httpx 异步连接池绑在第一个 loop 上 -> 第二轮 Event loop is closed
        for q in QUERIES:
            out = await rag_graph.ainvoke({"question": q, "strategy": "struct"})
            print(f"\nQ: {q}")
            for d in out.get("docs") or []:
                print(
                    f"  {d['rerank_score']:+.4f}  {d['source']}#{d['chunk_index']}"
                    f"  recall={d['recall_score']:.4f}"
                )
            print(f"A: {out.get('answer')}")

    asyncio.run(_main())
