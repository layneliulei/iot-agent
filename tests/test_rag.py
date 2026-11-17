"""S3 D2：三段各一条。断言只锚关键词，不锚标点/空格/措辞（你自己会改写文案）。"""

import os

import pytest

from app.rag.retrieve import rerank, retrieve


def test_retrieve_returns_payload_fields():
    cands = retrieve("上报间隔是多少?", strategy="struct", top_k=10)
    assert len(cands) == 10
    assert all(str(c["source"]).endswith(".md") for c in cands)
    assert all(c["text"] for c in cands)


def test_rerank_puts_interval_block_first():
    cands = retrieve("上报间隔是多少?", strategy="struct", top_k=10)
    docs = rerank("上报间隔是多少?", cands)
    assert len(docs) == 4
    assert "上报间隔" in docs[0]["text"]


@pytest.mark.skipif(os.getenv("RUN_LLM") != "1", reason="需要 DeepSeek key，默认跳过")
async def test_generate_says_no_material():
    from app.rag.graph import rag_graph

    out = await rag_graph.ainvoke(
        {"question": "这个设备的出厂价是多少?", "strategy": "struct"}
    )
    assert "没有相关信息" in out["answer"]
