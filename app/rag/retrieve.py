"""检索 + 重排：召回求全，重排求准。

retrieve():向量召回top-10(粗排)
rerank():  默认CrossEncoder 逐对打分取前4（精排）
            RERANK=kw 时退化为【汉字bigram 重叠率】打分，README须写明原因

"""

from __future__ import annotations

import os
from typing import Any

from app.rag.index import Q_PREFIX, get_client

RETRIEVE_K = 10
RERANK_K = 4
MIN_RERANK = 0.5  # 低于此分视为「无相关材料」；只对 CrossEncoder 分数生效

_encoder = None
_reranker = None


def get_encoder():
    """模型只加载一次：20条评测要跑40次检索，每次重载bge-small 会白烧几分钟。"""
    global _encoder
    if _encoder is None:
        from app.rag.index import STEncoder

        _encoder = STEncoder()
    return _encoder


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def retrieve(
    question: str, strategy: str = "struct", top_k: int = RETRIEVE_K
) -> list[dict[str, Any]]:
    """向量召回，query侧加前缀、文档侧不加 ----与index.py建库时的口径一致。"""
    vec = get_encoder().encode([Q_PREFIX + question])[0]
    hits = (
        get_client()
        .query_points(
            collection_name=f"manuals_{strategy}",
            query=vec,
            with_payload=True,
            limit=top_k,
        )
        .points
    )
    out: list[dict[str, Any]] = []
    for h in hits:
        p = h.payload or {}
        out.append(
            {
                "text": str(p.get("text", "")),
                "source": p.get("source", "?"),
                "chunk_index": p.get("chunk_index", -1),
                "recall_score": float(h.score),
            }
        )
    return out


def _kw_rerank(
    question: str, candidates: list[dict], top_k: int = RERANK_K
) -> list[dict]:
    """退化方案：汉字bigram重叠率。只在重排模型拿不到时用，README必须写明。"""
    q = question.replace("?", "").replace("？", "")
    grams = {q[i : i + 2] for i in range(len(q) - 1)}
    scored = []
    for c in candidates:
        text = c["text"]
        hit = sum(1 for g in grams if g in text) / max(len(grams), 1)
        scored.append({**c, "rerank_score": hit})
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


def rerank(
    question: str,
    candidates: list[dict],
    top_k: int = RERANK_K,
    min_score: float | None = None,
) -> list[dict]:
    """精排。分数只用于排序与阈值校准，不要当概率写进 README。
    min_score=None 用模块常量 MIN_RERANK；评测复核阈值时传 0.0 拿原始分数。
    """
    if not candidates:
        return []
    if os.getenv("RERANK", "ce") == "kw":
        return _kw_rerank(question, candidates, top_k)
    floor = MIN_RERANK if min_score is None else min_score
    scores = get_reranker().predict([(question, c["text"]) for c in candidates])
    ranked = sorted(
        ({**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)),
        key=lambda c: c["rerank_score"],
        reverse=True,
    )

    if floor > 0:
        ranked = [d for d in ranked if d["rerank_score"] >= floor]
    return ranked[:top_k]
