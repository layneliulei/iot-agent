"""S3 D3：20 条评测集 → 命中率 / 正确率 / 拒答率 / P95 / 平均 token。

口径（写进 README 时照抄）：
  命中率@4 = gold_source 出现在进 prompt 的前 4 块里（分母只算 17 条正样本）
  正确率   = 答案里出现该条任一 gold_kw（下界口径，不判措辞）
  拒答率   = 3 条负样本答出「没有相关信息」的比例
  P95      = 20 条端到端耗时的 95 分位（n=20 时取第 19 名，近似最大值）
  token    = 响应 usage.total_tokens；走零 token 确定性分支的记 0

用法（仓库根目录，PowerShell）：
    python -m eval.run --config both      # 手册要求的两种对照
    python -m eval.run --config all       # 加跑 struct 不重排，用来分离变量
    python -m eval.run --config naive
    python -m eval.run --config recursive_rerank
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from pathlib import Path

from app.rag.graph import generate_node
from app.rag.retrieve import MIN_RERANK, RERANK_K, rerank, retrieve

DATASET = Path("eval/dataset.jsonl")
OUT = Path("eval/results.json")

CONFIGS = {
    # A：朴素切块 + 不重排（手册 Day1 的朴素方案）
    "naive": {"strategy": "naive", "use_rerank": False},
    # B：结构切块 + 不重排（分离「切片」与「重排」两个变量，可选）
    "struct_only": {"strategy": "struct", "use_rerank": False},
    # C：结构切块 + CrossEncoder 重排 + 阈值过滤（今天的主配置）
    "recursive_rerank": {"strategy": "struct", "use_rerank": True},
}
GROUPS = {
    "both": ["naive", "recursive_rerank"],
    "all": ["naive", "struct_only", "recursive_rerank"],
}


def norm(s: str) -> str:
    """比对前去掉所有空白：模型常写「90秒」，而关键词标的是「90 秒」。"""
    return re.sub(r"\s+", "", s or "")


def load_dataset() -> list[dict]:
    # utf-8-sig：PowerShell 5.1 写出的 utf8 文件带 BOM，用 utf-8 读会 JSONDecodeError
    text = DATASET.read_text(encoding="utf-8-sig")
    rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    assert len(rows) == 20, f"评测集应为 20 条，实际 {len(rows)}"
    for r in rows:
        assert "id" in r and "q" in r, f"缺 id/q：{r}"
        assert r.get("gold_source") or r.get("expect_refusal"), (
            f"负样本须显式标 expect_refusal：{r.get('id')}"
        )
    return rows


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def pick_docs(question: str, cands: list[dict], use_rerank: bool):
    """返回 (进 prompt 的 docs, 阈值过滤前的 top1 精排分数 | None)。"""
    if not use_rerank:
        return cands[:RERANK_K], None
    raw = rerank(question, cands, top_k=len(cands), min_score=0.0)
    top1 = float(raw[0]["rerank_score"]) if raw else None
    docs = [d for d in raw if d["rerank_score"] >= MIN_RERANK][:RERANK_K]
    return docs, top1


async def run_one(row: dict, cfg: dict) -> dict:
    q = row["q"]
    t0 = time.perf_counter()
    cands = retrieve(q, strategy=cfg["strategy"], top_k=10)
    docs, top1 = pick_docs(q, cands, cfg["use_rerank"])
    out = await generate_node({"question": q, "docs": docs})
    ms = (time.perf_counter() - t0) * 1000
    answer = out.get("answer") or ""
    usage = out.get("usage") or {}
    sources = [d["source"] for d in docs]
    gold = row.get("gold_source")
    kws = row.get("gold_kw") or []
    na = norm(answer)
    return {
        "id": row["id"],
        "q": q,
        "gold_source": gold,
        "n_docs": len(docs),
        "top1_source": sources[0] if sources else None,
        "hit4": bool(gold) and gold in sources,
        "hit1": bool(gold) and bool(sources) and sources[0] == gold,
        "kw_any": bool(kws) and any(norm(k) in na for k in kws),
        "kw_all": bool(kws) and all(norm(k) in na for k in kws),
        "refused": "没有相关信息" in answer,
        "ms": round(ms, 1),
        "tokens": int(usage.get("total_tokens") or 0),
        "top1_rerank": None if top1 is None else round(top1, 4),
        "answer": answer,
    }


def rate(xs: list[dict], key: str) -> float:
    return sum(bool(x[key]) for x in xs) / max(len(xs), 1)


def summarize(name: str, rows: list[dict]) -> dict:
    pos = [r for r in rows if r["gold_source"]]
    neg = [r for r in rows if not r["gold_source"]]
    return {
        "config": name,
        "n": len(rows),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "hit4": rate(pos, "hit4"),
        "hit1": rate(pos, "hit1"),
        "kw_any": rate(pos, "kw_any"),
        "kw_all": rate(pos, "kw_all"),
        "refuse": rate(neg, "refused"),
        "p95_ms": p95([r["ms"] for r in rows]),
        "avg_ms": sum(r["ms"] for r in rows) / max(len(rows), 1),
        "avg_tokens": sum(r["tokens"] for r in rows) / max(len(rows), 1),
    }


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="both", choices=list(GROUPS) + list(CONFIGS))
    args = ap.parse_args()
    names = GROUPS[args.config] if args.config in GROUPS else [args.config]

    rows_in = load_dataset()
    results: dict[str, list[dict]] = {}

    for name in names:
        cfg = CONFIGS[name]
        # 预热：模型加载与首次建连不计入耗时，否则第一条必然就是 P95
        cands = retrieve(rows_in[0]["q"], strategy=cfg["strategy"], top_k=10)
        if cfg["use_rerank"]:
            rerank(rows_in[0]["q"], cands, top_k=len(cands), min_score=0.0)
        await generate_node({"question": rows_in[0]["q"], "docs": cands[:1]})

        got = []
        for i, row in enumerate(rows_in, 1):
            r = await run_one(row, cfg)
            got.append(r)
            flag = "ok  " if (r["hit4"] or r["refused"]) else "MISS"
            print(
                f"[{name}] {i:>2}/20 {r['id']:<4} {flag} docs={r['n_docs']} "
                f"tok={r['tokens']:>5} {r['ms']:>7.0f}ms  {row['q'][:26]}"
            )
        results[name] = got

    print("\n" + "=" * 78)
    print("逐条明细")
    print("| config | id | 问题 | 块数 | 命中@4 | 关键词 | 拒答 | 耗时ms | token |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, got in results.items():
        for r in got:
            print(
                f"| {name} | {r['id']} | {r['q']} | {r['n_docs']} | "
                f"{'✅' if r['hit4'] else '—'} | {'✅' if r['kw_any'] else '—'} | "
                f"{'✅' if r['refused'] else '—'} | {r['ms']:.0f} | {r['tokens']} |"
            )

    print("\n" + "=" * 78)
    print("汇总")
    head = f"{'config':<18}{'命中@4':>8}{'命中@1':>8}{'关键词':>8}{'拒答率':>8}{'P95ms':>9}{'均token':>9}"
    print(head)
    for name, got in results.items():
        s = summarize(name, got)
        print(
            f"{name:<18}{pct(s['hit4']):>8}{pct(s['hit1']):>8}{pct(s['kw_any']):>8}"
            f"{pct(s['refuse']):>8}{s['p95_ms']:>9.0f}{s['avg_tokens']:>9.0f}"
        )

    rr = results.get("recursive_rerank")
    if rr:
        neg = [
            r["top1_rerank"]
            for r in rr
            if not r["gold_source"] and r["top1_rerank"] is not None
        ]
        pos = [
            r["top1_rerank"]
            for r in rr
            if r["gold_source"] and r["top1_rerank"] is not None
        ]
        print(
            f"\n阈值复核（MIN_RERANK = {MIN_RERANK}，11-17 只用 1 条负样本校准，今天用评测集复核）"
        )
        print(f"  负样本 top1 精排分（{len(neg)} 条）：{neg}")
        print(
            f"  正样本 top1 精排分最低 = {min(pos):.4f}（{len(pos)} 条）"
            if pos
            else "  无正样本分数"
        )
        if neg and pos and max(neg) < MIN_RERANK <= min(pos):
            print(
                f"  ✅ 阈值落在 [{max(neg):.4f}, {min(pos):.4f}] 之间：0.5 有实测支撑"
            )
        else:
            print("  ⚠️ 区间不干净：先别动 0.5，把边界那两条的原文贴出来一起看")

    if "naive" in results and "recursive_rerank" in results:
        a = summarize("naive", results["naive"])
        b = summarize("recursive_rerank", results["recursive_rerank"])
        print("\nREADME 行（复制这一行，数字全部来自本次实跑）")
        print(
            f"朴素切块命中率 {pct(a['hit4'])} → 递归切块 + 重排 {pct(b['hit4'])}"
            f"（20 条评测集，P95 延迟 {a['p95_ms'] / 1000:.1f}s → {b['p95_ms'] / 1000:.1f}s，"
            f"平均 token {a['avg_tokens']:.0f} → {b['avg_tokens']:.0f}）"
        )

    payload = {
        "configs": {name: summarize(name, got) for name, got in results.items()},
        "min_rerank": MIN_RERANK,
        "rows": results,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {OUT}（README 的数字从这里取，别手抄）")


if __name__ == "__main__":
    asyncio.run(main())
