"""RAG 索引:语料 -> 切片 -> embedding -> Qdrant 入库"""

import argparse
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

CORPUS = Path("docs/manuals")
DIM = 512
Q_PREFIX = "为这个句子生成表示以用于检索相关文章:"

STRATEGIES = {
    "naive": dict(chunk_size=500, chunk_overlap=0),
    "struct": dict(
        chunk_size=500, chunk_overlap=80, separators=["\n## ", "\n\n", "\n", "。"]
    ),
}


def get_client() -> QdrantClient:
    # return QdrantClient(url="http://localhost:6333")
    return QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))


class STEncoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.m.encode(texts, normalize_embeddings=True).tolist()


#  class FastEncoder:
#     def __init__(self):
#         from fastembed import TextEmbedding
#         self.m = TextEmbedding(model_name = "BAAI/bge-small-zh-v1.5")

#     def encode(self, texts: list[str]) -> list[list[float]]:
#          return [v.tolist() for v in self.m.embed(texts)]


def load_chunks(strategy: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(**STRATEGIES[strategy])
    files = sorted(CORPUS.glob("*.md"))
    assert len(files) == 20, f"语料应为20份,实际{len(files)}"
    items = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        title = lines[0].lstrip("# ").strip() if lines else f.stem
        for i, chunk in enumerate(splitter.split_text(text)):
            items.append(
                {
                    "text": chunk,
                    "payload": {
                        "source": f.name,
                        "title": title,
                        "chunk_index": i,
                        "strategy": strategy,
                    },
                }
            )
    return items


def build(client: QdrantClient, enc, strategy: str) -> tuple[str, int]:
    name = f"manuals_{strategy}"
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )
    items = load_chunks(strategy)
    vecs = enc.encode([it["text"] for it in items])
    client.upsert(
        name,
        points=[
            PointStruct(id=i, vector=v, payload={**it["payload"], "text": it["text"]})
            for i, (it, v) in enumerate(zip(items, vecs))
        ],
    )
    print(
        f"{name}:{len(items)} chunks ipserted,points_count = {client.count(name).count}"
    )
    return name, len(items)


def probe(client: QdrantClient, enc, name: str, q: str, limit: int = 3) -> None:
    vec = enc.encode([Q_PREFIX + q])[0]
    hits = client.query_points(name, query=vec, with_payload=True, limit=limit).points
    print(f"\n[{name}] {q}")
    for h in hits:
        p = h.payload or {}
        text = str(p.get("text", ""))[:40]
        print(
            f"  {h.score:.4f}  {p.get('source', '?')}#{p.get('chunk_index', '?')}  {text!r}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["st"], default="st")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    client = get_client()
    enc = STEncoder()
    names = [build(client, enc, s) for s in STRATEGIES]
    if args.probe:
        for name, _ in names:
            for q in [
                "设备离线了怎么处理?",
                "上报间隔是多少?",
                "这个设备的出厂价是多少?",
            ]:
                probe(client, enc, name, q)
