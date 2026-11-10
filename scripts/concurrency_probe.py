import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import llm

PROMPT = "只回复两个字：收到"


async def main() -> None:
    t0 = time.perf_counter()
    for _ in range(3):
        await llm.ainvoke(PROMPT)
    serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    await asyncio.gather(*(llm.ainvoke(PROMPT) for _ in range(3)))
    concurrent = time.perf_counter() - t0

    print(f"串行 3次：{serial:.2f}s")
    print(f"并发 3次：{concurrent:.2f}s")
    print(f"加速比：{serial / concurrent:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
