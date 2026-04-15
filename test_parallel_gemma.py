"""
test_parallel_gemma.py
----------------------
Fires 10 concurrent intro-generation requests directly to the local Gemma
server (bypassing Gemini) to verify parallel throughput.

Run:
    python test_parallel_gemma.py

Expected output: all 10 intros appear roughly at the same time (wall-clock
time ≈ single request time), demonstrating true concurrency.
"""

from __future__ import annotations

import asyncio
import time

from gemma_client import generate_intros_parallel

TOPICS = [
    "Artificial Intelligence",
    "Climate Change",
    "Black Holes",
    "The Roman Empire",
    "Machine Learning",
    "Renaissance Art",
    "Quantum Computing",
    "The Human Genome Project",
    "Cryptocurrency and Blockchain",
    "The James Webb Space Telescope",
]


async def main() -> None:
    print(f"Sending {len(TOPICS)} parallel requests to local Gemma…\n")
    t0 = time.perf_counter()

    results = await generate_intros_parallel(TOPICS, max_tokens=200, temperature=0.7)

    elapsed = time.perf_counter() - t0
    print(f"All {len(TOPICS)} responses received in {elapsed:.1f}s\n")
    print("=" * 70)

    for i, (topic, intro) in enumerate(zip(TOPICS, results), 1):
        print(f"\n[{i:02d}] Topic: {topic}")
        print(f"     {intro[:250]}{'…' if len(intro) > 250 else ''}")
        print("-" * 70)

    ok   = sum(1 for r in results if not r.startswith("[ERROR]"))
    err  = len(results) - ok
    print(f"\n✅ Successful: {ok}/{len(TOPICS)}   ❌ Errors: {err}")


if __name__ == "__main__":
    asyncio.run(main())
