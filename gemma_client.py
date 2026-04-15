"""
gemma_client.py
---------------
Async client for a locally-running Gemma 4 model served via an
OpenAI-compatible API (Ollama or vLLM).

Works with BOTH backends — just change GEMMA_BASE_URL / GEMMA_MODEL below.

  Ollama  →  GEMMA_BASE_URL = "http://localhost:11434/v1"
             GEMMA_MODEL    = "gemma3:4b"          (or gemma3:12b)

  vLLM    →  GEMMA_BASE_URL = "http://localhost:8000/v1"
             GEMMA_MODEL    = "google/gemma-4-E2B-it"
"""

import asyncio
from typing import Optional

import httpx

# ─── Configuration ────────────────────────────────────────────────────────────
GEMMA_BASE_URL: str = "http://localhost:11434/v1"   # change for vLLM → :8000
GEMMA_MODEL:    str = "gemma3:4b"                   # change for vLLM → HF model id
GEMMA_API_KEY:  str = "ollama"                      # Ollama ignores this; vLLM uses "EMPTY"

REQUEST_TIMEOUT: float = 120.0   # seconds per request
MAX_PARALLEL:    int   = 10      # httpx connection pool size
# ──────────────────────────────────────────────────────────────────────────────


async def generate_intro(
    topic: str,
    *,
    max_tokens: int = 300,
    temperature: float = 0.7,
    system_prompt: str = (
        "You are a concise academic writer. "
        "Write a short, engaging introduction paragraph about the given topic."
    ),
    client: Optional[httpx.AsyncClient] = None,
) -> str:
    """
    Ask the local Gemma model to write an introduction paragraph about `topic`.

    Parameters
    ----------
    topic        : The subject to introduce.
    max_tokens   : Maximum tokens in the response.
    temperature  : Sampling temperature (0 = deterministic, 1 = creative).
    system_prompt: Override the default system role instruction.
    client       : Reuse an existing httpx.AsyncClient for connection pooling.
                   If None, a temporary client is created.

    Returns
    -------
    str  The generated introduction text.
    """
    payload = {
        "model": GEMMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Write an introduction about: {topic}"},
        ],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }

    headers = {
        "Authorization": f"Bearer {GEMMA_API_KEY}",
        "Content-Type":  "application/json",
    }

    async def _post(c: httpx.AsyncClient) -> str:
        resp = await c.post(
            f"{GEMMA_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    if client is not None:
        return await _post(client)

    async with httpx.AsyncClient() as c:
        return await _post(c)


async def generate_intros_parallel(topics: list[str], **kwargs) -> list[str]:
    """
    Call generate_intro concurrently for every topic in `topics`.
    Uses a single shared httpx client for connection pooling.

    Returns a list of intro strings in the same order as `topics`.
    """
    limits = httpx.Limits(
        max_connections=MAX_PARALLEL,
        max_keepalive_connections=MAX_PARALLEL,
    )
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [generate_intro(t, client=client, **kwargs) for t in topics]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # convert exceptions to readable error strings so one failure doesn't crash all
    return [
        r if isinstance(r, str) else f"[ERROR] {type(r).__name__}: {r}"
        for r in results
    ]
