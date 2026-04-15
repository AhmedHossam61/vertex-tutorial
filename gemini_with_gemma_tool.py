"""
gemini_with_gemma_tool.py
--------------------------
Demonstrates Gemini function-calling where "generate_intro" is a tool.

Flow:
  1. User sends a prompt to Gemini (e.g. "give me an intro about quantum computing").
  2. Gemini recognises it matches the `generate_intro` tool and returns a
     function-call response with the extracted topic argument.
  3. We call the local Gemma 4 model with that topic.
  4. We feed the result back to Gemini as a function response.
  5. Gemini wraps it in a polished final answer, which we print.

Prerequisites:
  - GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in .env
  - Local Gemma server must be running (see gemma_server.py for instructions)
  - `pip install httpx` in the project venv
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from google import genai
from google.genai import types

# ── local imports ─────────────────────────────────────────────────────────────
from gemma_client import generate_intro

# ── environment ───────────────────────────────────────────────────────────────
def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env(Path(__file__).with_name(".env"))

# ── Gemini client ─────────────────────────────────────────────────────────────
gemini_client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)

GEMINI_MODEL = "gemini-2.5-flash"

# ── Tool definition ────────────────────────────────────────────────────────────
# This tells Gemini that it CAN call a function called generate_intro.
GENERATE_INTRO_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="generate_intro",
            description=(
                "Generate a short introduction paragraph about a topic "
                "using a local language model (Gemma 4). "
                "Call this whenever the user asks for an intro, overview, "
                "or introductory paragraph about any subject."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "topic": types.Schema(
                        type=types.Type.STRING,
                        description="The topic or subject to write an introduction about.",
                    ),
                    "max_tokens": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum number of tokens to generate (default 300).",
                    ),
                },
                required=["topic"],
            ),
        )
    ]
)

# ── Core agentic loop ──────────────────────────────────────────────────────────

async def chat_with_tool(user_message: str) -> str:
    """
    Send `user_message` to Gemini.
    If Gemini requests the generate_intro tool, execute it against Gemma,
    then return the final synthesised Gemini response.
    """
    print(f"\n[Gemini ←] {user_message}")

    # --- Turn 1: send user message, give Gemini the tool ---
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            tools=[GENERATE_INTRO_TOOL],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO,
                )
            ),
        ),
    )

    # --- Check if Gemini wants to call a function ---
    candidate = response.candidates[0]
    part = candidate.content.parts[0]

    if not part.function_call:
        # Gemini answered directly without a tool call
        return part.text

    fn = part.function_call
    print(f"[Gemini  →] Tool call: {fn.name}({dict(fn.args)})")

    # --- Dispatch to local Gemma server ---
    topic      = fn.args["topic"]
    max_tokens = int(fn.args.get("max_tokens", 300))

    print(f"[Gemma   ←] Generating intro for: '{topic}'  (max_tokens={max_tokens})")
    gemma_result = await generate_intro(topic, max_tokens=max_tokens)
    print(f"[Gemma   →] {gemma_result[:120]}…\n")

    # --- Turn 2: feed the tool result back to Gemini ---
    history = [
        types.Content(role="user",  parts=[types.Part(text=user_message)]),
        types.Content(role="model", parts=[types.Part(function_call=fn)]),
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="generate_intro",
                        response={"result": gemma_result},
                    )
                )
            ],
        ),
    ]

    final_response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=history,
        config=types.GenerateContentConfig(tools=[GENERATE_INTRO_TOOL]),
    )

    return final_response.text


# ── Demo ───────────────────────────────────────────────────────────────────────

DEMO_PROMPTS = [
    "Can you give me an intro about quantum computing?",
    "Write an introductory paragraph about the history of the internet.",
    "What's the weather like today?",   # should NOT trigger the tool
]


async def main() -> None:
    for prompt in DEMO_PROMPTS:
        result = await chat_with_tool(prompt)
        print(f"\n[Final answer]\n{result}")
        print("─" * 70)


if __name__ == "__main__":
    asyncio.run(main())
