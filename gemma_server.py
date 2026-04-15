"""
gemma_server.py
---------------
Helper that prints clear instructions on how to start a local Gemma 4 server
on Windows using EITHER Ollama (recommended) OR vLLM inside WSL2/Docker.

The client code (gemma_client.py) uses the OpenAI-compatible REST API that
both backends expose, so switching is just a one-line URL change.

Run this file to see setup instructions:
    python gemma_server.py
"""

import sys

OLLAMA_MODEL   = "gemma3:4b"          # closest Ollama tag for Gemma 3/4 2B-instruct
HF_MODEL_ID    = "google/gemma-4-E2B-it"
OLLAMA_URL     = "http://localhost:11434/v1"
VLLM_URL       = "http://localhost:8000/v1"

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║          Local Gemma 4 Server  –  Setup Instructions            ║
╚══════════════════════════════════════════════════════════════════╝

vLLM does NOT run natively on Windows. Choose one of:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTION A  ·  Ollama  (easiest, Windows-native)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Download & install Ollama:
      https://ollama.com/download

2. Pull the Gemma model (choose one):
      ollama pull gemma3:4b          ← ~3 GB, fastest on CPU/GPU
      ollama pull gemma3:12b         ← better quality, needs ~8 GB VRAM

3. Ollama starts automatically; API is at:
      http://localhost:11434/v1

In gemma_client.py, set:
      GEMMA_BASE_URL = "http://localhost:11434/v1"
      GEMMA_MODEL    = "gemma3:4b"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTION B  ·  vLLM inside WSL2  (full parallelism, needs CUDA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Enable WSL2 and install Ubuntu from the Microsoft Store.

2. Inside WSL2:
      pip install vllm
      python -m vllm.entrypoints.openai.api_server \
          --model google/gemma-4-E2B-it \
          --host 0.0.0.0 --port 8000 \
          --max-model-len 4096 \
          --max-num-seqs 32 \
          --trust-remote-code \
          --dtype bfloat16

3. From Windows the API is at:
      http://localhost:8000/v1

In gemma_client.py, set:
      GEMMA_BASE_URL = "http://localhost:8000/v1"
      GEMMA_MODEL    = "google/gemma-4-E2B-it"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTION C  ·  vLLM via Docker  (GPU pass-through)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  docker run --runtime nvidia --gpus all -p 8000:8000 \
      -e HF_TOKEN=<your-token> \
      vllm/vllm-openai:latest \
      --model google/gemma-4-E2B-it \
      --trust-remote-code

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After the server is running, test with:
    python gemini_with_gemma_tool.py
    python test_parallel_gemma.py
"""


if __name__ == "__main__":
    print(BANNER)
    sys.exit(0)
