#!/usr/bin/env bash
# Start the local MLX model server for bioagent (Apple silicon).
# Auto-downloads the model from HuggingFace on first run (~20 GB, cached after).
set -euo pipefail

MODEL="${BIOAGENT_MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
PORT="${BIOAGENT_PORT:-8080}"
PY="$(dirname "$0")/../.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "No project venv at $PY — run: uv venv && uv pip install -e '.[dev]'" >&2
  exit 1
fi

echo "Serving $MODEL on port $PORT (Ctrl-C to stop)..."
exec "$PY" -m mlx_lm.server --model "$MODEL" --port "$PORT"
