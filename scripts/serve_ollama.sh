#!/usr/bin/env bash
# Serve the local coding model via Ollama for opencode.
#
# Ollama (unlike mlx_lm.server) enforces a fixed context, so the KV cache can't grow
# unbounded and OOM-crash Metal during a long coding session. KEEP_ALIVE keeps the model
# resident between requests; HOST=0.0.0.0 lets a second machine connect over the LAN.
set -euo pipefail

MODEL="${OC_MODEL:-qwen3.6-coding}"

# one-time: pull the Metal-optimized mxfp8 weights + build the tuned model
if ! ollama list | grep -q "^${MODEL}"; then
  echo "building ${MODEL} from qwen3.6:35b-a3b-mxfp8…"
  ollama pull qwen3.6:35b-a3b-mxfp8
  ollama create "${MODEL}" -f "$(dirname "$0")/../Modelfile"
fi

echo "serving Ollama (model ${MODEL} stays warm; Ctrl-C to stop)…"
OLLAMA_HOST=0.0.0.0 OLLAMA_KEEP_ALIVE=24h exec ollama serve
