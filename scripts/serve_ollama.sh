#!/usr/bin/env bash
# Serve the local coding model via Ollama for opencode.
#
# Ollama (unlike mlx_lm.server) enforces a fixed context, so the KV cache can't grow
# unbounded and OOM-crash Metal during a long coding session. KEEP_ALIVE keeps the model
# resident between requests; HOST=0.0.0.0 lets a second machine connect over the LAN.
set -euo pipefail

MODEL="${OC_MODEL:-qwen3.6-coding}"

# one-time: pull the streaming GGUF q8_0 weights + build the tuned model
# (NOT mxfp8 — Ollama's MLX engine doesn't stream, which hangs opencode)
if ! ollama list | grep -q "^${MODEL}"; then
  echo "building ${MODEL} from qwen3.6:35b-a3b-q8_0…"
  ollama pull qwen3.6:35b-a3b-q8_0
  ollama create "${MODEL}" -f "$(dirname "$0")/../Modelfile"
fi

echo "serving Ollama (model ${MODEL} stays warm; Ctrl-C to stop)…"
# CONTEXT_LENGTH bounds the KV cache server-wide (matches the Modelfile's num_ctx), so a
# long session can't grow it toward the model's 262K max and OOM; KEEP_ALIVE keeps the
# model resident; HOST lets a second machine connect over the LAN.
OLLAMA_HOST=0.0.0.0 OLLAMA_KEEP_ALIVE=24h OLLAMA_CONTEXT_LENGTH=16384 exec ollama serve
