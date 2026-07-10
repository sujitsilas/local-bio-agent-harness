"""Ollama fallback local provider (§4.2).

Ollama exposes an OpenAI-compatible API at /v1. It supports `format: json` but not
full json_schema strict mode, so we lean on the prompted-JSON + repair fallback in
structured.py rather than server-side constraint.
"""
from __future__ import annotations

from bioagent.llm._openai_compat import OpenAICompatProvider


class OllamaProvider(OpenAICompatProvider):
    server_side_schema = False  # no strict json_schema -> use prompted-JSON + repair
