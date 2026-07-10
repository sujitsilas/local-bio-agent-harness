"""MLX provider — the primary path (§4.2).

Talks to any OpenAI-compatible MLX endpoint: LM Studio server, `mlx_lm.server`
(+ Outlines), or mlx-vlm server. All the wire logic lives in OpenAICompatProvider;
this class only exists to name the backend and let the factory pick it.
"""
from __future__ import annotations

from bioagent.llm._openai_compat import OpenAICompatProvider


class MLXOpenAIProvider(OpenAICompatProvider):
    """LM Studio / mlx_lm.server expose json_schema response_format (§14)."""

    server_side_schema = True
