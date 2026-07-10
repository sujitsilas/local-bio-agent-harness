"""LLMProvider Protocol (§4.1). Every LLM call in the harness goes through this.

Two rules adapters must enforce (§4.1):
  * `model_profile` (which weights) and `sampling` (temperature/top_p) are separate.
  * `chat` returns a structured `ChatResult`, never a raw dict.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from bioagent.models import ChatResult, Sampling, ToolSpec


@runtime_checkable
class LLMProvider(Protocol):
    def chat(
        self,
        messages: list[dict],
        *,
        model_profile: str = "primary",  # "primary" | "classifier" | "coder"
        sampling: Sampling | None = None,
        response_schema: dict | None = None,  # JSON schema -> structured output
        tools: list[ToolSpec] | None = None,
        thinking: bool | None = None,  # per-call override of reasoning (None = config default)
    ) -> ChatResult: ...
