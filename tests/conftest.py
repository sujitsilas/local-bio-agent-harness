"""Shared test fixtures. A scripted fake LLM provider lets us exercise the structured
output / planner / critic logic WITHOUT a live model.
"""
from __future__ import annotations

from collections import deque

import pytest

from bioagent.models import ChatResult, Usage


class FakeProvider:
    """Returns queued responses in order; records the messages it was called with."""

    def __init__(self, responses: list[str]):
        self._responses = deque(responses)
        self.calls: list[dict] = []

    def chat(self, messages, *, model_profile="primary", sampling=None,
             response_schema=None, tools=None, thinking=None) -> ChatResult:
        self.calls.append({"messages": messages, "model_profile": model_profile,
                           "sampling": sampling, "response_schema": response_schema,
                           "thinking": thinking})
        content = self._responses.popleft() if self._responses else "{}"
        return ChatResult(content=content, usage=Usage())

    def embed(self, texts, *, model_profile="embeddings"):
        return [[0.0, 1.0, 0.0] for _ in texts]


@pytest.fixture
def fake_provider():
    return FakeProvider
