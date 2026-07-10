"""Shared OpenAI-compatible HTTP adapter.

MLX (LM Studio / mlx_lm.server), Ollama, and cloud all speak this dialect, so they
subclass this and only override the config block (§4.2: "adding a backend = one file").
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

import httpx

from bioagent.config import Config, ModelSpec
from bioagent.models import ChatResult, Sampling, ToolCall, ToolSpec, Usage


class OpenAICompatProvider:
    """One shared HTTP client + config block for every OpenAI-compatible backend."""

    #: whether this backend can constrain output server-side via response_format
    server_side_schema = True

    def __init__(self, config: Config, timeout_s: float = 600.0):
        self.config = config
        self._client = httpx.Client(timeout=timeout_s)

    # -- config resolution --------------------------------------------------- #
    def _spec(self, profile: str) -> ModelSpec:
        return self.config.model_for(profile)

    def _headers(self, spec: ModelSpec) -> dict[str, str]:
        return {"Authorization": f"Bearer {spec.api_key}", "Content-Type": "application/json"}

    def _extra(self, spec: ModelSpec, thinking: bool | None) -> dict[str, Any]:
        """spec.extra_body, with enable_thinking overridden per-call when requested.

        Used to force thinking OFF on structured-output repair retries so we always
        recover clean JSON even though the first attempt streams reasoning.
        """
        extra = {k: (dict(v) if isinstance(v, dict) else v) for k, v in spec.extra_body.items()}
        if thinking is not None:
            ctk = dict(extra.get("chat_template_kwargs", {}))
            ctk["enable_thinking"] = thinking
            extra["chat_template_kwargs"] = ctk
        return extra

    # -- chat ---------------------------------------------------------------- #
    def chat(
        self,
        messages: list[dict],
        *,
        model_profile: str = "primary",
        sampling: Sampling | None = None,
        response_schema: dict | None = None,
        tools: list[ToolSpec] | None = None,
        thinking: bool | None = None,
    ) -> ChatResult:
        spec = self._spec(model_profile)
        sampling = sampling or Sampling()
        body: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": spec.max_tokens,
        }
        body.update(self._extra(spec, thinking))  # extras + per-call thinking override
        if response_schema is not None and spec.supports_response_format and self.server_side_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema, "strict": True},
            }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]

        resp = self._client.post(
            f"{spec.endpoint}/chat/completions", headers=self._headers(spec), json=body
        )
        resp.raise_for_status()
        data = resp.json()
        return self._to_result(data)

    def _to_result(self, data: dict) -> ChatResult:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            tool_calls.append(
                ToolCall(id=tc.get("id") or str(uuid.uuid4()), name=fn.get("name", ""), args=args)
            )
        u = data.get("usage", {}) or {}
        return ChatResult(
            content=msg.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model", ""),
            reasoning=msg.get("reasoning") or msg.get("reasoning_content") or "",
            usage=Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            ),
            raw=data,
        )

    # -- streaming chat ------------------------------------------------------ #
    def chat_stream(
        self,
        messages: list[dict],
        *,
        model_profile: str = "primary",
        sampling: Sampling | None = None,
        response_schema: dict | None = None,
        thinking: bool | None = None,
    ) -> Iterator[dict]:
        """Yield token deltas as {"channel": "reasoning"|"content", "text": ...} and a
        final {"channel": "done", "content", "reasoning", "model", "usage"} event.

        Reasoning models (Qwen3) stream `delta.reasoning` first, then `delta.content`.
        """
        spec = self._spec(model_profile)
        sampling = sampling or Sampling()
        body: dict[str, Any] = {
            "model": spec.model, "messages": messages, "stream": True,
            "temperature": sampling.temperature, "top_p": sampling.top_p,
            "max_tokens": spec.max_tokens,
        }
        body.update(self._extra(spec, thinking))
        if response_schema is not None and spec.supports_response_format and self.server_side_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "response", "schema": response_schema,
                                                       "strict": True}}

        content, reasoning, model, usage = [], [], spec.model, {}
        with self._client.stream("POST", f"{spec.endpoint}/chat/completions",
                                 headers=self._headers(spec), json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or line.startswith(":"):  # keepalive / blank
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                model = chunk.get("model", model)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if delta.get("reasoning"):
                    reasoning.append(delta["reasoning"])
                    yield {"channel": "reasoning", "text": delta["reasoning"]}
                if delta.get("content"):
                    content.append(delta["content"])
                    yield {"channel": "content", "text": delta["content"]}
        yield {"channel": "done", "content": "".join(content), "reasoning": "".join(reasoning),
               "model": model, "usage": usage}

    # -- embeddings ---------------------------------------------------------- #
    def embed(self, texts: list[str], *, model_profile: str = "embeddings") -> list[list[float]]:
        spec = self._spec(model_profile)
        resp = self._client.post(
            f"{spec.endpoint}/embeddings",
            headers=self._headers(spec),
            json={"model": spec.model, "input": texts},
        )
        resp.raise_for_status()
        return [row["embedding"] for row in resp.json()["data"]]

    def close(self) -> None:
        self._client.close()
