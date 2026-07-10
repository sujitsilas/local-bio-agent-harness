"""Structured output: schema enforce + prompted-JSON fallback + one repair retry,
plus emulated tool calls (§4.3).

Small local models are unreliable at native tool calls and free-text parsing, so we
enforce structure in this priority order:
  1. server-side json_schema via response_format (preferred; MLX/LM Studio),
  2. prompted JSON + pydantic validation + one repair retry,
  3. emulated tool calls: typed JSON action validated against a ToolSpec.
Never regex free text.
"""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from bioagent.llm.provider import LLMProvider
from bioagent.models import Sampling, ToolCall, ToolSpec

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredError(RuntimeError):
    """Raised when even the repair retry fails to yield valid structured output."""


def with_system(messages: list[dict], extra: str) -> list[dict]:
    """Fold `extra` into the leading system message (or add one at the front).

    Strict chat templates (Qwen3, etc.) reject a system message that isn't first or a
    duplicate system turn — mlx_lm.server surfaces that as a 404. So we never append a
    trailing system turn; we merge into position 0 and keep role ordering valid.
    """
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = {**msgs[0], "content": msgs[0]["content"].rstrip() + "\n\n" + extra}
    else:
        msgs = [{"role": "system", "content": extra}, *msgs]
    return msgs


def extract_json(text: str) -> str:
    """Pull the first JSON object/array out of a possibly fenced/chatty reply."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    # find the first balanced {...} or [...] if the model added prose around it
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        return text
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def generate_structured(
    provider: LLMProvider,
    messages: list[dict],
    schema_model: type[T],
    *,
    model_profile: str = "primary",
    sampling: Sampling | None = None,
    max_repairs: int = 2,
    first_thinking: bool | None = None,
    stats: dict | None = None,
) -> T:
    """Return a validated instance of `schema_model`, self-repairing on failure.

    First attempt uses `first_thinking` (None = model/config default, so reasoning streams);
    repair attempts force thinking OFF, which guarantees a compact, complete JSON answer
    even if the reasoning-on attempt truncated — visible reasoning AND reliability.

    `stats` (if given) records {"calls", "repaired", "failed"} for the ablation table's
    structured-output failure rate.
    """
    if stats is not None:
        stats["calls"] = stats.get("calls", 0) + 1
    schema = schema_model.model_json_schema()
    # Always show the model the target schema on the FIRST attempt. Server-side-schema
    # backends ignore this harmlessly; prompted backends (Ollama, small MLX) need it or
    # they return under-specified/empty JSON (§4.3). Merge into the leading system turn
    # so we don't break strict chat templates (see with_system).
    convo = with_system(
        messages,
        "Respond with ONLY a JSON object that validates against this JSON Schema "
        "(no prose, no markdown fences):\n" + json.dumps(schema),
    )

    result = provider.chat(
        convo, model_profile=model_profile, sampling=sampling, response_schema=schema,
        thinking=first_thinking,
    )
    last_err: Exception | None = None

    for attempt in range(max_repairs + 1):
        raw = result.content
        try:
            parsed = schema_model.model_validate_json(extract_json(raw))
            if stats is not None and attempt > 0:
                stats["repaired"] = stats.get("repaired", 0) + 1
            return parsed
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = e
            if attempt == max_repairs:
                break
            # one repair prompt: show the model its output + the exact error (§4.3)
            convo = convo + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply did not match the required JSON schema.\n"
                        f"Validation error:\n{e}\n\n"
                        f"Return ONLY valid JSON matching this schema:\n"
                        f"{json.dumps(schema)}"
                    ),
                },
            ]
            # repair with thinking OFF -> compact, complete JSON (no truncation)
            result = provider.chat(
                convo, model_profile=model_profile, sampling=sampling,
                response_schema=schema, thinking=False,
            )

    if stats is not None:
        stats["failed"] = stats.get("failed", 0) + 1
    raise StructuredError(f"structured output failed after repair: {last_err}")


def emulated_tool_call(
    provider: LLMProvider,
    messages: list[dict],
    tools: list[ToolSpec],
    *,
    model_profile: str = "primary",
    sampling: Sampling | None = None,
    max_repairs: int = 1,
) -> ToolCall:
    """Ask the model to pick a tool + args as typed JSON, validated against ToolSpec.

    Do NOT assume OpenAI-style native tool schemas work on the 9B tier (§4.3).
    """
    by_name = {t.name: t for t in tools}
    catalog = "\n".join(
        f"- {t.name}: {t.description}\n  args schema: {json.dumps(t.input_schema)}" for t in tools
    )
    action_schema = {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": list(by_name)},
            "args": {"type": "object"},
        },
        "required": ["tool", "args"],
    }
    convo = with_system(
        messages,
        "Choose exactly one tool to call. Reply with ONLY a JSON object "
        '{"tool": <name>, "args": {...}}.\n\nAvailable tools:\n' + catalog,
    )

    last_err: Exception | None = None
    for attempt in range(max_repairs + 1):
        result = provider.chat(
            convo, model_profile=model_profile, sampling=sampling, response_schema=action_schema
        )
        raw = result.content
        try:
            action = json.loads(extract_json(raw))
            name = action["tool"]
            if name not in by_name:
                raise ValueError(f"unknown tool {name!r}; valid: {list(by_name)}")
            args = action.get("args", {})
            if not isinstance(args, dict):
                raise ValueError("`args` must be a JSON object")
            return ToolCall(id=f"call_{attempt}", name=name, args=args)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            last_err = e
            if attempt == max_repairs:
                break
            convo = convo + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Invalid action: {e}. Reply with only the JSON object."},
            ]

    raise StructuredError(f"emulated tool call failed after repair: {last_err}")
