"""M1 acceptance: structured output enforces schema, self-repairs within one retry, and
model_profile / sampling are independently controllable (§4.1, §12 M1)."""
from __future__ import annotations

import pytest

from bioagent.llm.structured import (
    StructuredError,
    emulated_tool_call,
    extract_json,
    generate_structured,
    with_system,
)
from bioagent.models import Plan, Sampling, ToolSpec


def test_with_system_never_produces_trailing_system_message():
    # strict chat templates (Qwen3) 404 on a system turn that isn't first / is duplicated
    merged = with_system([{"role": "system", "content": "base"}, {"role": "user", "content": "q"}],
                         "schema instruction")
    assert [m["role"] for m in merged] == ["system", "user"]  # still exactly one system, first
    assert "schema instruction" in merged[0]["content"]
    # no leading system -> one is prepended, not appended
    prepended = with_system([{"role": "user", "content": "q"}], "extra")
    assert [m["role"] for m in prepended] == ["system", "user"]


def test_generate_structured_keeps_valid_role_ordering(fake_provider):
    provider = fake_provider(['{"steps": []}'])
    generate_structured(provider, [{"role": "system", "content": "s"},
                                   {"role": "user", "content": "u"}], Plan)
    roles = [m["role"] for m in provider.calls[0]["messages"]]
    assert roles == ["system", "user"]  # schema folded into system, not a 3rd turn


def test_extract_json_from_fenced_and_chatty():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('sure! {"a": {"b": 2}} done') == '{"a": {"b": 2}}'
    assert extract_json('[1, 2, 3]') == '[1, 2, 3]'


def test_generate_structured_happy_path(fake_provider):
    provider = fake_provider(['{"steps": [{"id": "s1", "intent": "load", "tool": "code_exec", "args": {}}]}'])
    plan = generate_structured(provider, [{"role": "user", "content": "x"}], Plan)
    assert isinstance(plan, Plan) and plan.steps[0].tool == "code_exec"


def test_generate_structured_self_repairs_once(fake_provider):
    # first reply is malformed; the repair reply is valid -> succeeds within one retry
    provider = fake_provider(["not json at all", '{"steps": []}'])
    plan = generate_structured(provider, [{"role": "user", "content": "x"}], Plan, max_repairs=1)
    assert plan.steps == []
    assert len(provider.calls) == 2  # original + one repair


def test_generate_structured_raises_after_repair_budget(fake_provider):
    provider = fake_provider(["garbage", "still garbage"])
    with pytest.raises(StructuredError):
        generate_structured(provider, [{"role": "user", "content": "x"}], Plan, max_repairs=1)


def test_sampling_and_profile_are_orthogonal(fake_provider):
    provider = fake_provider(['{"steps": []}'])
    generate_structured(provider, [{"role": "user", "content": "x"}], Plan,
                        model_profile="classifier", sampling=Sampling(temperature=0.0))
    call = provider.calls[0]
    assert call["model_profile"] == "classifier"
    assert call["sampling"].temperature == 0.0  # temp not conflated with model choice


def test_emulated_tool_call_validates_against_toolspec(fake_provider):
    tools = [ToolSpec(name="code_exec", description="run", input_schema={"type": "object"})]
    provider = fake_provider(['{"tool": "code_exec", "args": {"code": "print(1)"}}'])
    tc = emulated_tool_call(provider, [{"role": "user", "content": "run"}], tools)
    assert tc.name == "code_exec" and tc.args["code"] == "print(1)"


def test_emulated_tool_call_repairs_unknown_tool(fake_provider):
    tools = [ToolSpec(name="code_exec", description="run", input_schema={"type": "object"})]
    provider = fake_provider(['{"tool": "nope", "args": {}}', '{"tool": "code_exec", "args": {}}'])
    tc = emulated_tool_call(provider, [{"role": "user", "content": "run"}], tools, max_repairs=1)
    assert tc.name == "code_exec"
