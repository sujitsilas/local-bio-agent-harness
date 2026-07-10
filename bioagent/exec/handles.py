"""Handle registry + summaries (§5.3).

Tools reference in-kernel objects by name (`adata`, `adata_sub_3`, ...); the object
never leaves the kernel. We ask the kernel (via the helpers injected by sandbox.py) for
metadata only — shapes, obs/var keys — and return it as a HandleSummary. This is the
mechanism that keeps data out of the LLM context (§1.2).
"""
from __future__ import annotations

import json

from bioagent.exec.kernel import RunKernel
from bioagent.models import HandleSummary


class HandleRegistry:
    def __init__(self, kernel: RunKernel):
        self.kernel = kernel

    def list_handles(self) -> list[str]:
        out = self.kernel.execute("print(_bioagent_handles())", timeout_s=30)
        return _parse_json(out.stdout, default=[])

    def summary(self, name: str) -> HandleSummary:
        out = self.kernel.execute(f"print(_bioagent_summary({name!r}))", timeout_s=30)
        data = _parse_json(out.stdout, default={"name": name, "kind": "missing"})
        shape = data.get("shape")
        return HandleSummary(
            name=data.get("name", name),
            kind=data.get("kind", "missing"),
            shape=tuple(shape) if shape else None,
            obs_keys=data.get("obs_keys", []),
            var_keys=data.get("var_keys", []),
            extra=data.get("extra", {}),
        )


def _parse_json(stdout: str, default):
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith(("{", "[")):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return default
