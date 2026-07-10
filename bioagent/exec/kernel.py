"""A Jupyter kernel bound to the Run's uv venv (§5.3).

One kernel per Run. State (loaded AnnData, intermediate objects) persists across tool
calls. The kernel imports from the Run's venv — not the host — because we launch
`<venv>/bin/python -m ipykernel_launcher` via a per-Run kernelspec.
"""
from __future__ import annotations

import base64
import json
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


@dataclass
class RawOutput:
    """Everything the kernel emitted for one execution, pre-summarisation."""

    stdout: str = ""
    stderr: str = ""
    results: list[str] = field(default_factory=list)  # text/plain of execute_result
    images: list[bytes] = field(default_factory=list)  # decoded PNG bytes (display_data)
    error: str | None = None
    timed_out: bool = False


def _write_kernelspec(run_dir: Path, python: Path) -> tuple[KernelSpecManager, str]:
    name = "bioagent-run"
    spec_dir = run_dir / "kernels" / name
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "bioagent-run",
                "language": "python",
            }
        )
    )
    ksm = KernelSpecManager()
    ksm.kernel_dirs.insert(0, str(run_dir / "kernels"))
    return ksm, name


class RunKernel:
    def __init__(self, run_dir: Path, python: Path):
        self.run_dir = Path(run_dir)
        self.python = Path(python)
        self._km: KernelManager | None = None
        self._kc = None

    def start(self, startup_code: str | None = None) -> None:
        ksm, name = _write_kernelspec(self.run_dir, self.python)
        self._km = KernelManager(kernel_name=name, kernel_spec_manager=ksm)
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=60)
        if startup_code:
            self.execute(startup_code, timeout_s=60)

    def is_alive(self) -> bool:
        return self._km is not None and self._km.is_alive()

    def execute(self, code: str, *, timeout_s: int = 300) -> RawOutput:
        assert self._kc is not None, "kernel not started"
        msg_id = self._kc.execute(code)
        out = RawOutput()
        deadline = time.monotonic() + timeout_s

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                out.timed_out = True
                self._km.interrupt_kernel()  # kill + report on breach (§5.3)
                break
            try:
                msg = self._kc.get_iopub_msg(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = msg["msg_type"]
            content = msg["content"]
            if mtype == "stream":
                if content["name"] == "stdout":
                    out.stdout += content["text"]
                else:
                    out.stderr += content["text"]
            elif mtype in ("execute_result", "display_data"):
                data = content.get("data", {})
                if "image/png" in data:
                    out.images.append(base64.b64decode(data["image/png"]))
                if "text/plain" in data and mtype == "execute_result":
                    out.results.append(data["text/plain"])
            elif mtype == "error":
                out.error = "\n".join(content.get("traceback", [])) or content.get("evalue", "")
            elif mtype == "status" and content.get("execution_state") == "idle":
                break
        return out

    def shutdown(self) -> None:
        if self._kc is not None:
            self._kc.stop_channels()
        if self._km is not None and self._km.is_alive():
            self._km.shutdown_kernel(now=True)
        self._km = self._kc = None
