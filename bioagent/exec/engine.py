"""ExecEngine — the §5.4 contract. Ties venv + kernel + handles + sandbox together.

One engine instance manages many Runs, each with its own venv + kernel. Only summaries
and small tables ever cross back to the LLM; AnnData stays in the kernel (§1.2).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from bioagent.config import Config
from bioagent.exec.handles import HandleRegistry
from bioagent.exec.kernel import RunKernel
from bioagent.exec.sandbox import startup_preamble
from bioagent.exec.venv import RunVenv
from bioagent.models import EnvInfo, ExecResult, HandleSummary


class _RunCtx:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.venv = RunVenv(run_dir)
        self.kernel: RunKernel | None = None
        self.handles: HandleRegistry | None = None
        self.figures_dir = run_dir / "figures"
        self.tables_dir = run_dir / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)


class ExecEngine:
    def __init__(self, config: Config):
        self.config = config
        self.runs_dir = config.path("runs_dir")
        self._runs: dict[str, _RunCtx] = {}

    def _ctx(self, run_id: str) -> _RunCtx:
        if run_id not in self._runs:
            self._runs[run_id] = _RunCtx(self.runs_dir / run_id)
        return self._runs[run_id]

    # -- §5.4 contract ------------------------------------------------------- #
    def provision(self, run_id: str, base_env: Path | None = None) -> EnvInfo:
        ctx = self._ctx(run_id)
        base_env = base_env or self.config.path("base_env")
        info = ctx.venv.provision(base_env)
        self._launch_kernel(ctx, info)
        return info

    def install(self, run_id: str, packages: list[str]) -> EnvInfo:
        ctx = self._ctx(run_id)
        info = ctx.venv.install(packages)  # host-side uv, re-freezes lockfile
        return info  # kernel picks up new packages on next import

    def run_code(self, run_id: str, code: str, *, timeout_s: int | None = None) -> ExecResult:
        ctx = self._ctx(run_id)
        if ctx.kernel is None or not ctx.kernel.is_alive():
            raise RuntimeError(f"run {run_id} has no live kernel; call provision() first")
        timeout_s = timeout_s or int(self.config.execution.get("cell_timeout_s", 300))
        before = set(ctx.handles.list_handles()) if ctx.handles else set()

        raw = ctx.kernel.execute(code, timeout_s=timeout_s)

        figures = self._save_images(ctx, raw.images)
        after = set(ctx.handles.list_handles()) if ctx.handles else set()
        produced = sorted(after - before)

        cap = int(self.config.execution.get("output_size_cap_kb", 256)) * 1024
        stdout = _truncate(raw.stdout, cap)
        stderr = _truncate(raw.stderr, cap)

        if raw.timed_out:
            return ExecResult(
                ok=False, stdout=stdout, stderr=stderr, error=f"timeout after {timeout_s}s",
                summary=f"KILLED: cell exceeded {timeout_s}s wall-clock limit.",
            )
        if raw.error:
            return ExecResult(
                ok=False, stdout=stdout, stderr=stderr, error=_truncate(raw.error, cap),
                figures=figures, produced_handles=produced,
                summary="Execution raised an error (see error field).",
            )
        return ExecResult(
            ok=True, stdout=stdout, stderr=stderr, figures=figures,
            produced_handles=produced,
            summary=_summarize(stdout, produced, figures, raw.results),
        )

    def get_handle_summary(self, run_id: str, name: str) -> HandleSummary:
        ctx = self._ctx(run_id)
        assert ctx.handles is not None, "kernel not started"
        return ctx.handles.summary(name)

    def snapshot(self, run_id: str) -> Path:
        """Write kernel AnnData handles to .h5ad + copy lockfile for resume (§5.4)."""
        ctx = self._ctx(run_id)
        snap = ctx.run_dir / "snapshots" / f"snap_{int(time.time())}"
        snap.mkdir(parents=True, exist_ok=True)
        if ctx.venv.lockfile.exists():
            shutil.copy(ctx.venv.lockfile, snap / "lockfile.txt")
        # persist every AnnData handle to disk so a rebuilt kernel can restore it
        code = (
            "import anndata as _ad, os as _os\n"
            f"_snap = {str(snap)!r}\n"
            "for _h in _json.loads(_bioagent_handles()):\n"
            "    _o = globals().get(_h)\n"
            "    if type(_o).__name__ == 'AnnData':\n"
            "        _o.write_h5ad(_os.path.join(_snap, _h + '.h5ad'))\n"
        )
        if ctx.kernel and ctx.kernel.is_alive():
            ctx.kernel.execute(code, timeout_s=300)
        return snap

    def resume(self, run_id: str, snapshot: Path) -> EnvInfo:
        """Rebuild identical env from the lockfile, then restore AnnData handles."""
        ctx = self._ctx(run_id)
        info = ctx.venv.rebuild_from_lockfile(snapshot / "lockfile.txt")
        self._launch_kernel(ctx, info)
        restore = (
            "import anndata as _ad, glob as _glob, os as _os\n"
            f"for _f in _glob.glob(_os.path.join({str(snapshot)!r}, '*.h5ad')):\n"
            "    globals()[_os.path.basename(_f)[:-5]] = _ad.read_h5ad(_f)\n"
        )
        ctx.kernel.execute(restore, timeout_s=300)
        return info

    def teardown(self, run_id: str, *, keep_venv: bool = False) -> None:
        ctx = self._runs.pop(run_id, None)
        if ctx is None:
            return
        if ctx.kernel:
            ctx.kernel.shutdown()
        if not keep_venv and ctx.venv.venv_path.exists():
            shutil.rmtree(ctx.venv.venv_path, ignore_errors=True)

    # -- internals ----------------------------------------------------------- #
    def _launch_kernel(self, ctx: _RunCtx, info: EnvInfo) -> None:
        ctx.kernel = RunKernel(ctx.run_dir, info.python)
        preamble = startup_preamble(
            memory_ceiling_gb=self.config.execution.get("memory_ceiling_gb"),
            block_network=not self.config.execution.get("analysis_cell_network", False),
        )
        ctx.kernel.start(startup_code=preamble)
        ctx.handles = HandleRegistry(ctx.kernel)

    def _save_images(self, ctx: _RunCtx, images: list[bytes]) -> list[str]:
        paths = []
        for img in images:
            p = ctx.figures_dir / f"fig_{int(time.time()*1000)}_{len(paths)}.png"
            p.write_bytes(img)
            paths.append(str(p))
        return paths


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n...[truncated {len(text) - cap} chars]"


def _summarize(stdout: str, produced: list[str], figures: list[str], results: list[str]) -> str:
    bits = []
    if produced:
        bits.append(f"handles: {', '.join(produced)}")
    if figures:
        bits.append(f"{len(figures)} figure(s)")
    tail = stdout.strip().splitlines()[-5:]
    if tail:
        bits.append("stdout tail: " + " | ".join(tail))
    if results:
        bits.append("result: " + results[-1][:200])
    return "; ".join(bits) or "ok (no output)"
