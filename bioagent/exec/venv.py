"""Per-Run uv venv provisioning + on-demand install + lockfile capture (§5.2).

Each Run gets its own isolated environment. uv hardlinks from a shared global cache,
so per-Run venvs are fast and disk-cheap; repeated base installs cost near-zero after
the first. The lockfile is captured to provenance after provisioning and after every
on-demand install (§5.2 step 5), so a resume rebuilds the identical environment.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from bioagent.models import EnvInfo


class VenvError(RuntimeError):
    pass


def _uv(*args: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["uv", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise VenvError(f"uv {' '.join(args)} failed:\n{proc.stderr}")
    return proc.stdout


class RunVenv:
    """Owns one Run's virtual environment and its lockfile."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.venv_path = self.run_dir / "venv"
        self.lockfile = self.run_dir / "lockfile.txt"

    @property
    def python(self) -> Path:
        return self.venv_path / "bin" / "python"

    # -- provisioning -------------------------------------------------------- #
    def provision(self, base_env: Path) -> EnvInfo:
        """Create the venv and install the pinned base set (§5.2 steps 1–2, 5)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _uv("venv", str(self.venv_path))
        pkgs = _read_requirements(base_env)
        if pkgs:
            _uv("pip", "install", "--python", str(self.python), *pkgs)
        return self._freeze()

    def install(self, packages: list[str]) -> EnvInfo:
        """uv pip install into THIS Run's venv, then re-freeze the lockfile (§5.2 step 4)."""
        if packages:
            _uv("pip", "install", "--python", str(self.python), *packages)
        return self._freeze()

    def _freeze(self) -> EnvInfo:
        frozen = _uv("pip", "freeze", "--python", str(self.python))
        self.lockfile.write_text(frozen)
        installed = [line for line in frozen.splitlines() if line and not line.startswith("#")]
        return EnvInfo(
            venv_path=self.venv_path,
            python=self.python,
            installed=installed,
            lockfile=self.lockfile,
        )

    def rebuild_from_lockfile(self, lockfile: Path) -> EnvInfo:
        """Resume path (§5.2): rebuild the identical env from pinned versions."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _uv("venv", str(self.venv_path))
        _uv("pip", "install", "--python", str(self.python), "-r", str(lockfile))
        return self._freeze()


def _read_requirements(path: Path) -> list[str]:
    lines = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines
