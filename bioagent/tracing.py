"""Minimal event tracer. The benchmark uses NullTracer (no-op); the real Tracer writes
JSONL if a caller wants a record of a calling run's steps (e.g. the NCBI searches)."""
from __future__ import annotations

import contextvars
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_current_stage: contextvars.ContextVar[str] = contextvars.ContextVar("stage", default="")


class Tracer:
    def __init__(self, run_id: str, trace_path: Path):
        self.run_id = run_id
        self.path = Path(trace_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, kind: str, **fields) -> None:
        with self._lock:
            self._seq += 1
            evt = {"seq": self._seq, "ts": time.time(), "kind": kind,
                   "stage": _current_stage.get(), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(evt, default=str) + "\n")

    @contextmanager
    def stage(self, name: str, detail: str = ""):
        token = _current_stage.set(name)
        self.emit("node_enter", node=name, detail=detail)
        try:
            yield
        finally:
            _current_stage.reset(token)


class NullTracer(Tracer):
    def __init__(self):
        pass

    def emit(self, kind: str, **fields) -> None:
        pass

    @contextmanager
    def stage(self, name: str, detail: str = ""):
        yield
