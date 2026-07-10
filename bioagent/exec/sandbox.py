"""Sandbox controls for model-written code (§5.3, §13.1).

Enforced as an in-kernel startup preamble plus engine-side timeout (in kernel.py).
Analysis cells get NO network: dependencies arrive only via the `install_package`
tool (host-side uv) and facts only via knowledge-layer plugins (host-side httpx),
never from arbitrary model-written code. Kill + report on breach.

macOS note: RLIMIT_AS is unreliable on Darwin, so the memory ceiling is best-effort
here; the hard guarantee is the per-cell wall-clock timeout in RunKernel.execute.
"""
from __future__ import annotations

NETWORK_BLOCK = r"""
import socket as _socket
class _NoNetwork(_socket.socket):
    def connect(self, *a, **k):
        raise OSError("network disabled in analysis cells (§13.1); use install_package "
                      "for deps or a knowledge plugin for facts")
    def connect_ex(self, *a, **k):
        raise OSError("network disabled in analysis cells (§13.1)")
_socket.socket = _NoNetwork
"""

RESOURCE_LIMIT = r"""
try:
    import resource as _resource
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_AS)
    _ceiling = {ceiling_bytes}
    if _ceiling:
        try:
            _resource.setrlimit(_resource.RLIMIT_AS, (_ceiling, _hard))
        except (ValueError, OSError):
            pass  # Darwin often refuses RLIMIT_AS; wall-clock timeout is the real guard
except Exception:
    pass
"""

# Registers a helper the engine calls to build HandleSummary without moving data.
HANDLE_HELPERS = r"""
import json as _json
def _bioagent_summary(_name):
    _obj = globals().get(_name)
    if _obj is None:
        return _json.dumps({"name": _name, "kind": "missing"})
    _info = {"name": _name, "kind": type(_obj).__name__}
    _shape = getattr(_obj, "shape", None)
    if _shape is not None:
        _info["shape"] = list(_shape)
    if type(_obj).__name__ == "AnnData":
        _info["obs_keys"] = list(_obj.obs.columns)
        _info["var_keys"] = list(_obj.var.columns)
        _info["extra"] = {"obsm": list(_obj.obsm.keys()), "layers": list(_obj.layers.keys())}
    elif type(_obj).__name__ == "DataFrame":
        _info["obs_keys"] = [str(c) for c in _obj.columns][:50]
    return _json.dumps(_info)

def _bioagent_handles():
    _out = []
    for _k, _v in list(globals().items()):
        if _k.startswith("_"):
            continue
        if type(_v).__name__ in ("AnnData", "DataFrame"):
            _out.append(_k)
    return _json.dumps(_out)
"""


def startup_preamble(memory_ceiling_gb: float | None, block_network: bool = True) -> str:
    ceiling_bytes = int(memory_ceiling_gb * (1024**3)) if memory_ceiling_gb else 0
    parts = [RESOURCE_LIMIT.format(ceiling_bytes=ceiling_bytes), HANDLE_HELPERS]
    if block_network:
        parts.append(NETWORK_BLOCK)
    return "\n".join(parts)
