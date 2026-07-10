"""Config loads the 48 GB dev-tier profile; local-first defaults hold."""
from __future__ import annotations

from bioagent.config import load_config


def test_config_loads_dev_tier():
    cfg = load_config()
    assert cfg.hardware_profile == "48gb"
    assert cfg.provider in ("mlx_openai", "ollama")
    assert cfg.allow_cloud_llm is False  # local-first: cloud off by default
    assert "primary" in cfg.models
    assert cfg.model_for("coder").model == cfg.models["primary"].model  # falls back to primary
