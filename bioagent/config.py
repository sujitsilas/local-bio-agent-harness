"""Config loading. Merges default.yaml with the selected hardware profile.

A model swap (§1, §14) is a config edit here, never a code change.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"


class ModelSpec(BaseModel):
    profile: str
    model: str
    endpoint: str = "http://localhost:1234/v1"
    api_key: str = "not-needed"
    supports_response_format: bool = False
    max_tokens: int = 2048  # cap per response; reasoning models need room to think + answer
    # backend-specific request extras merged verbatim into the JSON body. e.g. for a
    # Qwen3 reasoning model served by mlx_lm.server:
    #   {"chat_template_kwargs": {"enable_thinking": false}}
    # Other backends that don't understand a key simply ignore it.
    extra_body: dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel):
    """Flattened, validated view of default.yaml + the chosen hardware tier."""

    hardware_profile: str
    provider: str = "mlx_openai"
    serving_backend: str = "lm_studio"
    allow_cloud_llm: bool = False
    concurrent_models: bool = False

    models: dict[str, ModelSpec] = Field(default_factory=dict)
    paths: dict[str, str] = Field(default_factory=dict)
    sampling: dict[str, dict[str, float]] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    install_policy: dict[str, Any] = Field(default_factory=dict)
    graph: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    anndata: dict[str, Any] = Field(default_factory=dict)

    def path(self, key: str) -> Path:
        """Resolve a configured path relative to the repo root."""
        return (REPO_ROOT / self.paths[key]).resolve()

    def model_for(self, profile: str) -> ModelSpec:
        """Pick weights for a profile, falling back to primary (§2, router)."""
        return self.models.get(profile) or self.models["primary"]


def _read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=None)
def load_config(default_path: str | None = None) -> Config:
    base = _read_yaml(Path(default_path) if default_path else CONFIGS / "default.yaml")
    hw = _read_yaml(CONFIGS / "hardware" / f"{base['hardware_profile']}.yaml")

    models = {name: ModelSpec(**spec) for name, spec in hw.get("models", {}).items()}
    return Config(
        hardware_profile=base["hardware_profile"],
        provider=base.get("provider", "mlx_openai"),
        serving_backend=base.get("serving_backend", "lm_studio"),
        allow_cloud_llm=base.get("allow_cloud_llm", False),
        concurrent_models=hw.get("concurrent_models", False),
        models=models,
        paths=base.get("paths", {}),
        sampling=base.get("sampling", {}),
        execution=base.get("execution", {}),
        install_policy=base.get("install_policy", {}),
        graph=base.get("graph", {}),
        review=base.get("review", {}),
        anndata=hw.get("anndata", {}),
    )
