"""Build the configured LLMProvider. The only place a backend is chosen by name."""
from __future__ import annotations

from bioagent.config import Config, load_config
from bioagent.llm.cloud import CloudProvider
from bioagent.llm.mlx_openai import MLXOpenAIProvider
from bioagent.llm.ollama import OllamaProvider
from bioagent.llm.provider import LLMProvider

_BACKENDS = {
    "mlx_openai": MLXOpenAIProvider,
    "ollama": OllamaProvider,
    "cloud": CloudProvider,
}


def build_provider(config: Config | None = None) -> LLMProvider:
    config = config or load_config()
    try:
        cls = _BACKENDS[config.provider]
    except KeyError:
        raise ValueError(
            f"unknown provider {config.provider!r}; choose from {list(_BACKENDS)}"
        ) from None
    return cls(config)
