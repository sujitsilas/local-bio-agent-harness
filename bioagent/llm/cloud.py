"""Cloud provider — kept ONLY to preserve the model-agnostic abstraction (§4.2).

DISABLED in the shipped config (local-first, §1.7). Instantiating it without
`allow_cloud_llm: true` raises — no user data or expression matrix ever leaves the
machine, and enabling cloud LLM inference is an explicit, deliberate opt-in.
"""
from __future__ import annotations

from bioagent.config import Config
from bioagent.llm._openai_compat import OpenAICompatProvider


class CloudProviderDisabledError(RuntimeError):
    pass


class CloudProvider(OpenAICompatProvider):
    server_side_schema = True

    def __init__(self, config: Config, timeout_s: float = 600.0):
        if not config.allow_cloud_llm:
            raise CloudProviderDisabledError(
                "Cloud LLM inference is disabled by default (§1.7). Set "
                "`allow_cloud_llm: true` in configs/default.yaml to opt in."
            )
        super().__init__(config, timeout_s)
