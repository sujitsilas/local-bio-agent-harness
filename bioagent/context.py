"""AgentContext — the collaborators the cell-type caller needs, wired from config."""
from __future__ import annotations

from dataclasses import dataclass, field

from bioagent.bio.priors_kb import PriorsKB
from bioagent.config import Config, load_config
from bioagent.exec.engine import ExecEngine
from bioagent.llm.factory import build_provider
from bioagent.llm.provider import LLMProvider
from bioagent.tracing import NullTracer, Tracer


@dataclass
class AgentContext:
    config: Config
    provider: LLMProvider
    engine: ExecEngine
    kb: PriorsKB
    tracer: Tracer = field(default_factory=NullTracer)

    @classmethod
    def build(cls, config: Config | None = None) -> "AgentContext":
        config = config or load_config()
        kb = PriorsKB(config.path("state_db"))
        kb.seed(config.path("seeds_dir"))
        return cls(config=config, provider=build_provider(config),
                   engine=ExecEngine(config), kb=kb)
