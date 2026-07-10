"""Core pydantic data models (spec §9) plus the shared LLM/exec contracts (§4, §5).

Kept in one module so every layer imports from a single source of truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# LLM layer (§4)
# --------------------------------------------------------------------------- #


class Sampling(BaseModel):
    """Temperature/top_p — ORTHOGONAL to model choice (§4.1)."""

    temperature: float = 0.2
    top_p: float = 0.95


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCall(BaseModel):
    """One emulated tool call: model emits typed JSON, we validate it (§4.3)."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    """Advertised to the model; drives emulated-tool-call validation (§6.1)."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ChatResult(BaseModel):
    """Structured return from every adapter — never a raw dict (§4.1)."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = ""  # resolved model id the backend actually served (for tracing)
    reasoning: str = ""  # chain-of-thought, when the model exposes it (reasoning models)
    raw: dict | None = None  # backend-native, debugging only


# --------------------------------------------------------------------------- #
# Planner / orchestration (§8, §9)
# --------------------------------------------------------------------------- #


class PlanStep(BaseModel):
    id: str
    intent: str  # human-readable goal
    tool: str  # "code_exec" | "scanpy_ops" | "pubmed" | ...
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """Planner emits this via JSON schema; we validate — never regex free text."""

    steps: list[PlanStep] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Insights / claims (§9)
# --------------------------------------------------------------------------- #


class Claim(BaseModel):
    text: str
    support: Literal["data", "literature", "hypothesis"]
    citations: list[str] = Field(default_factory=list)


class InsightCard(BaseModel):
    population: str  # e.g. "Macrophage / subcluster 3"
    annotation: str
    top_markers: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)  # data-supported findings
    implications: list[Claim] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Execution engine (§5)
# --------------------------------------------------------------------------- #


class HandleSummary(BaseModel):
    """Metadata about an in-kernel object. Only this reaches the LLM (§1.2)."""

    name: str
    kind: str  # "AnnData" | "DataFrame" | ...
    shape: tuple[int, ...] | None = None
    obs_keys: list[str] = Field(default_factory=list)
    var_keys: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class ExecResult(BaseModel):
    """Return of every code execution. Only `summary` + small tables go to the LLM."""

    ok: bool = True
    stdout: str = ""
    stderr: str = ""
    produced_handles: list[str] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)  # file paths
    tables: list[str] = Field(default_factory=list)  # file paths (small csv)
    summary: str = ""
    error: str | None = None


class EnvInfo(BaseModel):
    venv_path: Path
    python: Path
    installed: list[str] = Field(default_factory=list)  # from uv pip freeze
    lockfile: Path | None = None  # captured into provenance


# --------------------------------------------------------------------------- #
# Bio priors (§7)
# --------------------------------------------------------------------------- #


class SignatureSet(BaseModel):
    id: str
    name: str
    tissue_context: str
    source_ref: str = ""
    genes: list[str] = Field(default_factory=list)
    direction: Literal["up", "down", "both"] = "up"
    notes: str = ""


class CompositionPrior(BaseModel):
    id: str
    tissue_type: str
    expected_compartments: list[str] = Field(default_factory=list)
    source_ref: str = ""


SystemType = Literal["immune_homogeneous", "heterogeneous_tissue", "unknown"]


class Classification(BaseModel):
    system_type: SystemType
    tissue_type: str = "unknown"
    rationale: str = ""


class DatasetProfile(BaseModel):
    """What the agent learned by inspecting the object's existing metadata (§ user req:
    'load the object and check for any metadata that is already available first').

    Column choices are made by the LLM from a summary of obs columns — never raw data.
    """

    species: str = "unknown"  # "mouse" | "human" | ...
    # Existing cell-type annotation kept as a REFERENCE to benchmark against — the agent
    # still generates its OWN annotations de novo so the user can evaluate its calling.
    reference_annotation_col: str | None = None
    condition_col: str | None = None  # experimental condition (e.g. Burn/Sham)
    cluster_col: str | None = None  # existing clustering to reuse, if any
    is_normalized: bool = False  # X already log-normalized
    has_embedding: bool = False  # PCA/UMAP/harmony already present
    embedding_key: str | None = None  # obsm key to cluster on (e.g. X_pca_harmony)
    counts_layer: str | None = None  # raw counts layer name, if preserved
    needs_preprocessing: bool = True  # any of QC/normalize/HVG/PCA/neighbors missing
    notes: str = ""


class CriticVerdict(BaseModel):
    ok: bool
    flags: list[str] = Field(default_factory=list)
    rationale: str = ""


# --------------------------------------------------------------------------- #
# Run lifecycle (§10)
# --------------------------------------------------------------------------- #

RunStatus = Literal["queued", "provisioning", "running", "paused", "done", "failed"]


class Run(BaseModel):
    id: str
    dataset_ref: str
    question: str
    status: RunStatus = "queued"
    checkpoint_ref: str | None = None
    artifact_dir: str | None = None
    created_at: str = ""
    updated_at: str = ""


class ReviewItem(BaseModel):
    id: str
    run_id: str
    kind: Literal["contradiction", "surprising", "low_confidence", "ungrounded"]
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False
