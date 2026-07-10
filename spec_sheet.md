# Local Bioinformatics Agent Harness — Build Spec v1.0

**Audience:** Claude Code (implementation agent)
**Goal:** A model-agnostic, local-first agent that autonomously runs iterative scRNA-seq (and general omics) analysis loops, produces figures/plots/insights, and writes a durable report the user reviews later. Long-running and resumable; the user submits a dataset + question and comes back to a reviewable report with per-population insight cards and broader-implication reasoning.

> **How to use this doc:** Build in the milestone order in §12. Each milestone has acceptance criteria — do not advance until they pass. The architecture is fixed; model choices in §2 are defaults and must stay swappable via config.

---

## 1. Non-negotiable design principles

1. **Model agnosticism.** Every LLM call goes through one `LLMProvider` interface. The *primary/reasoning* model is swapped by hardware tier via config. No provider-specific code outside the provider adapters.
2. **Data never enters the LLM context.** AnnData objects live in a persistent execution kernel. Tools operate on **handles** (references) and return **summaries** (shapes, cluster counts, marker tables, figure paths). Only summaries reach the prompt. This is the single most important constraint — it shapes the tool interface, memory, and data flow.
3. **Code-generation first, plugins second.** The agent writes analysis code into a persistent kernel. Fixed plugins are reserved for (a) external APIs and (b) deterministic ops we don't want the model improvising. Rationale: real analysis is exploratory and parameter-heavy; a fixed plugin pipeline breaks on real data.
4. **Biology priors before clustering.** A structured prior/marker knowledge base drives system classification, annotation scoring, and a validation critic — not just a system prompt.
5. **Autonomous but reviewable.** Runs execute unattended, checkpoint continuously, and surface uncertain decisions to a review queue instead of blocking.
6. **Provenance always.** Every figure, table, and claim is traceable to the code that produced it, the data version it ran on, and the exact environment (uv lockfile) it ran in.
7. **Local-first, single knowledge exception.** All compute, model inference, execution, embeddings, vector store, and structured state are **local**. The **only** runtime egress to the network is the **knowledge layer** (literature/annotation APIs), against a whitelist. No user data and no expression matrices leave the machine. Cloud LLM providers exist in the abstraction but are **disabled by default** and require explicit opt-in. See the network policy in §13.1.
8. **Per-analysis isolated environment.** Each Run provisions its **own `uv` virtual environment** with a base package set before analysis begins, then installs additional packages **on demand** as the analysis requires them. The Run's kernel executes inside that venv, and the resolved lockfile is captured for reproducibility.

---

## 2. Hardware profiles & model roster

The **primary/reasoning model is the only agnostic (swappable) one** and doubles as the code-writer on constrained machines. Support models (small classifier + embeddings) are fixed and small enough to coexist on any 16GB+ machine. A dedicated coder is optional and only enabled on 48GB+ tiers.

Serve MLX models through an **OpenAI-compatible server** (LM Studio server, or `mlx_lm.server` + Outlines) so the same adapter code works for local and cloud. All numbers below are verified against current MLX community reports (July 2026) — see §14.

| Role | 16 GB tier | 24–36 GB tier | 48 GB+ tier (dev machine) | Swappable? |
|---|---|---|---|---|
| **Primary / reasoning / code** | Qwen3.5-9B MLX 4-bit (~6 GB) | Qwen3.6-35B-A3B MLX 4-bit (~22 GB) | Qwen3.6-27B MLX 4-bit (~18–20 GB) or 35B-A3B | **Yes (config)** |
| **Fast classifier** (routing, tissue class, cheap yes/no) | Qwen3.5-2B or 0.8B, or reuse primary | same | same | fixed |
| **Embeddings** (RAG over papers/results) | Qwen3-Embedding-0.6B or nomic-embed-text (~0.5 GB) | same | same | fixed |
| **Dedicated coder** (optional) | — (primary writes code) | — | Qwen3-Coder-Next only on 64 GB+; on 48 GB the primary writes code | optional |

**Memory reality on 16 GB:** primary (~6 GB) + embeddings (~0.5 GB) + Python/scanpy stack (~3 GB) + AnnData. Load AnnData in **backed mode** and **subsample for interactive steps** to stay in budget. Do **not** run concurrent multi-model routing on 16 GB — "routing" there means load/unload with swap latency. Concurrent multi-model routing is a 48 GB+ behavior, gated by profile.

**Dev machine (48 GB M5 Pro):** run one big primary (27B or 35B-A3B) as the unified reasoning+coder brain, plus embeddings, plus optional tiny classifier, plus data. A separate 46 GB coder does **not** coexist with a 20 GB primary on 48 GB — keep single-primary unless on 64 GB+.

---

## 3. Architecture

```
                         ┌───────────────────────┐
   submit run / review → │  FastAPI (API layer)  │ ← poll status / fetch report
                         └───────────┬───────────┘
                                     │  (durable Run in SQLite)
                         ┌───────────▼───────────┐
                         │   Background Worker    │
                         │  (executes one Run)    │
                         └───────────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  LangGraph Orchestrator │  ← SqliteSaver checkpointer
                         │  (stateful, cyclic)     │
                         └───────────┬───────────┘
        ┌───────────────┬───────────┼───────────────┬────────────────┐
 ┌──────▼──────┐ ┌──────▼──────┐ ┌─▼──────────┐ ┌───▼────────┐ ┌─────▼──────┐
 │ LLM Router  │ │ Exec Engine │ │ Bio Priors │ │  Memory    │ │  Reviewer  │
 │ +Provider   │ │ (persistent │ │  KB +      │ │ vector +   │ │  (critic + │
 │  adapters   │ │  kernel,    │ │ classifier │ │ SQLite +   │ │  review    │
 │             │ │  handles)   │ │ + critic   │ │ provenance │ │  queue)    │
 └──────┬──────┘ └──────┬──────┘ └────────────┘ └────────────┘ └────────────┘
        │               │
 ┌──────▼──────┐ ┌──────▼──────────────┐
 │ MLX server  │ │ Tool registry:      │
 │ (OpenAI     │ │  • code_exec (kernel)│
 │  compat) /  │ │  • scanpy helpers   │
 │ Ollama/API  │ │  • GEO/PubMed/Ensembl│
 └─────────────┘ └─────────────────────┘
                          │
                 ┌────────▼─────────┐
                 │ Report builder   │ → .ipynb (executable record) + .html (review)
                 └──────────────────┘
```

---

## 4. LLM provider layer

### 4.1 Interface (strict)

```python
# llm/provider.py
from typing import Protocol
from pydantic import BaseModel

class ChatResult(BaseModel):
    content: str
    tool_calls: list["ToolCall"] = []
    usage: "Usage"
    raw: dict | None = None   # backend-native, for debugging only

class LLMProvider(Protocol):
    def chat(
        self,
        messages: list[dict],
        *,
        model_profile: str = "primary",      # "primary" | "classifier" | "coder"
        sampling: "Sampling" | None = None,  # temp/top_p — ORTHOGONAL to model choice
        response_schema: dict | None = None, # JSON schema → constrained/structured output
        tools: list["ToolSpec"] | None = None,
    ) -> ChatResult: ...

    def embed(self, texts: list[str], *, model_profile: str = "embeddings") -> list[list[float]]: ...
```

**Two design rules Claude Code must enforce:**
- `model_profile` (which weights) and `sampling` (temperature/top_p) are **separate arguments**. Never conflate "fast" with "hotter." Classification runs at temp≈0 regardless of which model serves it.
- `chat` returns a **structured `ChatResult`**, not a raw dict. Each adapter normalizes to it.

### 4.2 Adapters
- `MLXOpenAIProvider` — talks to any OpenAI-compatible endpoint (LM Studio server, `mlx_lm.server`, mlx-vlm server). Primary path.
- `OllamaProvider` — fallback local.
- `CloudProvider` — OpenAI/Anthropic/Gemini. Kept only to preserve the agnostic abstraction; **disabled in the shipped config** (local-first, §1.7). Enabling it requires an explicit opt-in flag.

All adapters share one HTTP client and one config block. Adding a backend = adding one file.

### 4.3 Structured output & tool calling (critical for local models)
Small local models are unreliable at native tool calls and free-text plan parsing. Enforce structure in this priority order:
1. **Server-side JSON schema** via `response_format` (LM Studio / mlx-vlm expose this; `mlx_lm.server` via Outlines). Preferred.
2. **Prompted JSON + pydantic validation + one repair retry** if the server can't constrain.
3. Tool calls are **emulated**: model emits a typed JSON action, a parser validates against the `ToolSpec`, a repair prompt fires on failure. Do **not** assume OpenAI-style native tool schemas work on the 9B tier.

The **planner** always emits a schema-validated step list (see §9 data models). Never regex free text.

---

## 5. Execution engine (per-Run uv env + persistent kernel + handles)

### 5.1 Why
AnnData is hundreds of MB to several GB, not JSON-serializable, and must never be prompted. Tools mutate objects living in a long-lived kernel and return only metadata. Each Run is also **environment-isolated**: its own `uv` venv, so one analysis's package installs can never contaminate another's, and the exact environment is reproducible.

### 5.2 Per-Run environment provisioning (uv)
Every Run provisions a fresh venv **before** the analysis graph starts:

1. `uv venv .runs/<run_id>/venv` — creates the environment. uv **hardlinks from a shared global cache**, so per-Run venvs are fast and disk-cheap even though each is nominally isolated; repeated base installs cost near-zero after the first.
2. `uv pip install` the **base set** (pinned in `configs/base_env.txt`): `ipykernel`, `anndata`, `scanpy`, `numpy`, `pandas`, `scipy`, `scikit-misc`, `igraph`, `leidenalg`, `matplotlib`, `seaborn`. This is the floor every scRNA-seq Run needs.
3. Launch the Run's Jupyter kernel **bound to that venv's Python** (`.../venv/bin/python -m ipykernel_launcher`), via `jupyter_client.KernelManager`. The kernel therefore imports from the Run's venv, not the host.
4. **On-demand installs mid-analysis** go through the `install_package` tool (§6.2), which runs `uv pip install <pkg>` **into the Run's venv** — e.g. `harmonypy`, `scvi-tools`, `decoupler`, `gseapy`, `celltypist`, `scrublet` — only when a step actually needs them. The kernel picks up newly installed packages on next import.
5. **Capture the lockfile.** After provisioning and after each on-demand install, run `uv pip freeze` (or `uv lock`) and store the result in provenance. The report pins the exact environment (aligns with FAIR practice).

**Network note:** PyPI/uv-index access for installs is **provisioning egress**, not runtime data egress — it downloads packages, never user data. It is whitelisted separately from analysis-cell network (which stays blocked) and from the knowledge layer. See §13.1.

### 5.3 Execution design
- One **Jupyter kernel per Run**, running inside the Run's uv venv. State (loaded AnnData, intermediate objects) persists across tool calls for that Run.
- A **handle registry**: `adata`, `adata_sub_<cluster>`, etc. Tools reference handles by name; the object stays in the kernel.
- Every code execution returns `ExecResult { stdout, produced_handles, figures: [paths], tables: [paths], summary }`. Only `summary` + small tables go to the LLM.
- **Sandboxing (required before running model-written code):** per-cell wall-clock timeout, memory ceiling, output-size cap, and **no network from analysis cells** — network is reachable only via the `install_package` tool (PyPI whitelist) and the knowledge-layer API plugins (scientific-source whitelist), never from arbitrary model-written code. Kill + report on breach.

### 5.4 Engine contract
```python
class ExecEngine:
    def provision(self, run_id: str, base_env: Path) -> "EnvInfo": ...   # uv venv + base install + kernel bind
    def install(self, run_id: str, packages: list[str]) -> "EnvInfo": ...# uv pip install into Run venv, re-freeze lockfile
    def run_code(self, run_id: str, code: str, *, timeout_s: int = 300) -> ExecResult: ...
    def get_handle_summary(self, run_id: str, name: str) -> HandleSummary: ...
    def snapshot(self, run_id: str) -> Path: ...     # write .h5ad + kernel state + lockfile for resume
    def teardown(self, run_id: str, *, keep_venv: bool = False) -> None: ...

class EnvInfo(BaseModel):
    venv_path: Path
    python: Path
    installed: list[str]      # from uv pip freeze
    lockfile: Path            # captured into provenance
```

**Resumability:** a snapshot includes the lockfile, so resuming a Run rebuilds the identical environment from the pinned versions before restoring kernel state.

---

## 6. Tool / plugin system

### 6.1 Interface
```python
class Tool(Protocol):
    name: str
    input_schema: dict     # JSON schema, used for emulated tool calls
    def execute(self, args: dict, ctx: "RunContext") -> "ExecResult | dict": ...
```

### 6.2 Core tools (build these)
- **`code_exec`** — runs model-written Python in the Run's kernel. The primary analysis tool. No direct network.
- **`install_package`** — the **only** way analysis-side code gains a new dependency. Runs `uv pip install <pkg>` into the Run's venv (§5.2), re-freezes the lockfile into provenance, and returns updated `EnvInfo`. Guarded by a PyPI-package allowlist/policy (name validation, optional pin, size cap). The agent calls this when a step needs e.g. `harmonypy`, `scvi-tools`, `decoupler`, `gseapy`, `celltypist`, `scrublet`.
- **`scanpy_ops`** — thin, *parameterized* helpers for steps we want deterministic and logged (QC, normalize, HVG, PCA, neighbors, leiden, rank_genes_groups, subset). These wrap scanpy but expose parameters; they are **not** a fixed pipeline.
- **`plotting`** — standard figures (UMAP colored by X, dotplot of markers, violin, DE volcano, abundance barplot). Saves to the Run's figure dir and registers provenance.
- **Knowledge-layer API plugins** (the one runtime egress; own scientific-source whitelist, isolated from analysis cells): `pubmed`, `geo`, `ensembl`, `uniprot`, optionally `cellxgene`. Each returns structured data, cached to SQLite. Send **minimal queries** (gene symbols, signature names, search terms) — never raw expression data.

### 6.3 Rule
If the model can reasonably write it as short scanpy/pandas code → `code_exec` (installing deps via `install_package` first if needed). If it's an external scientific-source lookup, an expensive deterministic op, or something with a strict contract → a plugin. Don't build a plugin per analysis step.

---

## 7. Biology reasoning layer (the differentiator)

### 7.1 Prior / marker knowledge base
A structured store (SQLite + YAML seed files), **not prose in a prompt**:
- Cell-type marker sets and gene signatures, per tissue context.
- **Seed content:** the Davies et al. 2013 macrophage framework and the user's own inflammatory-monocyte / resident-macrophage / recruited-macrophage signatures. Make the schema extensible so the user adds signature sets over time.
- Expected-composition priors per tissue (e.g. "solid tumor → expect tumor, immune infiltrate, stromal compartments").

Schema:
```
signature_sets(id, name, tissue_context, source_ref, genes[], direction, notes)
composition_priors(id, tissue_type, expected_compartments[], source_ref)
```

### 7.2 System classifier
`classify_system(metadata) -> SystemType` (`immune_homogeneous` | `heterogeneous_tissue` | ...), backed by the KB, not hardcoded strings. Output drives the analysis plan (deconvolution/origin-inference first for heterogeneous tissue; activation-state/trajectory for PBMC).

### 7.3 Validation critic
After clustering + annotation, a **separate LLM node** checks:
- Annotation vs expected composition ("tumor tissue, zero stromal → FLAG").
- Marker coherence (does the assigned type's signature actually score high in that cluster?).
Contradictions → review queue, not silent acceptance. This is the loop a linear `for step in steps` cannot express — it requires the cyclic graph in §8.

---

## 8. Orchestration — LangGraph stateful graph

Use **LangGraph** (you already work in it) with a **SqliteSaver checkpointer** so runs are durable and resumable. The graph has **cycles** (cluster → inspect → recluster) and **branches** (from the system classifier) — not the rigid plan→execute→synthesize line.

### 8.1 The scRNA-seq iterative subclustering loop (primary use case)

```
load ─► QC ─► normalize ─► HVG ─► PCA ─► [batch? ─► integrate] ─► cluster
                                                                      │
                                                    coarse annotate (priors)
                                                                      │
                                                    system classifier (branch)
                                                                      │
                                        ┌─────── interestingness scoring ───────┐
                                        │  push interesting clusters to FRONTIER │
                                        └──────────────────┬─────────────────────┘
                                                           │  (work queue, budgeted)
                    ┌──────────────────────────────────────▼───────────────┐
                    │  POP LOOP (per frontier population, recursive depth≤N)│
                    │   subset → re-HVG/PCA/cluster on subset               │
                    │   annotate subclusters (priors)                       │
                    │   derive insights: markers, DE vs siblings,           │
                    │       differential abundance across conditions,       │
                    │       signature/pathway enrichment, trajectory (opt)  │
                    │   CRITIC validates annotation                         │
                    │   recurse? (heterogeneity & budget & depth) ──────────┼─► back to POP LOOP
                    └──────────────────────────┬───────────────────────────┘
                                               │
                              cross-population synthesis
                                               │
                              broader-implications reasoning (§8.3)
                                               │
                                      report assembly (§10)
```

### 8.2 Interestingness scoring
Rank clusters for subclustering by a composite score: internal heterogeneity (e.g. silhouette / sub-structure), **differential abundance across conditions**, ambiguous/novel marker profile, hits against disease-associated signatures in the KB, and cluster size. Frontier is a budgeted priority queue (cap total subcluster passes; cap recursion depth).

### 8.3 Broader-implications reasoning (explicit deliverable)
A dedicated node takes each subpopulation's signature + annotation and:
1. Retrieves related literature: local paper vector DB **and** the PubMed plugin.
2. Reasons about functional / disease relevance **grounded in retrieved context**.
3. **Labels every statement** as `data-supported`, `literature-supported`, or `hypothesis`, with citations for the middle category.

**Guardrail (mandatory):** implications must cite retrieved sources; ungrounded claims are marked `hypothesis` and flagged. This prevents confident hallucinated biology — the failure mode that makes a research tool untrustworthy.

---

## 9. Core data models (pydantic)

```python
class Sampling(BaseModel):
    temperature: float = 0.2
    top_p: float = 0.95

class PlanStep(BaseModel):
    id: str
    intent: str                      # human-readable goal
    tool: str                        # "code_exec" | "scanpy_ops" | "pubmed" | ...
    args: dict
    depends_on: list[str] = []

class Plan(BaseModel):
    steps: list[PlanStep]            # planner emits this via JSON schema, validated

class InsightCard(BaseModel):
    population: str                  # e.g. "Macrophage / subcluster 3"
    annotation: str
    top_markers: list[str]
    evidence: list[str]              # data-supported findings
    implications: list["Claim"]      # each tagged + optionally cited
    figures: list[str]
    caveats: list[str]

class Claim(BaseModel):
    text: str
    support: Literal["data", "literature", "hypothesis"]
    citations: list[str] = []
```

---

## 10. Async run + review workflow

### 10.1 Run lifecycle
- `Run` is durable (SQLite row: id, dataset ref, question, status, checkpoint ref, artifact dir).
- FastAPI submits a Run → background worker executes it → LangGraph checkpoints each node → on completion, report is assembled. User polls status / fetches report.
- Keep the worker simple for single-user local: a background process + SQLite job table (RQ/Huey optional if durability across crashes is wanted). LangGraph's checkpointer already makes the *analysis* resumable.

### 10.2 Report (the review artifact)
Produce **two linked outputs**:
- **`report.ipynb`** — the executable record: every code cell the agent ran, in order, with outputs and figures. Reproducible; the user can re-run or edit. (Build via papermill-style parameterized execution or by appending executed cells.)
- **`report.html`** — self-contained review view (nbconvert or a Jinja template): narrative summary, one **insight card per population** (§9), embedded figures, the decision log, flagged review items, and suggested next experiments.

### 10.3 Review queue
Agent-flagged items the user works through later: contradictory annotations, surprising findings, low-confidence decisions, ungrounded implications. Stored in SQLite, rendered at the top of `report.html`. Default is **autonomous with flagging**, not blocking; optionally support LangGraph `interrupt()` for a human-in-the-loop mode.

---

## 11. Repo structure

```
bioagent/
├── pyproject.toml
├── configs/
│   ├── hardware/16gb.yaml  24-36gb.yaml  48gb.yaml     # model profiles per tier
│   ├── base_env.txt        # pinned base package set for every Run's uv venv
│   ├── network_allowlist.yaml  # provisioning + knowledge-layer whitelists (§13.1)
│   └── default.yaml
├── bioagent/
│   ├── api/                 # FastAPI: submit, status, report, review
│   ├── worker/              # background run executor
│   ├── llm/
│   │   ├── provider.py      # Protocol + ChatResult
│   │   ├── mlx_openai.py  ollama.py  cloud.py
│   │   ├── router.py        # profile selection (deterministic; learned later)
│   │   └── structured.py    # schema enforce + repair-retry + emulated tool calls
│   ├── exec/
│   │   ├── venv.py          # per-Run uv venv provision + on-demand install + lockfile
│   │   ├── kernel.py        # jupyter_client kernel bound to the Run venv
│   │   ├── handles.py       # handle registry + summaries
│   │   └── sandbox.py       # timeouts, mem cap, egress allowlist (§13.1)
│   ├── tools/
│   │   ├── code_exec.py  install_package.py  scanpy_ops.py  plotting.py
│   │   └── apis/pubmed.py  geo.py  ensembl.py  uniprot.py   # knowledge layer (only runtime egress)
│   ├── bio/
│   │   ├── priors_kb.py     # marker/signature store + seed loader
│   │   ├── classifier.py    # system classification
│   │   └── critic.py        # annotation validation
│   ├── graph/
│   │   ├── build.py         # LangGraph assembly + SqliteSaver
│   │   ├── nodes/           # qc, cluster, score, poploop, implications, synth
│   │   └── state.py         # graph state (frontier, completed, artifacts)
│   ├── memory/
│   │   ├── vector.py        # LanceDB (papers, results)
│   │   ├── store.py         # SQLite (experiments, runs, cache)
│   │   └── provenance.py    # figure/table/claim → code + data version
│   ├── report/
│   │   ├── notebook.py      # build report.ipynb
│   │   └── html.py          # build report.html + insight cards
│   ├── models.py            # pydantic data models (§9)
│   └── seeds/               # Davies 2013 + user signature YAMLs
└── tests/
```

---

## 12. Build milestones (do in order; gate on acceptance criteria)

**M1 — LLM layer + structured output.**
Provider interface, MLX-OpenAI adapter, schema-enforced planner, emulated tool call + repair.
*Accept:* planner returns a schema-valid `Plan` from the 9B and the 27B; malformed output self-repairs within one retry; `model_profile`/`sampling` are independently controllable.

**M2 — Execution engine + per-Run uv env + sandbox.**
uv venv provisioning with base install, kernel bound to the Run venv, `install_package` on-demand install with lockfile capture, handle registry, `ExecResult`, timeouts/mem cap/network isolation, snapshot/resume.
*Accept:* a fresh Run provisions a uv venv with the base set and launches a kernel inside it; mid-run `install_package("harmonypy")` succeeds and the new package imports in the same kernel; the lockfile is captured to provenance; analysis-cell network is blocked while `install_package` (PyPI) and knowledge plugins still work; load a 3–5 GB `.h5ad` in backed mode on a 16 GB machine; run 20 sequential code cells sharing state; a runaway cell is killed by timeout; snapshot + resume rebuilds the identical env from the lockfile and restores state.

**M3 — Core tools + provenance.**
`code_exec`, `scanpy_ops`, `plotting`, provenance linking figures→code→data version.
*Accept:* a single-cluster QC→normalize→HVG→PCA→UMAP→leiden run produces a UMAP figure whose provenance record reconstructs the exact code + input hash.

**M4 — Biology priors + classifier + critic.**
Seed the KB (Davies 2013 + user signatures); classifier; critic node.
*Accept:* PBMC vs tumor metadata route to different analysis plans; a deliberately mislabeled cluster is FLAGGED by the critic against composition priors.

**M5 — LangGraph loop (single pass).**
Global pass → coarse annotate → interestingness scoring → **one** subcluster level, checkpointed.
*Accept:* on a public burn/wound or PBMC dataset, the agent selects ≥1 interesting population, subclusters it, annotates subclusters, and checkpoints; killing the process mid-run and resuming continues from the last node.

**M6 — Recursion + implications + synthesis.**
Recursive subclustering (depth≤N, budgeted), broader-implications node with RAG + PubMed and claim tagging.
*Accept:* recursion respects depth/budget caps; every implication is tagged `data`/`literature`/`hypothesis`; literature claims carry citations; ungrounded claims are flagged.

**M7 — Async runs + report + review queue.**
FastAPI submit/status/report, background worker, `report.ipynb` + `report.html`, review queue.
*Accept:* submit a Run, disconnect, return later to a complete `report.html` with per-population insight cards, embedded figures, a decision log, and a populated review queue; `report.ipynb` re-runs end-to-end.

**M8 — Multi-model routing (optional, 48 GB+).**
Deterministic router for classifier/coder profiles; measure whether a dedicated coder beats the primary before enabling it.
*Accept:* routing is config-gated per hardware tier; 16 GB profile never loads more than primary + embeddings concurrently.

---

### 13.1 Network policy (local-first, single knowledge exception)

Three egress tiers; everything else is local-only and blocked. Implement as an allowlist enforced at the process/proxy level, not just by convention.

| Tier | Who may use it | Destinations (whitelist) | May it send user data? |
|---|---|---|---|
| **Provisioning** | `install_package` tool + one-time model-weight download | PyPI / uv index; HF or model registry (weights only) | **No** — downloads packages/weights only |
| **Knowledge layer** | `pubmed` / `geo` / `ensembl` / `uniprot` / `cellxgene` plugins | those scientific-source APIs only | **Minimal queries only** (gene symbols, signature names, search terms) — never expression matrices or raw data |
| **Local-only (blocked from network)** | LLM inference, embeddings, vector store, structured store, `code_exec` analysis cells, report builder | — | n/a |

Rules: LLM inference is **local** (MLX/Ollama); cloud providers are **off by default** and require explicit opt-in. Analysis-cell code has **no network** — a dependency arrives only via `install_package` (provisioning tier), a fact only via a knowledge plugin. No user dataset, expression matrix, or intermediate result ever leaves the machine.

### 13.2 Checklist

- **Biology hallucination:** implications must be tagged and (for literature claims) cited; critic validates annotations vs priors; ungrounded claims → review queue.
- **Sandbox:** model-written code runs with timeout, memory ceiling, output cap, and **no direct network** — deps via `install_package`, facts via knowledge plugins, nothing else.
- **Data in context:** never serialize AnnData into a prompt; only summaries/handles.
- **Provenance:** no figure/table/claim without a provenance record; every Run stores its **uv lockfile** + input data hashes (aligns with FAIR-pipeline practice).
- **Memory hygiene:** vector memory stores results/papers, not raw expression matrices.
- **Env isolation:** one uv venv per Run; on-demand installs never touch the host or other Runs; teardown removes the venv unless `keep_venv` is set for debugging.

---

## 14. Verified model landscape (as of July 2026 — confirm before pinning)

- **Qwen3.6** ships as **27B dense** and **35B-A3B MoE** (~3B active), Apache-2.0, MLX-quantized. 27B at MLX 4-bit ≈ 18–20 GB unified memory; 35B-A3B 4-bit ≈ ~22 GB. Both fit the 48 GB dev machine comfortably. Sources: huggingface.co/unsloth/Qwen3.6-27B-MLX-8bit, ollama.com/library/qwen3.6, codersera.com Qwen3.6 guide.
- **16 GB tier:** Qwen3.5-9B MLX 4-bit runs on 8 GB+ at high throughput and is unusually strong for its size; Qwen3.5-4B for 8 GB machines. Source: modelfit.io Qwen-on-Mac RAM-tier guide, willitrunai.com MLX guide.
- **Dedicated coder:** Qwen3-Coder-Next (80B MoE, 3B active) needs ~46 GB → 64 GB+ machines only; on 48 GB the primary writes code. Source: modelfit.io.
- **Structured output on MLX:** OpenAI-compatible `response_format` json_schema is supported via LM Studio's mlx-engine and mlx-vlm server (Outlines-based constrained decoding); `mlx_lm.server` can use Outlines directly. Prompted-JSON + validation + repair is the universal fallback. Sources: lmstudio.ai structured-output docs, github.com/Blaizzy/mlx-vlm.

Because these move monthly, re-verify exact model IDs and RAM figures at build time and keep them in `configs/hardware/*.yaml` so swapping is a config edit, not a code change.

---

## 15. Open decisions for the user (defaults chosen; override if desired)

1. **Job durability:** default = background process + SQLite + LangGraph checkpointer. Upgrade to RQ/Huey only if you need crash-resilient queuing.
2. **Integration method** when batches exist: default = Harmony (fast, light). scVI is heavier and GPU-hungry — enable only on the dev machine.
3. **Serving backend:** default = LM Studio server for its solid json_schema support; `mlx_lm.server` + Outlines if you prefer a pure-CLI stack.
4. **Recursion depth / budget caps:** default depth ≤ 2, ≤ 8 total subcluster passes per Run. Tune per dataset.