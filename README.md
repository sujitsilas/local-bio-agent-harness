# bioagent-bench — how well can a local LLM agent call cell types?

A controlled benchmark of one narrow capability: **de novo cell-type annotation** by an
LLM agent. The agent clusters a scRNA-seq dataset itself, then calls each cluster's
identity from its differentially-expressed genes + live NCBI literature — with **no
pre-loaded gene→cell-type dictionary**. We score those calls against held-out trusted
labels and report numbers, not vibes: accuracy, ARI, macro-F1, structured-output failure
rate, and **calibration** (does the agent's stated confidence track its correctness?).
Then we **ablate** each component to measure what it's worth.

Everything runs locally on Apple silicon (MLX-served quantized model; the expression data
never enters the model's context — it lives in a Jupyter kernel and only summaries cross).

## What it measures

| metric | question |
|---|---|
| accuracy / macro-F1 | under a majority-vote label map, how often is the call right? |
| Adjusted Rand Index | name-agnostic: does the agent's partition agree with the reference? |
| ECE + reliability table | is the agent's confidence calibrated, or over/under-confident? |
| structured-output failure rate | how often does the local model fail to emit valid JSON? |

## Ablation knobs (the controlled experiment)

| knob | flag | hypothesis it tests |
|---|---|---|
| NCBI literature grounding | `--no-grounding` | does retrieving marker literature improve calls? |
| model reasoning (thinking) | `--no-thinking` | does chain-of-thought help — and at what latency? |
| self-discovered-signature reuse | `--no-reuse` | does the agent reusing its own learned signatures help? |
| critic pass | `--critic` | does a second "critic" agent improve calibration or just add latency? |

`bioagent bench <dataset> --sweep` runs the baseline + one-knob-off for each + the critic,
and prints a table.

## Datasets (trusted, held-out labels)

- **pbmc68k** — `sc.datasets.pbmc68k_reduced()`; `bulk_labels` are **FACS-sorted** populations
  (as close to ground truth as scRNA gets).
- **pbmc3k** — `sc.datasets.pbmc3k_processed()`; curated `louvain` cell-type names.
- any local `.h5ad` via `bioagent bench data.h5ad --reference-col <obs_col>`.

The reference column is used **only for scoring** — never shown to the agent.

## Baseline result (pbmc68k, FACS-sorted labels)

```
| ablation                                | accuracy | ari   | macro_f1 | ece   | struct_fail | n_clusters |
| ground=on,think=on,reuse=on,critic=off  | 0.731    | 0.472 | 0.474    | 0.139 | 0.0         | 10         |

Calibration (ECE=0.139):   the agent is over-confident
| bin        | mean conf | accuracy | n   |
| 0.5–0.6    | 0.60      | 1.00     | 13  |
| 0.7–0.8    | 0.79      | 0.54     | 84  |   <- states .79, right .54
| 0.8–0.9    | 0.87      | 0.75     | 603 |   <- states .87, right .75
```

The headline finding isn't the accuracy — it's the **miscalibration**: the agent's stated
confidence runs well ahead of its correctness. That's the kind of agent-behavior result the
benchmark exists to surface.

## Setup (Apple silicon)

```bash
uv venv && uv pip install -e ".[dev]"
uv pip install "mlx-lm==0.31.3" "transformers==5.10.4"   # see version note below
./scripts/serve_mlx.sh          # serves mlx-community/Qwen3.6-35B-A3B-4bit on :8080
bioagent doctor                 # checks uv, model endpoint
```

> Version pin: mlx_lm 0.31.3 needs transformers 5.x, but 5.13.0 breaks its import and 4.x
> lacks the Qwen3.6 tokenizer — 5.10.4 is the known-good middle.

## Run

```bash
bioagent bench pbmc68k                    # baseline (all knobs on)
bioagent bench pbmc68k --sweep            # full ablation table
bioagent bench pbmc68k --no-thinking      # ~10x faster (reasoning off)
bioagent bench data.h5ad --reference-col cell_type
```

Results (JSON + a markdown table) land in `.runs/benchmarks/<dataset>/`.

> Reasoning-on runs are slow — the 35B model streams thousands of chain-of-thought tokens
> per call (the pbmc68k baseline took ~40 min for 10 clusters). Use `--no-thinking` for
> quick iteration; the ablation sweep quantifies exactly what the reasoning buys you.

## How a call is made (per cluster)

1. leiden clustering on the dataset's embedding (the agent's own partition).
2. `rank_genes_groups` → top DE markers per cluster.
3. optional: NCBI/PubMed search on those markers (objective, not label-driven).
4. the model calls the identity from markers + literature, returning a label + supporting
   markers + citations + a confidence, as validated JSON (reasoning streamed on the first
   attempt; a thinking-off repair guarantees clean JSON if it truncates).
5. confident, novel calls are saved back as signatures the agent can reuse later.

## Layout

```
bioagent/
├── eval/            metrics.py · datasets.py · annotation.py · calling.py · benchmark.py
├── llm/             model-agnostic provider (MLX/Ollama), structured output + streaming
├── exec/            per-Run uv venv + Jupyter kernel (data stays out of the LLM context)
├── bio/priors_kb.py the agent's own discovered signatures (no pre-loaded dictionary)
├── tools/apis/      NCBI knowledge layer (pubmed/geo/ensembl/uniprot)
└── cli.py           bioagent bench · bioagent doctor
tests/               metrics, KB, structured-output, config
```

## Tests

```bash
pytest        # metrics (ARI/PRF/calibration), KB, structured-output repair, config
```
