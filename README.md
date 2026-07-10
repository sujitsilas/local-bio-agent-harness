# A local, style-aware bioinformatics coding assistant

Chat with a **local** LLM ([opencode](https://opencode.ai) + a Qwen3.6 model on your own
Mac) that writes single-cell / omics analysis and **publication-ready figures in *your*
code style** — because it has a searchable corpus of your own code and greps it before
writing anything.

Nothing leaves your machine. No API keys.

```
opencode  ──►  Ollama (qwen3.6-coding, local)      the model
    │
    ├── AGENTS.md          tells it to mirror your style
    └── examples/          your code, extracted + greppable  ◄── ingest.py ── examples_raw/ (drop notebooks here)
```

## Why Ollama and not mlx-lm

`mlx_lm.server` is faster on paper, but its KV cache grows unbounded and Metal wires that
memory, so a long coding session OOM-crashes with no warning (a known, unfixed bug as of
0.31.x — exactly the failure mode for an all-day assistant). Ollama enforces a fixed
context, so it stays stable. We use `qwen3.6:35b-a3b-mxfp8` — the Metal-optimized 8-bit
float weights, which are noticeably better than the generic GGUF Q4_K_M Ollama pulls by
default — tuned via a [`Modelfile`](Modelfile) (16K context, no presence penalty).

## Setup (Apple silicon)

```bash
brew install ollama          # runtime
npm i -g opencode-ai          # the agent

# serve the tuned local model (pulls mxfp8 ~37GB + builds qwen3.6-coding the first time)
./scripts/serve_ollama.sh
```

Then teach it your style — drop your notebooks/scripts into `examples_raw/` and ingest:

```bash
cp ~/my_analysis.ipynb examples_raw/
python ingest.py             # extracts code -> examples/*.py + examples/INDEX.md
```

`ingest.py` is stdlib-only (any Python). It pulls **code only** (never outputs/data),
splits it into one snippet per task titled by your own comments, and records the key
plotting/analysis calls + libraries in a grep-friendly index.

## Use it

```bash
opencode                     # TUI chat in this folder
# or one-shot:
opencode run "make a Burn-vs-Sham volcano plot for the recruited macrophages"
```

opencode greps `examples/` for the relevant pattern, reads your snippet, and writes new
code that matches your imports, your matplotlib/seaborn + adjustText conventions, your
palettes and helper functions, and your `# ═══` section style — saved at `dpi=300`,
publication-ready. Config: [`opencode.json`](opencode.json); instructions:
[`AGENTS.md`](AGENTS.md).

### Remote use (assistant on a second Mac)

Point the second machine's `opencode.json` `baseURL` at `http://<server-ip>:11434/v1`
(`ipconfig getifaddr en0`), or tunnel: `ssh -L 11434:localhost:11434 user@your-mac`.

## Layout

```
opencode.json     opencode config → local Ollama model + AGENTS.md
AGENTS.md         "write in this user's style; grep examples/ first"
Modelfile         tuned qwen3.6-coding (mxfp8, 16K ctx, no presence penalty)
ingest.py         examples_raw/*.ipynb|*.py  →  examples/*.py + INDEX.md
examples/         your code, extracted + greppable (committed)
examples_raw/     drop zone for raw notebooks (gitignored — can be large)
scripts/serve_ollama.sh   pull + build + serve the model, kept warm
```
