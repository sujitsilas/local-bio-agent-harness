#!/usr/bin/env python3
"""Turn dropped notebooks / scripts into a greppable examples corpus of YOUR code style.

Drop `.ipynb` or `.py` files into `examples_raw/` (or pass paths), then run:

    python ingest.py

It extracts CODE ONLY (never outputs/data), groups cells into coherent snippets titled by
your own leading comments, detects the key plotting/analysis calls + libraries in each, and
writes:

  examples/<NN>_<slug>.py   one self-describing snippet per task
  examples/INDEX.md         a grep-friendly manifest (task -> key calls -> libraries)

The agent (opencode) greps this corpus before writing code, so new code mirrors your style.
Stdlib only — runs with any Python.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "examples_raw"
OUT = ROOT / "examples"

# calls worth surfacing so `grep volcano|umap|gsea examples/` finds the right snippet
_KEY = re.compile(
    r"\b(sc\.pl\.\w+|sc\.tl\.\w+|sc\.pp\.\w+|sns\.\w+|plt\.\w+|gseapy\.\w+|gp\.\w+|"
    r"adjust_text|\.plot\(|clustermap|heatmap|volcano|dotplot|violin|umap|dendrogram|"
    r"barplot|boxplot|scatter|rank_genes_groups|scvelo\.\w+|cellrank\.\w+|def\s+\w+)")
_IMPORT = re.compile(r"^\s*(?:import|from)\s+([A-Za-z0-9_]+)", re.M)
_SEP = " ═─━=-•*#·.:_|/\\<>~"  # separator chars the user decorates comments with


def _slug(text: str, n: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "_".join(words[:n]) or "snippet"


def _leading_title(cell: str) -> str | None:
    """First comment line in the cell's leading comment block that has real words —
    skipping the user's `# ═══` separator lines. None if the cell starts with code."""
    for line in cell.splitlines():
        st = line.strip()
        if st.startswith("#"):
            words = re.sub(r"^#+", "", st).strip(_SEP).strip()
            if sum(ch.isalnum() for ch in words) >= 3:
                return words[:80]
            continue  # separator-only comment -> keep scanning the block
        if st == "":
            continue
        break  # reached code before any titled comment
    return None


def _title_from_code(code: str) -> str:
    """Best-effort title when a cell has no titled comment: the dominant call."""
    calls = _KEY.findall(code)
    if calls:
        return calls[0].strip("(")
    for line in code.splitlines():
        line = line.strip()
        if line and not line.startswith(("import", "from", "#")):
            return line[:50]
    return "snippet"


def _code_cells(path: Path) -> list[str]:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        return ["".join(c.get("source", [])) for c in nb.get("cells", [])
                if c.get("cell_type") == "code" and "".join(c.get("source", [])).strip()]
    # a .py file: split on jupyter `# %%` cell markers if present, else whole file
    text = path.read_text()
    if "# %%" in text:
        return [c.strip() for c in re.split(r"^# %%.*$", text, flags=re.M) if c.strip()]
    return [text]


def _group(cells: list[str]) -> list[tuple[str, str]]:
    """Group runs of cells into (title, code) snippets. A cell whose first line is a
    comment starts a new snippet titled by that comment; following un-commented cells join
    it (they are usually continuations of the same task)."""
    groups: list[tuple[str, list[str]]] = []
    for cell in cells:
        title = _leading_title(cell)
        if title is not None or not groups:
            groups.append((title or _title_from_code(cell), [cell]))
        else:
            groups[-1][1].append(cell)  # continuation of the current task
    return [(t, "\n\n".join(cs)) for t, cs in groups]


def _snippet_file(idx: int, title: str, code: str, source: str) -> tuple[str, dict]:
    libs = sorted(set(_IMPORT.findall(code)))
    calls = sorted(set(c.strip("(") for c in _KEY.findall(code)))[:12]
    header = (f'"""{title}\n\n'
              f'Source: {source}\n'
              f'Libraries: {", ".join(libs) or "-"}\n'
              f'Key calls: {", ".join(calls) or "-"}\n'
              f'"""\n')
    name = f"{idx:02d}_{_slug(title)}.py"
    return name, {"name": name, "title": title, "libs": libs, "calls": calls,
                  "content": header + "\n" + code + "\n"}


def ingest(paths: list[Path]) -> None:
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.py"):  # regenerate cleanly
        old.unlink()
    (OUT / "INDEX.md").unlink(missing_ok=True)

    manifest = []
    idx = 0
    for path in paths:
        for title, code in _group(_code_cells(path)):
            idx += 1
            name, meta = _snippet_file(idx, title, code, path.name)
            (OUT / name).write_text(meta["content"])
            manifest.append(meta)

    lines = ["# Examples corpus — the user's code style", "",
             "Grep this folder for a task (e.g. `grep -rl volcano examples/`), then read the",
             "matching snippet and mirror its style. Extracted code only; no outputs.", "",
             "| # | task | key calls | libraries |", "| - | - | - | - |"]
    for m in manifest:
        lines.append(f"| {m['name']} | {m['title']} | "
                     f"{', '.join(m['calls'][:6])} | {', '.join(m['libs'][:6])} |")
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n")
    print(f"ingested {len(manifest)} snippets from {len(paths)} file(s) -> {OUT}/")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        args = sorted(RAW.glob("*.ipynb")) + sorted(RAW.glob("*.py")) if RAW.exists() else []
    if not args:
        print(f"drop .ipynb/.py files into {RAW}/ (or pass paths), then re-run.")
        sys.exit(0)
    ingest(args)
