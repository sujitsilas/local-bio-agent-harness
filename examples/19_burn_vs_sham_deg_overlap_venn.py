"""Burn vs Sham DEG overlap (Venn) + GO enrichment of the condition-UNIQUE genes

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, numpy
Key calls: def _barh, def _enrich, def _sig, def _venn2, gp.enrichr, plt.close, plt.show, plt.subplots
"""

# ══════════════════════════════════════════════════════════════════════════════
# Burn vs Sham DEG overlap (Venn) + GO enrichment of the condition-UNIQUE genes.
#   Per identity × direction: rows = timepoint pairs; cols = Venn | Burn-only GO | Sham-only GO.
# Reuses: DE (or _run_tp), _sig, gp, GO_LIB, contains_excluded, thresholds, _slug, FIGDIR_MAC.
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Circle

BURN_C, SHAM_C = "#C0392B", "#2980B9"
N_GO = 6                                    # top GO terms per side

try:
    DE
except NameError:
    DE = {(ident, cond, pair): _run_tp(ident, cond, *pair)
          for ident in IDENTS for cond in CONDS for pair in PAIRS}

def _sig(df, direction):
    if df is None: return set()
    m = ((df.padj < FDR_THRESH) & (df.lfc > LFC_THRESH)) if direction == "up" \
        else ((df.padj < FDR_THRESH) & (df.lfc < -LFC_THRESH))
    return set(df.loc[m, "gene"])

def _enrich(genes, n=N_GO):
    genes = list(genes)
    if len(genes) < 5: return None
    try:
        res = gp.enrichr(gene_list=genes, gene_sets=GO_LIB, organism="mouse",
                         outdir=None, verbose=False).res2d.copy()
    except Exception as e:
        print("  enrichr error:", e); return None
    res["term_clean"] = res["Term"].apply(lambda t: str(t).split("(")[0].strip().replace("_", " "))
    res["term_clean"] = res["term_clean"].apply(lambda t: t[0].upper() + t[1:] if t else t)
    res = res[~res["term_clean"].apply(contains_excluded)]
    res["nlp"] = -np.log10(res["Adjusted P-value"].astype(float).clip(lower=1e-300))
    return res.sort_values("Adjusted P-value").head(n)

def _venn2(ax, A, B, cA, cB, title=""):
    a, b, ab = len(A - B), len(B - A), len(A & B)
    ax.add_patch(Circle((-0.38, 0), 0.72, color=cA, alpha=0.45, lw=0))
    ax.add_patch(Circle(( 0.38, 0), 0.72, color=cB, alpha=0.45, lw=0))
    ax.text(-0.74, 0, str(a),  ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text( 0.74, 0, str(b),  ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text( 0.00, 0, str(ab), ha="center", va="center", fontsize=15, fontweight="bold", color="white")
    ax.text(-0.5, 0.95, "Burn", ha="center", color=cA, fontweight="bold", fontsize=12)
    ax.text( 0.5, 0.95, "Sham", ha="center", color=cB, fontweight="bold", fontsize=12)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.0, 1.25); ax.set_aspect("equal"); ax.axis("off")
    if title: ax.set_title(title, fontsize=13, fontweight="bold", pad=4)

def _barh(ax, edf, color, title):
    ax.set_title(title, fontsize=12, fontweight="bold")
    if edf is None or edf.empty:
        ax.text(0.5, 0.5, "n < 5 genes\n(or no terms)", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey"); ax.axis("off"); return
    terms = [t if len(t) <= 44 else t[:44] + "…" for t in edf["term_clean"]][::-1]
    vals  = edf["nlp"].tolist()[::-1]
    ax.barh(range(len(vals)), vals, color=color, alpha=0.85, edgecolor="black", lw=0.4)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(terms, fontsize=10)
    ax.set_xlabel("$-$log$_{10}$(padj)", fontsize=11, fontweight="bold")
    ax.axvline(-np.log10(0.05), ls="--", color="grey", lw=0.8)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

for ident in IDENTS:
    for direction in ["up", "down"]:
        fig, axes = plt.subplots(len(PAIRS), 3, figsize=(18, 4.6 * len(PAIRS)), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1, 1.7, 1.7], wspace=0.55, hspace=0.55))
        when = "later" if direction == "up" else "earlier"
        fig.suptitle(f"{ident} — DEGs ↑ at the {when} timepoint:  Burn vs Sham, GO of unique genes",
                     fontsize=18, fontweight="bold", y=1.0)
        for r, (later, earlier) in enumerate(PAIRS):
            B = _sig(DE[(ident, "Burn", (later, earlier))], direction)
            S = _sig(DE[(ident, "Sham", (later, earlier))], direction)
            dlab = f"↑ at {later}" if direction == "up" else f"↑ at {earlier}"
            _venn2(axes[r][0], B, S, BURN_C, SHAM_C, title=f"{later} vs {earlier}\n{dlab}")
            _barh(axes[r][1], _enrich(B - S), BURN_C, f"Burn-only ({len(B - S)} genes)")
            _barh(axes[r][2], _enrich(S - B), SHAM_C, f"Sham-only ({len(S - B)} genes)")
        fig.tight_layout()
        fn = f"venn_enrich_burn_vs_sham_{_slug(ident)}_{direction}"
        fig.savefig(FIGDIR_MAC / f"{fn}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(FIGDIR_MAC / f"{fn}.png", dpi=300, bbox_inches="tight")
        plt.show(); plt.close(fig)

