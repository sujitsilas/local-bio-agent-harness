"""Burn vs Sham DEG overlap (Venn) + top-10 unique genes per side

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, numpy
Key calls: def _sig, def _top_unique, def _venn2, plt.close, plt.show, plt.subplots
"""

# ══════════════════════════════════════════════════════════════════════════════
# Burn vs Sham DEG overlap (Venn) + top-10 unique genes per side.
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Circle

BURN_C, SHAM_C = "#C0392B", "#2980B9"

try:                                   # reuse DE from the previous cell if present
    DE
except NameError:
    DE = {(ident, cond, pair): _run_tp(ident, cond, *pair)
          for ident in IDENTS for cond in CONDS for pair in PAIRS}

def _sig(df, direction):
    if df is None: return set()
    m = ((df.padj < FDR_THRESH) & (df.lfc > LFC_THRESH)) if direction == "up" \
        else ((df.padj < FDR_THRESH) & (df.lfc < -LFC_THRESH))
    return set(df.loc[m, "gene"])

def _top_unique(df, genes, n=10):
    if df is None or not genes: return []
    s = df[df.gene.isin(genes)].copy()
    s["_a"] = s["lfc"].abs()
    s = s.sort_values(["padj", "_a"], ascending=[True, False])   # most sig, then biggest effect
    return s["gene"].head(n).tolist()

def _venn2(ax, A, B, cA, cB, genesA, genesB, title=""):
    a, b, ab = len(A - B), len(B - A), len(A & B)
    ax.add_patch(Circle((-0.38, 0), 0.72, color=cA, alpha=0.45, lw=0))
    ax.add_patch(Circle(( 0.38, 0), 0.72, color=cB, alpha=0.45, lw=0))
    ax.text(-0.74, 0.0, str(a),  ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text( 0.74, 0.0, str(b),  ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text( 0.00, 0.0, str(ab), ha="center", va="center", fontsize=16, fontweight="bold", color="white")
    ax.text(-0.5, 0.95, "Burn", ha="center", color=cA, fontweight="bold", fontsize=13)
    ax.text( 0.5, 0.95, "Sham", ha="center", color=cB, fontweight="bold", fontsize=13)
    if genesA:
        ax.text(-0.75, -1.0, "\n".join(genesA), ha="center", va="top", fontsize=10,
                color=cA, fontweight="bold", linespacing=1.4)
    if genesB:
        ax.text( 0.75, -1.0, "\n".join(genesB), ha="center", va="top", fontsize=10,
                color=cB, fontweight="bold", linespacing=1.4)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-3.4, 1.3); ax.set_aspect("equal"); ax.axis("off")
    if title: ax.set_title(title, fontsize=14, fontweight="bold", pad=6)

rows_out = []
for ident in IDENTS:
    fig, axes = plt.subplots(len(PAIRS), 2, figsize=(11, 6.0 * len(PAIRS)), squeeze=False)
    fig.suptitle(f"{ident}: Burn vs Sham — shared time-varying DEGs (top-10 unique shown)",
                 fontsize=18, fontweight="bold", y=1.0)
    for r, (later, earlier) in enumerate(PAIRS):
        dfB, dfS = DE[(ident, "Burn", (later, earlier))], DE[(ident, "Sham", (later, earlier))]
        for c, (direction, dlab) in enumerate([("up", f"↑ at {later}"), ("down", f"↑ at {earlier}")]):
            B, S = _sig(dfB, direction), _sig(dfS, direction)
            gB, gS = _top_unique(dfB, B - S), _top_unique(dfS, S - B)
            _venn2(axes[r][c], B, S, BURN_C, SHAM_C, gB, gS, title=f"{later} vs {earlier}  ·  {dlab}")
            rows_out.append({"identity": ident, "pair": f"{later}v{earlier}", "direction": direction,
                             "burn_only": len(B - S), "shared": len(B & S), "sham_only": len(S - B),
                             "top_burn_only": ";".join(gB), "top_sham_only": ";".join(gS),
                             "shared_genes": ";".join(sorted(B & S))})
    fig.tight_layout()
    fn = f"venn_burn_vs_sham_overlap_{_slug(ident)}_topgenes"
    fig.savefig(FIGDIR_MAC / f"{fn}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR_MAC / f"{fn}.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

ov = pd.DataFrame(rows_out)
ov.to_csv(FIGDIR_MAC / "burn_sham_deg_overlap_topgenes.csv", index=False)
print(ov[["identity", "pair", "direction", "burn_only", "shared", "sham_only"]].to_string(index=False))

