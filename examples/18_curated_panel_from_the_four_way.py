"""Curated panel from the four-way temporal analysis → dotplot by Type_Timepoint_C

Source: macrophages_resident_recruited.ipynb
Libraries: numpy
Key calls: def _classify, dendrogram, dotplot, plt.savefig, plt.show, sc.pl.dotplot
"""

# ══════════════════════════════════════════════════════════════════════════════
# Curated panel from the four-way temporal analysis → dotplot by Type_Timepoint_C
#   Per compartment (Inf.Mono, Recruited), top-N genes per category:
#   Persistent Burn / Persistent Sham / Transient Burn / Emerging Burn.
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, scanpy as sc, matplotlib.pyplot as plt

TOPN = 10
CATS = ["Persistent Burn", "Emerging Burn", "Transient Burn", "Persistent Sham"]

def _classify(ident, topn=TOPN):
    de  = {tp: _de_bvs(ident, tp) for tp in TPS}
    tps = [t for t in TPS if de.get(t) is not None]
    genes = sorted(set().union(*[set(de[t]["gene"]) for t in tps]))
    L = pd.DataFrame({t: de[t].set_index("gene")["lfc"].reindex(genes) for t in tps}).fillna(0.0)
    S = pd.DataFrame({t: ((de[t].set_index("gene")["padj"] < FDR_THRESH) &
                          (de[t].set_index("gene")["lfc"].abs() > LFC_THRESH)).reindex(genes).fillna(False)
                      for t in tps})
    et = [t for t in ["D7", "D10"] if t in tps]; lt = [t for t in ["D14", "D19"] if t in tps]
    nsig, meanL = S.sum(axis=1), L.mean(axis=1)
    e_sig, l_sig = S[et].any(axis=1), S[lt].any(axis=1)
    e_mean, l_mean = L[et].mean(axis=1), L[lt].mean(axis=1)
    rule = {   # (membership mask, ranking score — higher = better)
        "Persistent Burn": ((nsig >= 2) & (meanL > 0), meanL),
        "Persistent Sham": ((nsig >= 2) & (meanL < 0), -meanL),
        "Transient Burn":  (e_sig & (e_mean > 0) & ~l_sig, e_mean),
        "Emerging Burn":   (l_sig & (l_mean > 0) & ~e_sig, l_mean),
    }
    return {c: list(sco[m].sort_values(ascending=False).index[:topn]) for c, (m, sco) in rule.items()}


# pool top genes per category across compartments, dedupe in category order
comps = [c for c in [INF, RECR] if c]
per   = {c: _classify(c) for c in comps}
seen, var_dict, rows = set(), {}, []
for cat in CATS:
    genes = []
    for c in comps:
        for g in per[c][cat]:
            if g in adata_mac.var_names and g not in seen:
                genes.append(g); seen.add(g); rows.append({"gene": g, "category": cat})
    if genes:
        var_dict[cat] = genes
panel_df = pd.DataFrame(rows)
panel_df.to_csv(FIGDIR_MAC / "fourway_panel_genes.csv", index=False)
print(panel_df["category"].value_counts().to_string(), f"\ntotal genes: {len(panel_df)}")

# Type_Timepoint_C order: Sham block then Burn block, D7→D19
present = set(adata_mac.obs["Type_Timepoint_C"].astype(str))
tt = [f"{ty} {d}" for ty in ["Sham", "Burn"] for d in ["D7", "D10", "D14", "D19"] if f"{ty} {d}" in present]
adata_mac.obs["Type_Timepoint_C"] = pd.Categorical(
    adata_mac.obs["Type_Timepoint_C"].astype(str), categories=tt, ordered=True)

sc.pl.dotplot(
    adata_mac, var_dict, groupby="Type_Timepoint_C", categories_order=tt,
    standard_scale="var", cmap="Reds", dendrogram=False,
    figsize=(27, 3.5), show=False,
)
plt.savefig(FIGDIR_MAC / "dotplot_fourway_panel_by_type_timepoint.png", dpi=300, bbox_inches="tight")
plt.savefig(FIGDIR_MAC / "dotplot_fourway_panel_by_type_timepoint.pdf", bbox_inches="tight")
plt.show()

