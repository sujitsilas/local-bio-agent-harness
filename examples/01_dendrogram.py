"""dendrogram

Source: macrophages_resident_recruited.ipynb
Libraries: adjustText, cellrank, matplotlib, numpy, os, pandas, re, scanpy, scipy, scvelo, seaborn, warnings
Key calls: adjust_text, def style_umap, dendrogram, plt.figure, plt.show, plt.subplots, sc.pl.umap, sc.tl.score_genes
"""

import warnings
warnings.filterwarnings('ignore')
import re, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scvelo as scv
import cellrank as cr
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy.ndimage import uniform_filter1d
from adjustText import adjust_text


adata_full = sc.read_h5ad('/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/filtered_final_06252026.h5ad')
adata_mac = sc.read_h5ad('/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/macrophages_final_06252026.h5ad')

import os, pandas as pd, scanpy as sc
from scipy.io import mmread

WDIR = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608"
out  = f"{WDIR}/sce_export"

X        = mmread(f"{out}/matrix.mtx").T.tocsr()
genes    = pd.read_csv(f"{out}/genes.txt",    header=None)[0].values
barcodes = pd.read_csv(f"{out}/barcodes.txt", header=None)[0].values
meta     = pd.read_csv(f"{out}/metadata.csv", index_col=0)

adata_full = sc.AnnData(X=X, obs=meta.loc[barcodes].copy(),
                        var=pd.DataFrame(index=genes))
for f in os.listdir(out):
    if f.endswith('.csv') and f != 'metadata.csv':
        adata_full.obsm['X_' + f[:-4].lower()] = pd.read_csv(f"{out}/{f}", index_col=0).loc[barcodes].values

adata_full.write(f"{WDIR}/filtered_final_burn_sham.h5ad")
print(adata_full)


import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------
# Gene sets — grounded in Davies, Jenkins, Allen & Taylor, Nat Immunol 2013
# ----------------------------------------------------------------------
# Inflammatory (classical Ly6C-hi, CCR2+) monocytes = the recruited precursor pool
inflammatory_mono_genes = [
    "Ly6c2",   # Ly6C  — defining classical/inflammatory monocyte marker
    "Ly6c1",   # Ly6C
    "Ccr2",    # CCR2  — CCR2–CCL2 axis drives recruitment (central to the review)
    "Sell",    # CD62L — classical monocyte
    "Plac8",   # classical monocyte
    "Chil3",   # Ym1   — Ly6C-hi monocyte / early infiltrate
    "F13a1",   # classical monocyte
    "Vcan",    # classical monocyte
    "Hp",      # classical monocyte
    "Gngt2",   # classical monocyte
]

# Recruited / monocyte-derived macrophages (Ly6C-hi monocytes maturing in tissue)
recruited_macs_genes = [
    "Ccr2",    # retained on freshly recruited cells
    "Ly6c2",   # residual monocyte identity
    "Arg1",    # inflammatory monocyte-derived mac (wound)
    "Fn1",     # monocyte-derived
    "Spp1",    # monocyte-derived / scar-associated  (later scRNA-seq)
    "Trem2",   # lipid/scar-associated mac           (later scRNA-seq)
    "Gpnmb",   # monocyte-derived                    (later scRNA-seq)
    "Cd9",     # monocyte-derived
    "Ms4a7",   # monocyte-derived
    "Itgax",   # CD11c — monocyte-derived in tissue
]

# Resident (embryonic, self-renewing) macrophages
resident_macs_genes = [
    "Adgre1",  # F4/80 (high in resident)
    "Mertk",   # MerTK — apoptotic clearance, resident program
    "Timd4",   # Tim-4 — self-renewing resident populations
    "Cd163",   # resident / perivascular
    "Mrc1",    # CD206
    "Folr2",   # resident perivascular                (later scRNA-seq)
    "Lyve1",   # resident perivascular                (later scRNA-seq)
    "Gas6",    # resident
    "Selenop", # Sepp1 — resident
    "C1qa", "C1qb", "C1qc",   # complement — resident core
    "Pf4",     # Cxcl4 — resident
    "Maf",     # MAF — resident identity TF
]

signatures = {
    "inflammatory_mono_score": inflammatory_mono_genes,
    "recruited_macs_score":    recruited_macs_genes,
    "resident_macs_score":     resident_macs_genes,
}

# ----------------------------------------------------------------------
# Keep only genes present (avoids KeyError) and score each signature
# ----------------------------------------------------------------------
for score_name, genes in signatures.items():
    present = [g for g in genes if g in adata_full.var_names]
    missing = sorted(set(genes) - set(present))
    if missing:
        print(f"{score_name}: not found -> {missing}")
    print(f"{score_name}: scoring {len(present)}/{len(genes)} genes")
    sc.tl.score_genes(adata_full, present, score_name=score_name, use_raw=False)

# ----------------------------------------------------------------------
# Per-cell identity = argmax of z-scored signatures (gate ambiguous cells)
# ----------------------------------------------------------------------
score_cols = ["inflammatory_mono_score", "recruited_macs_score", "resident_macs_score"]
labels     = ["Inflammatory Monocytes", "MΦ-Recruited", "MΦ-Resident/Repair"]

Z = adata_full.obs[score_cols].apply(lambda x: (x - x.mean()) / x.std())  # z per signature
assign = np.array(labels)[Z.values.argmax(1)]
assign[Z.values.max(1) < 0.25] = "Ambiguous / low"   # tune threshold as needed

adata_full.obs["mac_identity"] = pd.Categorical(
    assign, categories=labels + ["Ambiguous / low"]
)
print(adata_full.obs["mac_identity"].value_counts())

# ----------------------------------------------------------------------
# Publication-ready figure: 3 score UMAPs + categorical assignment
# ----------------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "Arial",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in Illustrator
})

def style_umap(ax, title):
    ax.set_title(title, fontsize=20, fontweight="bold", pad=12)
    ax.set_xlabel("UMAP 1", fontsize=18, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=0, length=0)    # UMAP coords are arbitrary

fig, axes = plt.subplots(2, 2, figsize=(14, 13))

panels = [
    ("inflammatory_mono_score", "Inflammatory Monocytes"),
    ("recruited_macs_score",    "MΦ-Recruited"),
    ("resident_macs_score",     "MΦ-Resident/Repair"),
]
for ax, (score, title) in zip(axes.flat[:3], panels):
    sc.pl.umap(adata_full, color=score, cmap="RdBu_r", size=12, vcenter=0,
               sort_order=True, frameon=True, show=False, ax=ax, colorbar_loc="right")
    style_umap(ax, title)

# Categorical assignment panel
id_palette = {
    "Inflammatory Monocytes": "#E69F00",
    "MΦ-Recruited":  "#D55E00",
    "MΦ-Resident/Repair":   "#0072B2",
    "Ambiguous / low":       "#D9D9D9",
}

ax = axes.flat[3]
sc.pl.umap(adata_full, color="mac_identity", palette=id_palette, size=12,
           sort_order=False, frameon=True, show=False, ax=ax,
           legend_loc="right margin", legend_fontsize=13)
style_umap(ax, "Assigned identity")

fig.tight_layout()
fig.savefig("macrophage_origin_umap.png", dpi=300, bbox_inches="tight")
fig.savefig("macrophage_origin_umap.pdf", bbox_inches="tight")
plt.show()


from matplotlib.ticker import PercentFormatter

# ----------------------------------------------------------------------
# Macrophage compartment (Mono/MDM + Mφ) + condition column
# ----------------------------------------------------------------------
target_groups = ["Mono.", "MΦ"]
celltype_col = next(
    (c for c in adata_full.obs.columns
     if adata_full.obs[c].astype(str).isin(target_groups).any()),
    None,
)
assert celltype_col is not None, "No obs column contains 'Mono.' or 'MΦ' — set celltype_col manually."
mask = adata_full.obs[celltype_col].astype(str).isin(target_groups).values

type_col = next(
    (c for c in ["Type", "type", "condition", "Condition", "group", "Group",
                 "treatment", "Treatment"] if c in adata_full.obs.columns),
    None,
)
assert type_col is not None, "Set type_col to your Burn/Sham (condition) column."
print(f"Compartment col: {celltype_col!r} | split by: {type_col!r} | {mask.sum()} cells")

# ----------------------------------------------------------------------
# Proportions of the 3 identities within each Type (compartment only)
# ----------------------------------------------------------------------
comp = adata_full.obs.loc[mask, [type_col, "mac_identity"]].copy()
comp = comp[comp["mac_identity"].isin(labels)]          # uses the CURRENT labels
prop = pd.crosstab(comp[type_col], comp["mac_identity"], normalize="index")[labels]
print(prop.round(3))

# ----------------------------------------------------------------------
# Display scores: 0–2, NaN outside the compartment (greyed out)
# ----------------------------------------------------------------------
disp_cols = {}
for score in score_cols:
    v = adata_full.obs[score].astype(float).values.copy()
    v[~mask] = np.nan
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    adata_full.obs[f"{score}_disp"] = (v - vmin) / (vmax - vmin) * 2.0
    disp_cols[score] = f"{score}_disp"

# ----------------------------------------------------------------------
# Palette keyed to the CURRENT labels -> red / orange / green
# ----------------------------------------------------------------------
id_palette = dict(zip(labels, ["#D62728", "#F39C12", "#2CA02C"]))  # infl, recruited, resident

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "Arial",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
NA_GREY = "#E0E0E0"

def style_umap(ax, title):
    ax.set_title(title, fontsize=30, fontweight="bold", pad=12)
    ax.set_xlabel("UMAP 1", fontsize=45, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=45, fontweight="bold")
    ax.tick_params(labelsize=15, length=0)

# 5 columns: 3 UMAPs + thin spacer + proportions bar
fig = plt.figure(figsize=(35, 7))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.18, 0.9], wspace=0.3)

panels = [
    ("inflammatory_mono_score", "Inflammatory Monocytes"),
    ("recruited_macs_score",    "MΦ-Recruited"),
    ("resident_macs_score",     "MΦ-Resident/Repair"),
]
CBAR_TICK_FS = 25

for i, (score, title) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    existing = set(fig.axes)
    sc.pl.umap(adata_full, color=disp_cols[score], cmap="RdBu_r",
               vmin=0, vmax=2, size=12, sort_order=True, frameon=True,
               show=False, ax=ax, na_color=NA_GREY, na_in_legend=False,
               colorbar_loc="right")
    style_umap(ax, title)
    for cax in fig.axes:
        if cax not in existing and cax is not ax:
            cax.tick_params(labelsize=CBAR_TICK_FS, width=1.5, length=5)
            cax.set_yticks([0, 0.5, 1, 1.5, 2])

# proportions bar (last column)
ax_prop = fig.add_subplot(gs[0, 4])
x = np.arange(len(prop)) * 0.55
bottom = np.zeros(len(prop))
for lab in labels:
    ax_prop.bar(x, prop[lab].values, bottom=bottom, width=0.4,
                color=id_palette[lab], edgecolor="white", linewidth=1.2, label=lab)
    bottom += prop[lab].values

ax_prop.set_xticks(x)
ax_prop.set_xlim(x[0] - 0.4, x[-1] + 0.4)
ax_prop.margins(x=0)
ax_prop.set_xticklabels(prop.index, fontsize=35, fontweight="bold",
                        rotation=45, ha="right", rotation_mode="anchor")
ax_prop.set_xlabel(type_col, fontsize=35, fontweight="bold", labelpad=10)
ax_prop.set_ylabel("Proportion of cells", fontsize=35, fontweight="bold", labelpad=10)
ax_prop.set_ylim(0, 1)
ax_prop.yaxis.set_major_formatter(PercentFormatter(1.0))
ax_prop.tick_params(axis="y", labelsize=35, width=1.5, length=6)
for s in ["top", "right"]:
    ax_prop.spines[s].set_visible(False)
ax_prop.legend(fontsize=30, loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False)

fig.savefig("macrophage_origin_umap.png", dpi=600, bbox_inches="tight")
fig.savefig("macrophage_origin_umap.pdf", dpi=600, bbox_inches="tight")
plt.show()


adata_full

sc.pl.umap(adata_full, color='cell_types_full', palette='tab10')
