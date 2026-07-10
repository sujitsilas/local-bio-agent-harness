"""4. Apply mapping, order categories, and assign colors to your AnnData object

Source: macrophages_resident_recruited.ipynb
Libraries: adjustText, math, matplotlib, numpy, pathlib, scanpy, scipy, statsmodels
Key calls: .plot, adjust_text, def _lm, def _lm_contrasts, def _overall_p, def _per_tp_ttest, def _prop_by_tp, def _stars, def _tt, def cluster_gene_order, def style_umap_axes, dotplot
"""

# 4. Apply mapping, order categories, and assign colors to your AnnData object
# Ensure the cluster column is string type to match the dictionary keys
adata_mac.obs["SCT_snn_res.0.5"] = adata_mac.obs["SCT_snn_res.0.5"].astype(str)

adata_mac.obs["macrophage_subtypes"] = (adata_mac.obs["SCT_snn_res.0.5"]
                                      .map(mac_subset_ids)
                                      .astype("category"))

# Set the factor levels to your preferred order

adata_mac.obs["macrophage_subtypes"] = adata_mac.obs["macrophage_subtypes"].cat.set_categories(desired_order)

import numpy as np, pandas as pd, re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── 1. map macrophage_subtypes from adata_mac -> adata_full by barcode ─────────
sub_map  = adata_mac.obs["macrophage_subtypes"].astype(str)
full_sub = sub_map.reindex(adata_full.obs_names)
print(f"mapped {full_sub.notna().sum()}/{adata_full.n_obs} cells")
adata_full.obs["macrophage_subtypes"] = full_sub.values

# ── 2. pick highlight subtypes (tolerant to φ/Φ, spaces, dots, slash) ──────────
key = lambda s: re.sub(r'[^a-z0-9]', '', s.lower().replace('Φ', '').replace('φ', ''))
want = [ 'MΦ-Act', 'Inf. Mono.',
       'MΦ-Res/Rep', 'MDM']
cats   = list(pd.unique(sub_map))
catkey = {key(c): c for c in cats}
highlight = [catkey[key(w)] for w in want if key(w) in catkey]
missing   = [w for w in want if key(w) not in catkey]
if missing:
    print("NOT found among subtypes:", missing, "| available:", cats)
print("highlighting:", highlight)

print("highlighting:", highlight)

# ── 3. plot: highlight in mac_colors, everything else grey ─────────────────────
xy  = adata_full.obsm["X_umap"]
arr = adata_full.obs["macrophage_subtypes"].values
NA_GREY = "#E6E6E6"

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(xy[:, 0], xy[:, 1], s=3, c=NA_GREY, linewidths=0, rasterized=True)   # grey context
for st in highlight:
    m = arr == st
    ax.scatter(xy[m, 0], xy[m, 1], s=0.8, c=mac_colors.get(st, "#999999"),
               linewidths=0, rasterized=True)                                   # colored on top

ax.set_xlabel("UMAP 1", fontsize=24, fontweight="bold")
ax.set_ylabel("UMAP 2", fontsize=24, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(True); ax.spines[side].set_linewidth(1.4)

# ── horizontal, multi-row legend at the bottom ────────────────────────────────
handles = [Patch(facecolor=mac_colors.get(st, "#999999"), label=st) for st in highlight]
ncol =3        # 2 rows
ax.legend(handles=handles, title="x", fontsize=13, title_fontsize=15,
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.04),
          ncol=ncol, columnspacing=1.2, handletextpad=0.5)

fig.tight_layout()
#fig.savefig(FIGDIR_MAC / "umap_full_highlight_macsubtypes.png", dpi=300, bbox_inches="tight")
#fig.savefig(FIGDIR_MAC / "umap_full_highlight_macsubtypes.pdf", bbox_inches="tight")
plt.show()


adata_mac = sc.read_h5ad("/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/macrophages_final_06252026.h5ad")

import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)

inflammatory_mono_genes = ["Ly6c2","Ly6c1","Ccr2","Sell","Plac8","Chil3","F13a1","Vcan","Hp","Gngt2", "Bst2"]
recruited_macs_genes    = ["Ccr2","Ly6c2","Arg1","Fn1","Spp1","Trem2","Gpnmb","Cd9","Ms4a7","Itgax"]
resident_macs_genes     = ["Adgre1","Mertk","Timd4","Cd163","Mrc1","Folr2","Lyve1","Gas6",
                           "Selenop","C1qa","C1qb","C1qc","Pf4","Maf", "Cx3cl1", "Cmkbrl1"]

# additional markers spanning the annotated macrophage subtypes
other_markers = [
    "Il1b","Tnf","Nos2","Cxcl2","Ccl3","Ccl4","S100a8","S100a9",   # M1 / activated inflammatory
    "Retnla",                                                       # M2 repair
    "Fabp5","Lpl","Lgals3",                                         # lipid-associated (LAM)
    "Cd74","H2-Ab1","H2-Aa","H2-Eb1",                              # antigen-presenting (APM)
    "Isg15","Ifit1","Ifit3","Irf7","Rsad2","Cxcl10",              # interferon-stimulated
    "Mki67","Top2a",                                               # proliferating
    "Csf1r","Fcgr1","Cd68","Apoe", "Mrc1", "Arg1", "Nos2", "Cd64"                               # pan-macrophage
]

# combine, dedup (preserve order), keep only genes present
seen, panel = set(), []
for g in inflammatory_mono_genes + recruited_macs_genes + resident_macs_genes + other_markers:
    if g not in seen:
        seen.add(g); panel.append(g)
present = [g for g in panel if g in adata_mac.var_names]
missing = [g for g in panel if g not in adata_mac.var_names]
if missing:
    print("Not in adata_mac.var_names:", missing)
print(f"{len(present)} genes on the dot plots")

# ── two dot plots: by subtype, and by SCT cluster ─────────────────────────────
for groupby, fname, title in [
    ("macrophage_subtypes", "dotplot_markers_by_subtype",       ""),
    ("SCT_snn_res.0.5",     "dotplot_markers_by_SCT_res0.5",    ""),
]:
    assert groupby in adata_mac.obs.columns, (
        f"{groupby!r} not in obs. Candidates: "
        f"{[c for c in adata_mac.obs.columns if 'subtype' in c.lower() or 'snn' in c.lower()]}"
    )
    n_groups = adata_mac.obs[groupby].astype(str).nunique()

    axd = sc.pl.dotplot(
        adata_mac, present, groupby=groupby,
        use_raw=False, standard_scale="var",        # scale each gene 0–1 for comparability
        cmap="Reds", dot_max=1.0,
        figsize=(0.34 * len(present) + 2, 0.5 * n_groups + 2),
        title=title, show=False,
    )
    # enlarge labels for publication
    mainax = axd["mainplot_ax"]
    mainax.tick_params(axis="x", labelsize=12)
    mainax.tick_params(axis="y", labelsize=15)
    mainax.set_title(title, fontsize=22, fontweight="bold", pad=14)
    for lbl in mainax.get_yticklabels():
        lbl.set_fontweight("bold")

    fig = plt.gcf()
    fig.savefig(FIGDIR_MAC / f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR_MAC / f"{fname}.pdf", bbox_inches="tight")
    plt.show()


import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, leaves_list

ID_COL = "mac_identity"

# identity order: Inflammatory Monocytes -> Recruited -> Resident
have = list(pd.unique(adata_mac.obs[ID_COL].astype(str)))
find = lambda key: next((h for h in have if key in h.lower()), None)
mono, recr, resi = find("inflamm"), find("recruit"), find("resident")
id_order = [x for x in [mono, recr, resi] if x]          # <- Inf, Recruited, Resident
print("identity order:", id_order)

adata_sub = adata_mac[adata_mac.obs[ID_COL].astype(str).isin(id_order)].copy()
adata_sub.obs[ID_COL] = pd.Categorical(adata_sub.obs[ID_COL].astype(str),
                                       categories=id_order, ordered=True)

def cluster_gene_order(adata, genes, groupby, group_order, method="average"):
    """Order genes by similarity of BOTH mean expression and % cells expressing (dot color + size)."""
    g = [x for x in genes if x in adata.var_names]
    X = adata[:, g].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    grp = adata.obs[groupby].astype(str).values
    Xdf = pd.DataFrame(X, columns=g)

    M = Xdf.assign(_grp=grp).groupby("_grp").mean().reindex(group_order)            # mean expr  (color)
    F = (Xdf > 0).assign(_grp=grp).groupby("_grp").mean().reindex(group_order)      # frac expressing (size)
    sc01 = lambda D: ((D - D.min()) / (D.max() - D.min())).fillna(0)                # per-gene 0–1
    Mz, Fz = sc01(M), sc01(F)
    feat = pd.concat([Mz, Fz], axis=0)            # (2 × n_groups) × genes -> color + size profile

    dom = Fz.values.argmax(0)                     # block by identity expressing it in the MOST cells
    ordered = []
    for gi in range(len(group_order)):
        block = [g[j] for j in range(len(g)) if dom[j] == gi]
        if len(block) > 2:
            Z = linkage(feat[block].T.values, method=method, metric="euclidean",
                        optimal_ordering=True)    # nearest-neighbour genes adjacent
            block = [block[k] for k in leaves_list(Z)]
        ordered += block
    return ordered

present_clustered = cluster_gene_order(adata_sub, present, ID_COL, id_order)
print("Clustered gene order:", present_clustered)


present_clustered = cluster_gene_order(adata_sub, present, ID_COL, id_order)
print("Clustered gene order:", present_clustered)

axd = sc.pl.dotplot(
    adata_sub, present_clustered, groupby=ID_COL,
    use_raw=False, standard_scale="var",
    cmap="Reds", dot_max=1.0,
    figsize=(18, 3.5),
    title="", show=False,
)

mainax = axd["mainplot_ax"]
mainax.tick_params(axis="x", labelsize=16)
mainax.tick_params(axis="y", labelsize=18)
for lbl in mainax.get_xticklabels(): lbl.set_fontweight("bold")
for lbl in mainax.get_yticklabels(): lbl.set_fontweight("bold")

for key in ("size_legend_ax", "color_legend_ax"):
    lax = axd.get(key)
    if lax is not None:
        lax.tick_params(labelsize=9)
        if lax.get_title():
            lax.set_title(lax.get_title(), fontsize=9)

fig = plt.gcf()
#fig.savefig(FIGDIR_MAC / "dotplot_markers_by_mac_identity_geneclustered.png", dpi=300, bbox_inches="tight")
#fig.savefig(FIGDIR_MAC / "dotplot_markers_by_mac_identity_geneclustered.pdf", bbox_inches="tight")
plt.show()


manual_order = [
    # ── Inflammatory Monocytes (classical monocyte) ──
    "F13a1", "Il1b", "Vcan", "Hp", "Plac8", "Sell", "Ccr2", "Gngt2", "Chil3", "Fcgr1",
    # ── Interferon-stimulated (kept together) ──
    "Isg15", "Irf7",
    # ── Recruited — inflammatory effectors (chemokines/alarmins grouped) ──
    "Tnf", "Nos2", "Cxcl2", "Ccl3", "Ccl4", "S100a8", "S100a9",
    # ── Recruited — LAM / scar-associated ──
    "Spp1", "Arg1", "Cd9", "Lgals3", "Fn1", "Itgax", "Trem2", "Gpnmb", "Fabp5", "Lpl", "Ms4a7",
    # ── Resident — complement (grouped) ──
    "C1qa", "C1qb", "C1qc",
    # ── Resident — core program ──
    "Pf4", "Cd68", "Maf", "Mertk", "Folr2", "Mrc1",  "Gas6", "Adgre1", "Selenop", "Apoe", 
    # ── Antigen presentation / MHC-II (grouped) ──
    "Cd74", "H2-Ab1", "H2-Aa", "H2-Eb1",
    # ── Pan-macrophage ──
    "Csf1r",
]

# keep only genes present (safety) and plot
present_clustered = [g for g in manual_order if g in adata_sub.var_names]
missing = [g for g in manual_order if g not in adata_sub.var_names]
if missing: print("not in adata_sub:", missing)

axd = sc.pl.dotplot(
    adata_sub, present_clustered, groupby=ID_COL,
    use_raw=False, standard_scale="var", cmap="Reds", dot_max=1.0,
    figsize=(18, 2), title="", show=False,
)
mainax = axd["mainplot_ax"]
mainax.tick_params(axis="x", labelsize=16); mainax.tick_params(axis="y", labelsize=18)
for lbl in mainax.get_xticklabels(): lbl.set_fontweight("bold")
for lbl in mainax.get_yticklabels(): lbl.set_fontweight("bold")
for key in ("size_legend_ax", "color_legend_ax"):
    lax = axd.get(key)
    if lax is not None:
        lax.tick_params(labelsize=9)
        if lax.get_title(): lax.set_title(lax.get_title(), fontsize=9)

fig = plt.gcf()
fig.savefig(FIGDIR_MAC / "dotplot_markers_by_mac_identity_manual.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "dotplot_markers_by_mac_identity_manual.pdf", bbox_inches="tight")
plt.show()


adata_mac

import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
from adjustText import adjust_text
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)

PUB = dict(
    axis_label_fs = 25, axis_label_fw = "bold",
    border_lw     = 1.6,
    legend_fs     = 25, legend_fw     = "bold", legend_outline = 3,
    point_size    = 40,
)
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

def style_umap_axes(ax, xlabel="UMAP 1", ylabel="UMAP 2", title="Mono./MΦ Sub-clustering"):
    ax.set_xlabel(xlabel, fontsize=PUB["axis_label_fs"], fontweight=PUB["axis_label_fw"])
    ax.set_ylabel(ylabel, fontsize=PUB["axis_label_fs"], fontweight=PUB["axis_label_fw"])
    ax.set_title(title, fontsize=PUB["axis_label_fs"], fontweight=PUB["axis_label_fw"], pad=10)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(PUB["border_lw"])
        ax.spines[side].set_color("black")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")

fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(
    adata_mac, color="mac_identity", 
    ax=ax, show=False, frameon=True, size=PUB["point_size"],
    legend_loc="none",
    legend_fontsize=PUB["legend_fs"],
    legend_fontweight=PUB["legend_fw"],
    legend_fontoutline=PUB["legend_outline"],
    title="",  # Handled by style_umap_axes
)

# ── 1. Compute centroids & create text objects ───────────────────────
min_cells = 30  # ← Skip labeling tiny clusters to avoid clutter
valid_subs = adata_mac.obs["mac_identity"].value_counts()[lambda x: x >= min_cells].index

texts = []
for sub in valid_subs:
    mask = adata_mac.obs["mac_identity"] == sub
    x = adata_mac.obsm["X_umap"][mask, 0].mean()
    y = adata_mac.obsm["X_umap"][mask, 1].mean()
    
    txt = ax.text(x, y, sub, fontsize=18, fontweight="bold", ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.85),
                  zorder=10)
    texts.append(txt)

# ── 2. De-collide labels ─────────────────────────────────────────────
adjust_text(
    texts, ax=ax,
    force_text=(0.25, 0.25),    # How much labels push away from each other
    force_pull=(0.95, 0.95),    # How strongly they stick to their original centroid
    expand_points=(1.1, 1.1),   # Expand label bounding box before pushing
    max_iter=30,
    arrowprops=dict(arrowstyle="-", lw=1.2, color="black", alpha=0.5),
)

style_umap_axes(ax)
plt.tight_layout()
fig.savefig(FIGDIR_MAC / "umap_macrophage_subtypes_labeled.png", dpi=600,
            bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "umap_macrophage_subtypes_labeled.pdf",
            bbox_inches="tight", facecolor="white")
plt.show()


import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
from adjustText import adjust_text          # <- uncommented; was causing NameError
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)

PUB = dict(
    axis_label_fs = 28, axis_label_fw = "bold",
    border_lw     = 1.6,
    legend_fs     = 20, legend_fw     = "bold", legend_outline = 3,
    point_size    = 40,
)
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

def style_umap_axes(ax, xlabel="UMAP 1", ylabel="UMAP 2", title=""):
    ax.set_xlabel(xlabel, fontsize=PUB["axis_label_fs"], fontweight=PUB["axis_label_fw"])
    ax.set_ylabel(ylabel, fontsize=PUB["axis_label_fs"], fontweight=PUB["axis_label_fw"])
    ax.set_title(title)
    for side in ("top", "right", "bottom", "left"):     # full box on all sides
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(PUB["border_lw"])
        ax.spines[side].set_color("black")
    ax.set_xticks([]); ax.set_yticks([])                 # UMAP axes are unitless
    ax.set_aspect("equal", adjustable="datalim")

fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(
    adata_mac, color="macrophage_subtypes", palette=mac_colors,
    ax=ax, show=False, frameon=True, size=PUB["point_size"],
    legend_loc="on data",
    legend_fontsize=PUB["legend_fs"],
    legend_fontweight=PUB["legend_fw"],
    legend_fontoutline=PUB["legend_outline"],
    title="",
)

# gentle de-collision: strong pull back to the centroid, small max movement
#adjust_text(
#    list(ax.texts), ax=ax,
#    force_text=(0.12, 0.15),     # weak label–label push
#    force_pull=(0.9, 0.9),       # strong tether to original cluster position
#    expand=(1.02, 1.02),
#    max_move=4,                  # cap drift -> labels barely move
#    min_arrow_len=5,             # suppress tiny leader lines
#    only_move={"text": "xy", "static": "xy", "explode": "xy"},
#    arrowprops=dict(arrowstyle="-", color="black", lw=1.5, alpha=1),
#)
# ==============================================================================
# 1. Manual Label Adjustments & Adding Semi-Transparent Background Boxes
# ==============================================================================
TARGET = "MΦ2-Rep"
yspan  = adata_mac.obsm["X_umap"][:, 1]
DY     = -0.06 * (yspan.max() - yspan.min())  # ~6% of the UMAP y-range

# Define the background box style
# boxstyle="round,pad=0.2" creates smooth corners with a bit of padding around the text
bbox_props = dict(
    boxstyle="round,pad=0.2", 
    facecolor="white", 
    edgecolor="none",       # "none" for borderless, or change to "black" if desired
    alpha=0.65              # Transparency: 0 (invisible) to 1 (opaque). Tweak to taste!
)

moved = False
for t in ax.texts:
    # A. Apply the white background box to every label
    t.set_bbox(bbox_props)
    
    # B. Apply your specific manual shift for the target label
    if t.get_text().strip() == TARGET:
        x, y = t.get_position()
        t.set_position((x, y + DY))
        moved = True

if not moved:
    print("Label not found. On-data labels present:", [t.get_text() for t in ax.texts])


style_umap_axes(ax)
plt.tight_layout()
fig.savefig(FIGDIR_MAC / "umap_macrophage_subtypes_onlabel.png", dpi=600,
            bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "umap_macrophage_subtypes_onlabel.pdf",
            bbox_inches="tight", facecolor="white")
plt.show()


import matplotlib.pyplot as plt

cats = list(adata_mac.obs["macrophage_subtypes"].astype("category").cat.categories)

LEGEND_ORDER = ["MΦ-Act", "MΦ-Inf", "LAM-I", "MΦ-IFN/AS DCs",
                "LAM-II", "Inf. Mono.", "Early MDM", "MΦ-Res/Rep"]
missing = [l for l in LEGEND_ORDER if l not in cats]
if missing:
    print("Not found in categories (check exact spelling):", missing, "\nAvailable:", cats)
labels = [l for l in LEGEND_ORDER if l in cats] + [c for c in cats if c not in LEGEND_ORDER]

DOT_X, TEXT_X = 0.06, 0.13      # gap = TEXT_X - DOT_X
ROW_FACTOR    = 0.30            # smaller → tighter rows
n = len(labels)

fig, ax = plt.subplots(figsize=(3.0, ROW_FACTOR * n + 0.15))
ax.set_xlim(0, 1); ax.set_ylim(-0.5, n - 0.5); ax.axis("off")
for i, l in enumerate(labels):
    y = n - 1 - i
    ax.scatter(DOT_X, y, s=180, color=mac_colors.get(l, "#999999"),
               edgecolors="none", clip_on=False, zorder=3)
    ax.text(TEXT_X, y, l, va="center", ha="left", fontsize=14, fontweight="bold")

fig.savefig(FIGDIR_MAC / "umap_macrophage_subtypes_legend.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "umap_macrophage_subtypes_legend.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()


 import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
PUB = globals().get("PUB", dict(axis_label_fs=28, axis_label_fw="bold", border_lw=1.6,
                                legend_fs=20, legend_fw="bold"))
TYPE_PAL = {"Sham": "#2980B9", "Burn": "#C0392B"}

SPLIT_COL, SUB_COL = "Type_Timepoint_C", "macrophage_subtypes"
obs = adata_mac.obs

# ── x-axis order: Sham D7, Burn D7, Sham D10, Burn D10, … (interleaved) ─────────
present = set(obs[SPLIT_COL].astype(str))
order   = [f"{ty} {d}" for d in ["D7", "D10", "D14", "D19"] for ty in ["Sham", "Burn"]
           if f"{ty} {d}" in present]

# ── subtypes in palette order ──────────────────────────────────────────────────
present_subs = set(obs[SUB_COL].astype(str).unique())
subtypes = [s for s in mac_colors if s in present_subs] + \
           [s for s in sorted(present_subs) if s not in mac_colors]

# ── proportions (each bar sums to 100%) + per-group n ──────────────────────────
prop  = pd.crosstab(obs[SPLIT_COL].astype(str), obs[SUB_COL].astype(str),
                    normalize="index").reindex(order)[subtypes]
n_per = obs[SPLIT_COL].astype(str).value_counts().reindex(order)
print(prop.round(3))

# ── stacked bar figure ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, 1.15 * len(order) + 3), 7))
x, bottom = np.arange(len(order)), np.zeros(len(order))
for s in subtypes:
    ax.bar(x, prop[s].values, bottom=bottom, width=0.82,
           color=mac_colors.get(s, "#999999"), edgecolor="white", linewidth=0.6)
    bottom += prop[s].values

# per-group cell counts above each bar
#for xi, tt in zip(x, order):
#    ax.text(xi, 1.012, f"n={int(n_per[tt])}", ha="center", va="bottom",
#            fontsize=11, color="0.25")

# ── styling ─────────────────────────────────────────────────────────────────────
ax.set_ylim(0, 1); ax.set_xlim(-0.6, len(order) - 0.4)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylabel("Proportion of cells", fontsize=PUB["axis_label_fs"], fontweight=PUB["axis_label_fw"])
ax.set_xlabel("")
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=30, fontweight="bold")
for lab in ax.get_xticklabels():
    lab.set_color(TYPE_PAL["Burn"] if lab.get_text().startswith("Burn") else TYPE_PAL["Sham"])
ax.tick_params(axis="y", labelsize=24)
for lab in ax.get_yticklabels(): lab.set_fontweight("bold")
for side in ("left", "bottom"):
    ax.spines[side].set_visible(True); ax.spines[side].set_linewidth(PUB["border_lw"]); ax.spines[side].set_color("black")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# legend (subtypes), outside right
handles = [Patch(facecolor=mac_colors.get(s, "#999999"), edgecolor="white", label=s) for s in subtypes]
ax.legend(handles=handles, title="Macrophage subtype", bbox_to_anchor=(1.01, 1.0),
          loc="upper left", frameon=False, fontsize=13, title_fontsize=14)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "proportions_macrophage_subtypes_by_type_timepoint.png",
            dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "proportions_macrophage_subtypes_by_type_timepoint.pdf",
            bbox_inches="tight", facecolor="white")
plt.show()


import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
PUB = globals().get("PUB", dict(axis_label_fs=28, axis_label_fw="bold", border_lw=1.6))
TYPE_PAL = {"Sham": "#2980B9", "Burn": "#C0392B"}

SPLIT_COL, SUB_COL = "Type_Timepoint_C", "macrophage_subtypes"
obs = adata_mac.obs

# ── proportions of each subtype within every Type × Timepoint group ────────────
present = set(obs[SPLIT_COL].astype(str))
tp_order = [d for d in ["D7", "D10", "D14", "D19"]
            if any(f"{ty} {d}" in present for ty in ["Sham", "Burn"])]
tp_day   = {d: int(d[1:]) for d in tp_order}

present_subs = set(obs[SUB_COL].astype(str).unique())
subtypes = [s for s in mac_colors if s in present_subs] + \
           [s for s in sorted(present_subs) if s not in mac_colors]

ct = pd.crosstab(obs[SPLIT_COL].astype(str), obs[SUB_COL].astype(str),
                 normalize="index")[subtypes]

def _prop_by_tp(ty):
    rows = [f"{ty} {d}" for d in tp_order if f"{ty} {d}" in ct.index]
    df = ct.reindex(rows); df.index = [r.split()[1] for r in rows]
    return df

# ── figure: Sham | Burn, subtype-colored trajectories over time ────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
for ax, ty in zip(axes, ["Sham", "Burn"]):
    df = _prop_by_tp(ty)
    xd = [tp_day[d] for d in df.index]
    for s in subtypes:
        ax.plot(xd, df[s].values, marker="o", ms=9, lw=6.4,
                color=mac_colors.get(s, "#999999"), markeredgecolor="white",
                markeredgewidth=1.0, label=s, clip_on=False, zorder=3)
    ax.set_title(ty, fontsize=30, fontweight="bold", color=TYPE_PAL[ty], pad=10)
    ax.set_xlabel("", fontsize=PUB["axis_label_fs"], fontweight="bold")
    ax.set_xticks([tp_day[d] for d in tp_order]); ax.set_xticklabels(tp_order, fontsize=26, fontweight="bold")
    ax.set_xlim(min(tp_day.values()) - 0.6, max(tp_day.values()) + 0.6)
    ax.grid(axis="y", color="0.85", lw=0.8, zorder=0)
    ax.tick_params(axis="y", labelsize=25)
    for lab in ax.get_yticklabels(): lab.set_fontweight("bold")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True); ax.spines[side].set_linewidth(PUB["border_lw"]); ax.spines[side].set_color("black")
    for side in ("top", "right"): ax.spines[side].set_visible(False)

axes[0].set_ylabel("Proportion of cells", fontsize=PUB["axis_label_fs"], fontweight="bold")
axes[0].set_ylim(0, None); axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))

# shared subtype legend, outside right
#axes[1].legend(title="Macrophage subtype", bbox_to_anchor=(1.02, 1.0), loc="upper left",
#               frameon=False, fontsize=13, title_fontsize=14)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "proportions_lines_macrophage_subtypes_over_time.png",
            dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "proportions_lines_macrophage_subtypes_over_time.pdf",
            bbox_inches="tight", facecolor="white")
plt.show()


import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from pathlib import Path
import math
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
TYPE_PAL = {"Sham": "#2980B9", "Burn": "#C0392B"}

SPLIT_COL, SUB_COL = "Type_Timepoint_C", "macrophage_subtypes"
obs = adata_mac.obs

present  = set(obs[SPLIT_COL].astype(str))
tp_order = [d for d in ["D7", "D10", "D14", "D19"]
            if any(f"{ty} {d}" in present for ty in ["Sham", "Burn"])]
present_subs = set(obs[SUB_COL].astype(str).unique())
subtypes = [s for s in mac_colors if s in present_subs] + \
           [s for s in sorted(present_subs) if s not in mac_colors]

# ── per-SAMPLE subtype proportions (pseudobulk) ───────────────────────────────
SAMP_CANDS = ["Sample", "sample", "orig.ident", "orig_ident", "SampleID", "sample_id",
              "library", "Library", "mouse", "Mouse", "replicate", "Replicate", "batch"]
samp_col = next((c for c in SAMP_CANDS if c in obs.columns), None)
assert samp_col, f"Set sample col manually from: {list(obs.columns)}"

sp   = obs[SPLIT_COL].astype(str)
_d   = pd.DataFrame({"sample": obs[samp_col].astype(str).values,
                     "Type":   sp.str.split().str[0].values,
                     "Timepoint": sp.str.split().str[1].values,
                     "sub":    obs[SUB_COL].astype(str).values})
_d   = _d[_d["Timepoint"].isin(tp_order)]
_cnt = _d.groupby(["sample", "Type", "Timepoint", "sub"]).size().unstack("sub", fill_value=0)
for s in subtypes:
    if s not in _cnt.columns: _cnt[s] = 0
_prop = _cnt[subtypes].div(_cnt[subtypes].sum(1), axis=0).reset_index()
print(_prop.groupby(["Type", "Timepoint"]).size().rename("n_samples").reset_index().to_string(index=False))

# ── linear model per subtype: prop ~ Type*Timepoint → Burn-Sham contrast per tp ─
BURN = "C(Type, Treatment('Sham'))[T.Burn]"
def _per_tp_ttest(d):
    out = {}
    for tp in tp_order:
        dd = d[d["Timepoint"] == tp]
        b = dd.loc[dd.Type == "Burn", "y"]; s = dd.loc[dd.Type == "Sham", "y"]
        out[tp] = (b.mean() - s.mean(), ttest_ind(b, s, equal_var=False).pvalue) \
                  if len(b) >= 2 and len(s) >= 2 else (np.nan, np.nan)
    return out

def _lm_contrasts(sub):
    d = _prop[["Type", "Timepoint"]].copy(); d["y"] = _prop[sub].values
    d = d.dropna(subset=["y"])
    if d["Type"].nunique() < 2:
        return {tp: (np.nan, np.nan) for tp in tp_order}
    ref = tp_order[0]
    try:
        res = smf.ols(f"y ~ C(Type, Treatment('Sham'))*C(Timepoint, Treatment('{ref}'))", d).fit()
        if res.df_resid < 1:
            return _per_tp_ttest(d)
        names = res.params.index.tolist(); out = {}
        for tp in tp_order:
            c = np.zeros(len(names)); hit = False
            for i, nm in enumerate(names):
                if nm == BURN: c[i] = 1.0; hit = True
                elif nm.startswith(BURN + ":") and nm.endswith(f"[T.{tp}]"): c[i] = 1.0
            if not hit:
                out[tp] = (np.nan, np.nan); continue
            tt = res.t_test(c)
            out[tp] = (float(np.ravel(tt.effect)[0]), float(np.ravel(tt.pvalue)[0]))
        return out
    except Exception:
        return _per_tp_ttest(d)

lm    = {s: _lm_contrasts(s) for s in subtypes}
cells = [(s, tp) for s in subtypes for tp in tp_order]
raw_p = {k: lm[k[0]][k[1]][1] for k in cells}
_keys = [k for k in cells if np.isfinite(raw_p[k])]
padj  = {k: np.nan for k in cells}
if _keys:
    padj.update(dict(zip(_keys, multipletests([raw_p[k] for k in _keys], method="fdr_bh")[1])))

def _stars(p):
    if not np.isfinite(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

print("\nBurn vs Sham (linear model) — FDR-adjusted p per subtype × timepoint:")
for s in subtypes:
    print(" ", s, {tp: (f"{padj[(s,tp)]:.2g}" if np.isfinite(padj[(s,tp)]) else "na") for tp in tp_order})

# ── figure: one small panel per subtype, Burn vs Sham, points + SEM + stars ────
psm   = _prop.melt(id_vars=["sample", "Type", "Timepoint"], value_vars=subtypes,
                   var_name="sub", value_name="prop")
DODGE = {"Sham": -0.13, "Burn": 0.13}
rng   = np.random.default_rng(0)
xidx  = {d: i for i, d in enumerate(tp_order)}

ncol = min(4, len(subtypes)); nrow = math.ceil(len(subtypes) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.3 * nrow), squeeze=False)

for k, s in enumerate(subtypes):
    ax = axes[k // ncol][k % ncol]
    d_s = psm[psm["sub"] == s]
    for cond in ["Sham", "Burn"]:
        dc = d_s[d_s["Type"] == cond]
        xs = dc["Timepoint"].map(xidx).values + DODGE[cond] + rng.uniform(-0.04, 0.04, len(dc))
        ax.scatter(xs, dc["prop"], s=34, color=TYPE_PAL[cond], alpha=0.55,
                   edgecolors="black", linewidths=0.5, zorder=2)
        mx, my, me = [], [], []
        for tp in tp_order:
            v = dc.loc[dc["Timepoint"] == tp, "prop"].values
            if len(v):
                mx.append(xidx[tp] + DODGE[cond]); my.append(v.mean())
                me.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.errorbar(mx, my, yerr=me, fmt="-o", color=TYPE_PAL[cond], lw=2.4, ms=8,
                    mec="black", mew=1.0, capsize=4, elinewidth=1.8, zorder=3, label=cond)

    ymax = float(np.nan_to_num(d_s["prop"].max()))
    for tp in tp_order:
        txt = _stars(padj.get((s, tp), np.nan))
        if txt:
            ax.text(xidx[tp], ymax * 1.06 + 0.01, txt, ha="center", va="bottom",
                    fontsize=16, fontweight="bold", color="black")
    ax.set_title(s, fontsize=14, fontweight="bold", color=mac_colors.get(s, "black"))
    ax.set_xticks(range(len(tp_order))); ax.set_xticklabels(tp_order, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.4, len(tp_order) - 0.6)
    ax.set_ylim(0, max(ymax * 1.32, 0.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="y", labelsize=12)
    for lab in ax.get_yticklabels(): lab.set_fontweight("bold")
    if k % ncol == 0:
        ax.set_ylabel("Proportion", fontsize=15, fontweight="bold")
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_linewidth(1.4)

for k in range(len(subtypes), nrow * ncol):
    axes[k // ncol][k % ncol].set_visible(False)

handles = [Line2D([0], [0], color=TYPE_PAL[c], marker="o", lw=3, ms=9, mec="black", label=c)
           for c in ["Sham", "Burn"]]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=15,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(FIGDIR_MAC / "proportions_by_subtype_burn_vs_sham_stats.png",
            dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "proportions_by_subtype_burn_vs_sham_stats.pdf",
            bbox_inches="tight", facecolor="white")
plt.show()


import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

SPLIT_COL, SUB_COL = "Type_Timepoint_C", "macrophage_subtypes"
obs = adata_mac.obs
present  = set(obs[SPLIT_COL].astype(str))
tp_order = [d for d in ["D7", "D10", "D14", "D19"]
            if any(f"{ty} {d}" in present for ty in ["Sham", "Burn"])]
present_subs = set(obs[SUB_COL].astype(str).unique())
subtypes = [s for s in mac_colors if s in present_subs] + \
           [s for s in sorted(present_subs) if s not in mac_colors]

# ── per-sample proportions ────────────────────────────────────────────────────
SAMP_CANDS = ["Sample", "sample", "orig.ident", "orig_ident", "SampleID", "sample_id",
              "library", "Library", "mouse", "Mouse", "replicate", "Replicate", "batch"]
samp_col = next((c for c in SAMP_CANDS if c in obs.columns), None)
assert samp_col, f"Set sample col manually from: {list(obs.columns)}"
sp   = obs[SPLIT_COL].astype(str)
_d   = pd.DataFrame({"sample": obs[samp_col].astype(str).values,
                     "Type":   sp.str.split().str[0].values,
                     "Timepoint": sp.str.split().str[1].values,
                     "sub":    obs[SUB_COL].astype(str).values})
_d   = _d[_d["Timepoint"].isin(tp_order)]
_cnt = _d.groupby(["sample", "Type", "Timepoint", "sub"]).size().unstack("sub", fill_value=0)
for s in subtypes:
    if s not in _cnt.columns: _cnt[s] = 0
_prop = _cnt[subtypes].div(_cnt[subtypes].sum(1), axis=0).reset_index()

# ── LM per subtype: Burn-Sham contrast per timepoint (fallback = Welch t-test) ─
BURN = "C(Type, Treatment('Sham'))[T.Burn]"
def _tt(d):
    o = {}
    for tp in tp_order:
        dd = d[d.Timepoint == tp]; b = dd.loc[dd.Type=="Burn","y"]; s = dd.loc[dd.Type=="Sham","y"]
        o[tp] = ttest_ind(b, s, equal_var=False).pvalue if len(b) >= 2 and len(s) >= 2 else np.nan
    return o
def _lm(sub):
    d = _prop[["Type","Timepoint"]].copy(); d["y"] = _prop[sub].values
    d = d.dropna(subset=["y"])
    if d["Type"].nunique() < 2: return {tp: np.nan for tp in tp_order}
    ref = tp_order[0]
    try:
        res = smf.ols(f"y ~ C(Type, Treatment('Sham'))*C(Timepoint, Treatment('{ref}'))", d).fit()
        if res.df_resid < 1: return _tt(d)
        names = res.params.index.tolist(); o = {}
        for tp in tp_order:
            c = np.zeros(len(names)); hit = False
            for i, nm in enumerate(names):
                if nm == BURN: c[i] = 1.0; hit = True
                elif nm.startswith(BURN + ":") and nm.endswith(f"[T.{tp}]"): c[i] = 1.0
            o[tp] = float(np.ravel(res.t_test(c).pvalue)[0]) if hit else np.nan
        return o
    except Exception:
        return _tt(d)
pv    = {(s, tp): _lm(s)[tp] for s in subtypes for tp in tp_order}
_keys = [k for k in pv if np.isfinite(pv[k])]
padj  = {k: np.nan for k in pv}
if _keys:
    padj.update(dict(zip(_keys, multipletests([pv[k] for k in _keys], method="fdr_bh")[1])))
def _stars(p): return "" if not np.isfinite(p) else ("***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "")

# ── effect matrix: log2 fold-change of mean proportion (Burn / Sham) ──────────
EPS = 5e-3
eff = np.full((len(subtypes), len(tp_order)), np.nan)
for i, s in enumerate(subtypes):
    for j, tp in enumerate(tp_order):
        d = _prop[_prop.Timepoint == tp]
        b, sh = d.loc[d.Type=="Burn", s], d.loc[d.Type=="Sham", s]
        if len(b) and len(sh):
            eff[i, j] = np.log2((b.mean() + EPS) / (sh.mean() + EPS))

order   = np.argsort(-np.nanmean(eff, axis=1))          # burn-enriched subtypes on top
subs_o  = [subtypes[i] for i in order]; eff_o = eff[order]
vmax    = float(np.nanmax(np.abs(eff)))

# ── heatmap ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(0.85 * len(tp_order) + 3.0, 0.55 * len(subtypes) + 1.4))
im = ax.imshow(eff_o, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
for r, s in enumerate(subs_o):
    for c, tp in enumerate(tp_order):
        txt = _stars(padj.get((s, tp), np.nan))
        if txt:
            col = "white" if abs(eff_o[r, c]) > 0.6 * vmax else "black"
            ax.text(c, r, txt, ha="center", va="center", fontsize=15, fontweight="bold", color=col)

ax.set_xticks(range(len(tp_order))); ax.set_xticklabels(tp_order, fontsize=14, fontweight="bold")
ax.set_yticks(range(len(subs_o)));  ax.set_yticklabels(subs_o, fontsize=13, fontweight="bold")
ax.set_xticks(np.arange(-.5, len(tp_order), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(subs_o), 1), minor=True)
ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
for sp_ in ax.spines.values(): sp_.set_visible(False)
ax.set_title("Subtype shift: Burn vs Sham\n(* FDR < 0.05)", fontsize=14, fontweight="bold", pad=8)

cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cbar.set_label("log$_2$(Burn / Sham)", fontsize=12, fontweight="bold")
cbar.ax.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "subtype_burn_vs_sham_heatmap.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "subtype_burn_vs_sham_heatmap.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()


import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

SPLIT_COL, SUB_COL = "Type_Timepoint_C", "macrophage_subtypes"
obs = adata_mac.obs
present  = set(obs[SPLIT_COL].astype(str))
tp_order = [d for d in ["D7", "D10", "D14", "D19"]
            if any(f"{ty} {d}" in present for ty in ["Sham", "Burn"])]
present_subs = set(obs[SUB_COL].astype(str).unique())
subtypes = [s for s in mac_colors if s in present_subs] + \
           [s for s in sorted(present_subs) if s not in mac_colors]

# ── per-sample proportions ────────────────────────────────────────────────────
SAMP_CANDS = ["Sample", "sample", "orig.ident", "orig_ident", "SampleID", "sample_id",
              "library", "Library", "mouse", "Mouse", "replicate", "Replicate", "batch"]
samp_col = next((c for c in SAMP_CANDS if c in obs.columns), None)
assert samp_col, f"Set sample col manually from: {list(obs.columns)}"
sp   = obs[SPLIT_COL].astype(str)
_d   = pd.DataFrame({"sample": obs[samp_col].astype(str).values,
                     "Type":   sp.str.split().str[0].values,
                     "Timepoint": sp.str.split().str[1].values,
                     "sub":    obs[SUB_COL].astype(str).values})
_d   = _d[_d["Timepoint"].isin(tp_order)]
_cnt = _d.groupby(["sample", "Type", "Timepoint", "sub"]).size().unstack("sub", fill_value=0)
for s in subtypes:
    if s not in _cnt.columns: _cnt[s] = 0
_prop = _cnt[subtypes].div(_cnt[subtypes].sum(1), axis=0).reset_index()
print(_prop.groupby(["Type", "Timepoint"]).size().rename("n_samples").reset_index().to_string(index=False))

def _stars(p): return "" if not np.isfinite(p) else ("***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "")

# ── OVERALL Burn-vs-Sham per subtype (pool timepoints, control for timepoint) ──
def _overall_p(sub):
    d = _prop[["Type", "Timepoint"]].copy(); d["y"] = _prop[sub].values
    d = d.dropna(subset=["y"])
    if d["Type"].nunique() < 2 or len(d) < 4: return np.nan
    try:
        res = smf.ols("y ~ C(Type, Treatment('Sham')) + C(Timepoint)", d).fit()
        key = [k for k in res.pvalues.index if k.startswith("C(Type")]
        return float(res.pvalues[key[0]]) if key else np.nan
    except Exception:
        return np.nan
op   = {s: _overall_p(s) for s in subtypes}
_ok  = [s for s in subtypes if np.isfinite(op[s])]
opadj = {s: np.nan for s in subtypes}
if _ok:
    opadj.update(dict(zip(_ok, multipletests([op[s] for s in _ok], method="fdr_bh")[1])))
print("\nOverall Burn vs Sham per subtype (FDR):")
for s in subtypes:
    print(f"  {s:16s} p={op[s]:.2g}  padj={opadj[s]:.2g}" if np.isfinite(op[s]) else f"  {s:16s} na")

# ── effect matrix: log2 fold-change of mean proportion (Burn / Sham) ──────────
EPS = 5e-3
eff = np.full((len(subtypes), len(tp_order)), np.nan)
for i, s in enumerate(subtypes):
    for j, tp in enumerate(tp_order):
        d = _prop[_prop.Timepoint == tp]
        b, sh = d.loc[d.Type=="Burn", s], d.loc[d.Type=="Sham", s]
        if len(b) and len(sh):
            eff[i, j] = np.log2((b.mean() + EPS) / (sh.mean() + EPS))

order  = np.argsort(-np.nanmean(eff, axis=1))
subs_o = [subtypes[i] for i in order]; eff_o = eff[order]
vmax   = float(np.nanmax(np.abs(eff)))
ylabels = [f"{s}  {_stars(opadj[s])}".rstrip() for s in subs_o]     # append overall star

# ── heatmap ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(0.85 * len(tp_order) + 3.2, 0.55 * len(subtypes) + 1.4))
im = ax.imshow(eff_o, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(tp_order))); ax.set_xticklabels(tp_order, fontsize=14, fontweight="bold")
ax.set_yticks(range(len(subs_o)));  ax.set_yticklabels(ylabels, fontsize=13, fontweight="bold")
ax.set_xticks(np.arange(-.5, len(tp_order), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(subs_o), 1), minor=True)
ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
for sp_ in ax.spines.values(): sp_.set_visible(False)
ax.set_title("Subtype shift: Burn vs Sham\n(* = overall FDR < 0.05)", fontsize=14, fontweight="bold", pad=8)

cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cbar.set_label("log$_2$(Burn / Sham)", fontsize=12, fontweight="bold")
cbar.ax.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "subtype_burn_vs_sham_heatmap.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "subtype_burn_vs_sham_heatmap.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()

