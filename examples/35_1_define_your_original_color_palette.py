"""1. Define your original color palette

Source: macrophages_resident_recruited.ipynb
Libraries: adjustText, matplotlib, numpy, os, pandas, re, scanpy
Key calls: adjust_text, def add_ct_labels, def tp_per_cell, dotplot, plt.gcf, plt.show, plt.subplots, sc.pl.dotplot, sc.pl.umap, sc.tl.score_genes, scatter, umap
"""

# 1. Define your original color palette
major_cell_colors = {
    "Krt":     "#FF7F00",
    "Fibs":       "#33A02C",
    "Neu":       "#E31A1C",
    "MΦ":       "#1F78B4",
    "Mono.":  "#E6AB02",
    "cDCs":   "#6A3D9A",
    "Endo.": "#FB9A99",
    "T Cells":           "#A6CEE3",
    "Smcs":      "#B2DF8A",
    "Seb":         "#720946",
    #"Adipocytes":        "#CAB2D6"
}

# 2. Define the desired order for plotting/categories
desired_order = [
    "Krt",
    "Fibs",
    "Neu",
    "MΦ",
    "Mono.",
    "cDCs",
    "Endo.",
    "T Cells",
    "Smcs",
    "Seb",
    #"Adipocytes"
]

# 3. Map your clusters based on the DEG analysis
cluster_to_celltype = {
    "0":  "Neu",        # S100a8/9, Cxcl2, Ccl3, Acod1, Il1b — see caveat below
    "1":  "Smcs",       # Ttn, Neb, Ryr1, Actn3 — SKELETAL muscle
    "2":  "Fibs",       # Col1a1/2, Col3a1, Dcn, Mfap5, Dpt
    "3":  "cDCs",       # Ciita, H2-Aa/Ab1/Eb1, Cd74 (cDC/APC)
    "4":  "Neu",        # Acod1, Hdc, Ccl3/4, Nlrp3, Clec4e, Cd14
    "5":  "Neu",        # Ccl3, Cxcl2, Acod1, Cd274, Hilpda
    "6":  "MΦ",         # C1qb, Mrc1, Stab1, Pf4, Apoe, Csf1r (resident)
    "7":  "Krt",        # Lef1, Msx2, Dlx3, Krt25/28, Mki67 (HF matrix)
    "8":  "T Cells",    # Cd3d/e/g, Cd2, Lck, Zap70, Il2rb
    "9":  "Endo.",      # Pecam1, Cdh5, Kdr, Tie1, Egfl7
    "10": "Krt",        # Krt5/14, Col17a1, Trp63, Lama3 (basal)
    "11": "Krt",        # Krt15/17, Lhx2, Bnc2, Cxcl14 (follicle/stem)
    "12": "Neu",        # Cxcr2, Csf3r, Sell, Hp, Ncf2
    "13": "Krt",        # Krt1/10, Dsg1a/b, Krtdap (suprabasal)
    "14": "Fibs",       # Tnn, Lrrc15, Lox, Tnc, Thbs2 (activated)
    "15": "MΦ",         # Arg1, Nos2, F13a1, Hilpda (recruited/inflammatory)
    "16": "Seb",        # Awat1, Scd1, Far2, Mgll, Krt79, Cidea
    "17": "Mono.",        # Msr1, Fcgr1, Lyz1, S100a4, Arg1 (monocyte-derived)
    "18": "MΦ",         # Adgre1, Mrc1, C1qb, Stab1, Ms4a4a/6c (resident)
    "19": "Smcs",       # Myh11, Rgs5, Pdgfrb, Notch3 — mural/SMOOTH muscle
    "20": "Neu",        # Cxcr2, Csf3r, Mmp9, Trem1, Hp
    "21": "Krt",        # Tchh, Krt71/73/25/27, Padi3 (IRS)
    "22": "Krt",        # Krt31/33/36/83, Krtaps, Foxn1, Hoxc13 (hair shaft)
    "23": "Krt",        # Lor, Casp14, Aloxe3, Klk7, Cnfn (cornified)
}

# 4. Apply mapping, order categories, and assign colors to your AnnData object
# Ensure the cluster column is string type to match the dictionary keys
adata_full.obs["leiden_res0.8"] = adata_full.obs["leiden_res0.8"].astype(str)

adata_full.obs["cell_types_simple"] = (adata_full.obs["leiden_res0.8"]
                                      .map(cluster_to_celltype)
                                      .astype("category"))

# Set the factor levels to your preferred order
adata_full.obs["cell_types_simple"] = adata_full.obs["cell_types_simple"].cat.set_categories(desired_order)

# Map colors strictly matching your desired order
adata_full.uns["cell_types_simple_colors"] = [major_cell_colors[c] for c in desired_order]

# 5. QC checks
print("--- Cell Type Counts ---")
print(adata_full.obs["cell_types_simple"].value_counts())
print("\n--- QC Check ---")
print("Missing from dataset:", set(desired_order) - set(adata_full.obs["cell_types_simple"].dropna().unique()))


adata_full

import numpy as np
import matplotlib.patheffects as pe

# cell-type column for labels
CT_COL = next((c for c in ["cell_types_simple", "cell_types_full", "cell_type", "cell_types"]
               if c in adata_full.obs.columns), None)
assert CT_COL, f"set CT_COL manually from {list(adata_full.obs.columns)}"
print("labeling by:", CT_COL)

xy  = adata_full.obsm["X_umap"]
cts = adata_full.obs[CT_COL].astype(str)
centroids = {ct: np.median(xy[cts.values == ct], axis=0) for ct in cts.unique()}

def add_ct_labels(ax, fs=22):
    for ct, (cx, cy) in centroids.items():
        ax.text(cx, cy, ct, fontsize=fs, fontweight="bold", ha="center", va="center",
                color="black", zorder=10,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# ── rescale each score to 0–2 ─────────────────────────────────────────────────
disp = {}
for score in ["Glycolysis_Score", "OXPHOS_Score"]:
    v = adata_full.obs[score].astype(float).values
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    adata_full.obs[f"{score}_disp"] = (v - vmin) / (vmax - vmin) * 2.0
    disp[score] = f"{score}_disp"

# ── feature plots with cell-type labels on top ────────────────────────────────
panels = [("Glycolysis_Score", "Glycolysis"), ("OXPHOS_Score", "OXPHOS")]
fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6.8))
axes = np.atleast_1d(axes)
for ax, (score, title) in zip(axes, panels):
    sc.pl.umap(adata_full, color=disp[score], cmap="RdBu_r",
               vmin=0, vmax=2, sort_order=True, size=6,
               frameon=True, show=False, ax=ax, colorbar_loc="right")
    add_ct_labels(ax)                                    # <- labels on top
    ax.set_title(title, fontsize=30, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=30, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=30, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for cax in fig.axes:
        if cax.get_label() == "<colorbar>":
            cax.set_yticks([0, 0.5, 1, 1.5, 2])

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "umap_full_glyco_oxphos_scores_0to2_labeled.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "umap_full_glyco_oxphos_scores_0to2_labeled.pdf", bbox_inches="tight")
plt.show()


adata_full

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from adjustText import adjust_text

CT_COL = next((c for c in ["cell_types_simple", "cell_types_full", "cell_type", "cell_types"]
               if c in adata_full.obs.columns), None)
assert CT_COL, f"set CT_COL from {list(adata_full.obs.columns)}"
xy  = adata_full.obsm["X_umap"]
cts = adata_full.obs[CT_COL].astype(str)
centroids = {ct: np.median(xy[cts.values == ct], axis=0) for ct in cts.unique()}

FS_CT = 30
STROKE = [pe.withStroke(linewidth=3, foreground="white")]

# Glycolysis rescaled 0–2 (shared)
v = adata_full.obs["Glycolysis_Score"].astype(float).values
val = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) * 2.0

typ = adata_full.obs["Type"].astype(str).values
CMAP, NORM, NA_GREY = "RdBu_r", Normalize(0, 2), "#E8E8E8"
title_color = {"Sham": "#2471A3", "Burn": "#C0392B"}
conds = ["Sham", "Burn"]

pad  = (xy[:, 0].max() - xy[:, 0].min()) * 0.03
xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)

fig, axes = plt.subplots(1, len(conds), figsize=(7.5 * len(conds), 6.8), sharex=True, sharey=True)
axes = np.atleast_1d(axes)
for ax, ty in zip(axes, conds):
    m = typ == ty
    ax.scatter(xy[:, 0], xy[:, 1], s=5, c=NA_GREY, linewidths=0, rasterized=True)
    order = np.argsort(val[m])
    ax.scatter(xy[m, 0][order], xy[m, 1][order], c=val[m][order], cmap=CMAP, norm=NORM,
               s=7, linewidths=0, rasterized=True)
    ax.set_title(ty, fontsize=30, fontweight="bold", color=title_color[ty], pad=10)
    ax.set_xlabel("UMAP 1", fontsize=28, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=28, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xlim); ax.set_ylim(ylim)

# 1) de-overlap labels on the LEFT panel
texts = [axes[0].text(cx, cy, ct, fontsize=FS_CT, fontweight="bold", ha="center", va="center",
                      color="black", zorder=10, path_effects=STROKE)
         for ct, (cx, cy) in centroids.items()]
adjust_text(texts, ax=axes[0], expand=(1.05, 1.2), force_text=(0.4, 0.6),
            max_move=14, min_arrow_len=5, only_move={"text": "xy"},
            arrowprops=dict(arrowstyle="-", color="black", lw=2.0))

# 2) mirror resolved positions onto the RIGHT panel
final = {t.get_text(): t.get_position() for t in texts}
for ct, (cx, cy) in centroids.items():
    tx, ty2 = final.get(ct, (cx, cy))
    axes[1].annotate(ct, xy=(cx, cy), xytext=(tx, ty2), ha="center", va="center",
                     fontsize=FS_CT, fontweight="bold", color="black", zorder=10,
                     path_effects=STROKE, arrowprops=dict(arrowstyle="-", color="black", lw=1.0))

sm = ScalarMappable(norm=NORM, cmap=CMAP); sm.set_array([])
cbar = fig.colorbar(sm, ax=list(axes), fraction=0.025, pad=0.02)
cbar.set_ticks([0, 0.5, 1, 1.5, 2]); cbar.ax.tick_params(labelsize=20)
cbar.set_label("Glycolysis", fontsize=28, fontweight="bold")

fig.savefig(FIGDIR_MAC / "umap_full_glycolysis_split_by_type.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "umap_full_glycolysis_split_by_type.pdf", bbox_inches="tight")
plt.show()


adata_mac.write_h5ad("macrophages_final_06252026.rds")

CT = next((c for c in ["cell_types_full","cell_types_simple","cell_type"] if c in adata_full.obs.columns), None)
ifn_ligands = ([g for g in ["Ifng","Ifnb1"] if g in adata_full.var_names] +
               [g for g in adata_full.var_names if g.startswith("Ifna")])
print("IFN ligands found:", ifn_ligands)
sc.pl.dotplot(adata_full, ifn_ligands, groupby=CT, use_raw=False, standard_scale=None)


type1 = [g for g in ["Isg15","Ifit1","Ifit3","Irf7","Rsad2","Oasl2","Mx1","Mx2","Usp18","Ifi44","Oasl1"] if g in adata_full.var_names]
type2 = [g for g in ["Gbp2","Gbp3","Gbp5","Ciita","Cxcl9","Cxcl10", "Cxcl12","Irf1","Stat1","Socs1","Nos2"] if g in adata_full.var_names]
sc.tl.score_genes(adata_full, type1, score_name="ISG_typeI",  use_raw=False)
sc.tl.score_genes(adata_full, type2, score_name="ISG_typeII", use_raw=False)
# compare within the IFN-stim monocyte population
mask = adata_full.obs["macrophage_subtypes"].astype(str).str.contains("IFN", na=False)  # adjust to your label
print(adata_full.obs.loc[mask, ["ISG_typeI","ISG_typeII"]].mean())


CT = next(c for c in ["cell_types_full","cell_types_simple"] if c in adata_full.obs.columns)
pdc = [g for g in ["Siglech","Bst2","Tcf4","Irf8","Ccr9","Klk1","Cox6a2"] if g in adata_full.var_names]
sc.pl.dotplot(adata_full, pdc, groupby=CT, standard_scale=None, use_raw=False)


init = [g for g in ["Cgas","Mb21d1","Sting1","Tmem173","Tbk1","Ikbke","Irf3"] if g in adata_full.var_names]
sc.tl.score_genes(adata_full, init, score_name="cGAS_STING_init", use_raw=False)
print(adata_full.obs.groupby(CT)["cGAS_STING_init"].mean().sort_values(ascending=False))

# and is it burn-specific? split the machinery score within myeloid cells
mye = adata_full.obs[CT].isin(["Mono.", "MΦ", "cDCs"])
print(adata_full.obs[mye].groupby([CT, "Type"])["IFN_I_machinery"].mean().unstack())


prod = [g for g in ["Irf7","Irf3","Tbk1","Sting1","Tmem173","Cgas","Mb21d1",
                    "Ddx58","Ifih1","Tlr3","Tlr7","Tlr9"] if g in adata_full.var_names]
sc.tl.score_genes(adata_full, prod, score_name="IFN_I_machinery", use_raw=False)
print(adata_full.obs.groupby(CT)["IFN_I_machinery"].mean().sort_values(ascending=False))


adata_full.write_h5ad("filtered_final_06252026.rds")

import numpy as np

# ── rescale each score to 0–2 for display (no negatives) ──────────────────────
disp = {}
for score in ["Glycolysis_Score", "OXPHOS_Score"]:
    v = adata_full.obs[score].astype(float).values
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    adata_full.obs[f"{score}_disp"] = (v - vmin) / (vmax - vmin) * 2.0
    disp[score] = f"{score}_disp"

# ── feature plots on the full UMAP, 0–2 scale ─────────────────────────────────
panels = [("Glycolysis_Score", "Glycolysis"), ("OXPHOS_Score", "OXPHOS")]
fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6.8))
axes = np.atleast_1d(axes)
for ax, (score, title) in zip(axes, panels):
    sc.pl.umap(adata_full, color=disp[score], cmap="RdBu_r",
               vmin=0, vmax=2, sort_order=True, size=6,
               frameon=True, show=False, ax=ax, colorbar_loc="right")
    ax.set_title(title, fontsize=28, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=24, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=24, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    # clean 0–2 colorbar ticks
    for cax in fig.axes:
        if cax.get_label() == "<colorbar>":
            cax.set_yticks([0, 1, 2])

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "umap_full_glyco_oxphos_scores_0to2.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "umap_full_glyco_oxphos_scores_0to2.pdf", bbox_inches="tight")
plt.show()


import re, pandas as pd, numpy as np, scanpy as sc, matplotlib.pyplot as plt

# ── marker groups, in the desired order ───────────────────────────────────────
GROUPS = {
    "Glycolysis": ["Hk2", "Ldha", "Pgk1", "Pfkp", "Pkm", "Slc2a1"],
    "OXPHOS":     ["Ndufs1", "Sdhb", "Ndufb8", "Sdha", "mt-Co1", "mt-Co2", "mt-Co3", "mt-Nd1"],
    "Hypoxia":    ["Hif1a", "Ldha", "Vegfa", "Sod2", "Aldoa", "Hmox1"],
}
marker_groups = {}
for name, genes in GROUPS.items():
    present = [g for g in genes if g in adata_mac.var_names]
    missing = [g for g in genes if g not in adata_mac.var_names]
    if missing:
        print(f"{name}: missing {missing}")
    marker_groups[name] = present

# ── Type × Timepoint groupby (Sham block first, then Burn) ─────────────────────
def tp_per_cell(obs):
    for c in obs.columns:
        if "time" in c.lower(): return obs[c].astype(str)
    for c in obs.columns:
        if obs[c].astype(str).str.fullmatch(r"D?\d+").mean() > 0.8: return obs[c].astype(str)
    for c in obs.columns:
        ext = obs[c].astype(str).str.extract(r"(D\d+)")[0]
        if ext.notna().mean() > 0.8: return ext
    raise ValueError("no timepoint column")

tp    = tp_per_cell(adata_mac.obs).astype(str).str.extract(r"(\d+)")[0].radd("D")
combo = pd.Series(adata_mac.obs["Type"].astype(str).values + " " + tp.values, index=adata_mac.obs_names)
tps   = sorted(pd.unique(tp.dropna()), key=lambda t: int(re.search(r"\d+", t).group()))
order = [f"{t} {d}" for t in ["Sham", "Burn"] for d in tps if f"{t} {d}" in set(combo)]
adata_mac.obs["Type_TP"] = pd.Categorical(combo, categories=order, ordered=True)
print("rows:", order)

ncols = sum(len(v) for v in marker_groups.values())
axd = sc.pl.dotplot(
    adata_mac, marker_groups, groupby="Type_TP",
    standard_scale="var", cmap="Reds", dot_max=1.0, use_raw=False,
    var_group_rotation=0,                       # horizontal group labels
    figsize=(9, 3),                             # a touch taller -> legend has room
    show=False,
)

mainax = axd["mainplot_ax"]
mainax.tick_params(axis="x", labelsize=14)
mainax.tick_params(axis="y", labelsize=15)
for lbl in mainax.get_xticklabels(): lbl.set_fontweight("bold")
for lbl in mainax.get_yticklabels(): lbl.set_fontweight("bold")

# group bracket labels (Glycolysis / OXPHOS / Hypoxia): horizontal + smaller
gax = axd.get("gene_group_ax")
if gax is not None:
    for t in gax.texts:
        t.set_rotation(0); t.set_ha("center")
        t.set_fontsize(15); t.set_fontweight("bold")


# shrink the size legend + colorbar so they aren't squished
for key in ("size_legend_ax", "color_legend_ax"):
    lax = axd.get(key)
    if lax is not None:
        lax.tick_params(labelsize=8)
        if lax.get_title():
            lax.set_title(lax.get_title(), fontsize=12)

fig = plt.gcf()
fig.savefig( "dotplot_metabolic_programs_glyco_oxphos_hypoxia.png", dpi=600, bbox_inches="tight")
fig.savefig("dotplot_metabolic_programs_glyco_oxphos_hypoxia.pdf", bbox_inches="tight")
plt.show()



mac_subset_ids = {
    "0": "MΦ1-Act",   # Nos2-hi, Arg1-hi, Ptges, Ptgs2 — inflammatory M1
    "1": "MΦ2-Res/Rep",   # C1qa/b/c, Mrc1-hi, Mertk, Adgre1, Gas6; Slc40a1/Spic = iron-recycling resident
    "2": "MΦ1-Inf",    # Arg1, Vegfa, Hilpda, Pdpn — hypoxic/angiogenic, not inflammatory
    "3": "MDM",       # Ciita, H2-Ab1/Aa/Eb1, Cd74; lipid-program-low (Msr1/Abca1 deep-neg)
    "4": "MΦ2-Res/Rep",   # Stab1, Maf, Hpgds, C4b, Apoe, Mrc1-hi — reparative
    "5": "LAM-I",       # Spp1, Lgals3, Ctsb/l/s, Fth1-hi, Hmox1-hi, Grn, Cd68-peak
    "6": "Inf. Mono.",       # Ccr2, Vcan, Cd80, Tlr2, Osm, Olr1+ — recruited monocyte-derived
    "7": "LAM-II",    # PPP/redox (H6pd, Tkt, Prdx1, Txn1), Smad7/Igf2r-hi
    "8": "MΦ1-Act",     # Cd36, Msr1, Abca1, Plin2, Spp1, Trem2, Gpnmb, Hmox1-hi, Fth1-hi
    "9": "MΦ1-IFN",   # Irf7, Rsad2, Isg15, Stat1/2, Ifit3 — interferon
}

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc

mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial'})
sc.set_figure_params(dpi=200, dpi_save=600)

# ── row order: Sham D7→D19, then Burn D7→D19 ──────────────────────────────────
present = set(adata_mac.obs["Type_Timepoint_C"].astype(str))
tt_order = [f"{ty} {d}" for ty in ["Sham", "Burn"] for d in ["D7", "D10", "D14", "D19"]
            if f"{ty} {d}" in present]
adata_mac.obs["Type_Timepoint_C"] = pd.Categorical(
    adata_mac.obs["Type_Timepoint_C"].astype(str), categories=tt_order, ordered=True)
print("rows:", tt_order)

# ── gene panels (Davies et al. 2013 M1/M2 framework + burn-vs-sham volcano hits) ──
# M1 / classically-activated / inflammatory  (Davies: Nos2/Tnf/Il1b/Cd86; Ccr2 = Ly6Cʰⁱ mono recruit)
# M1 / inflammatory — grouped by function (Nos2+Arg1 adjacent)
m1_genes = [
    'Arg1',     'Nos2',                      # NO/M1 polarization — dual activation (adjacent)
    'Il1b', 'Tnf', 'Cd86',                   # pro-inflammatory cytokines / costimulation
    'Cxcl2', 'Cxcl3', 'Ccl4', 'S100a9',      # inflammatory chemokines / alarmin
    'Grina', 'Ier3', 'Egln3', 'Ero1l',       # hypoxia / ER-stress / immediate-early
    'Mmp12', 'Slpi', 'Lgals3',               # tissue-remodeling / effector
    'Ccr2',                                  # monocyte recruitment
]

# M2 / resident — grouped by function
m2_genes = [
    'Mrc1', 'Chil3', 'Retnla',               # canonical alternatively-activated (CD206/Ym1/Fizz1)
    'Csf1r', 'Fcgr1', 'Cd163',               # resident maintenance / surface markers
    'Gas6', 'Axl', 'Mertk', 'Tyro3',         # TAM efferocytosis axis
    'Apoe', 'C1qb', 'Selenop',               # complement / lipid handling (resident)
    'Folr2', 'Stab1',                        # perivascular / scavenger resident
    'Lpl', 'Igf1', 'Trem2',                  # lipid metabolism / tissue repair
    'Maf',                                # resident-identity TF
]

var_dict = {}
for label, genes in [('MΦ1-Act/Inf/Inf.Mono', m1_genes), ('MΦ2-Res/Rep', m2_genes)]:
    keep = [g for g in genes if g in adata_mac.var_names]
    drop = [g for g in genes if g not in adata_mac.var_names]
    if drop:
        print(f'{label}: dropped (not in var_names): {drop}')
    var_dict[label] = keep

dp = sc.pl.dotplot(
    adata_mac, var_names=var_dict, groupby='Type_Timepoint_C',
    standard_scale='var', cmap='Reds', dot_min=0.0, dot_max=1.0,
    var_group_rotation=0,
    figsize=(12, 2.5),
    return_fig=True,
)

# ── publication styling ───────────────────────────────────────────────────────
axd = dp.get_axes(); ax = axd['mainplot_ax']
for lbl in ax.get_xticklabels():
    lbl.set_fontsize(14); lbl.set_fontweight('bold'); lbl.set_fontstyle('italic')
for lbl in ax.get_yticklabels():
    lbl.set_fontsize(15); lbl.set_fontweight('bold')
if 'gene_group_ax' in axd:
    for txt in axd['gene_group_ax'].texts:
        txt.set_fontsize(15); txt.set_fontweight('bold')
for key in ("size_legend_ax", "color_legend_ax"):
    if key in axd and axd[key] is not None:
        axd[key].tick_params(labelsize=9)

import os
out_dir = os.path.join(os.getcwd(), 'figures'); os.makedirs(out_dir, exist_ok=True)
dp.savefig(os.path.join(out_dir, 'mac_dotplot_m1m2_burn_sham.pdf'), bbox_inches='tight')
dp.savefig(os.path.join(out_dir, 'mac_dotplot_m1m2_burn_sham.png'), dpi=600, bbox_inches='tight')
dp.show()


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc

mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial'})
sc.set_figure_params(dpi=200, dpi_save=600)

# ── row order: Sham D7→D19, then Burn D7→D19 ──────────────────────────────────
present = set(adata_mac.obs["Type_Timepoint_C"].astype(str))
tt_order = [f"{ty} {d}" for ty in ["Sham", "Burn"] for d in ["D7", "D10", "D14", "D19"]
            if f"{ty} {d}" in present]
adata_mac.obs["Type_Timepoint_C"] = pd.Categorical(
    adata_mac.obs["Type_Timepoint_C"].astype(str), categories=tt_order, ordered=True)
print("rows:", tt_order)

# ── gene panels (Davies et al. 2013 M1/M2 framework + burn-vs-sham volcano hits) ──
# M1 / classically-activated / inflammatory  (Davies: Nos2/Tnf/Il1b/Cd86; Ccr2 = Ly6Cʰⁱ mono recruit)
# M1 / inflammatory — grouped by function (Nos2+Arg1 adjacent)
m1_genes = [
    'Arg1',  'Nos2',                       # NO/M1 polarization — dual activation (adjacent)
    'Il1b', 'Tnf', 'Cd86',                   # pro-inflammatory cytokines / costimulation
    'Cxcl2', 'Cxcl3', 'Ccl4', 'S100a9',      # inflammatory chemokines / alarmin
    'Grina', 'Ier3', 'Egln3', 'Ero1l',       # hypoxia / ER-stress / immediate-early
    'Mmp12', 'Slpi', 'Lgals3',               # tissue-remodeling / effector
    'Ccr2', "Bst2", "Cd74", "Siglech" , "Siglec6", "Cd327" ,"Fcgr1", "Id2", "Spic", "Irf7", "Ccr9" , "Irf3","Il3ra", "Flt3" , "Cd4", "Cd3e", "Samhd1", "Cd33l"                          # monocyte recruitment
]

# M2 / resident — grouped by function
m2_genes = [
    'Mrc1', 'Chil3', 'Retnla',               # canonical alternatively-activated (CD206/Ym1/Fizz1)
    'Csf1r', 'Fcgr1', 'Cd163',               # resident maintenance / surface markers
    'Gas6', 'Axl', 'Mertk', 'Tyro3',         # TAM efferocytosis axis
    'Apoe', 'C1qb', 'Selenop',               # complement / lipid handling (resident)
    'Folr2', 'Stab1',                        # perivascular / scavenger resident
    'Lpl', 'Igf1', 'Trem2',                  # lipid metabolism / tissue repair
    'Maf',  "Alox15"                                 # resident-identity TF
]

var_dict = {}
for label, genes in [('MΦ1-Act/Inf/Inf.Mono', m1_genes), ('MΦ2-Res/Rep', m2_genes)]:
    keep = [g for g in genes if g in adata_mac.var_names]
    drop = [g for g in genes if g not in adata_mac.var_names]
    if drop:
        print(f'{label}: dropped (not in var_names): {drop}')
    var_dict[label] = keep

dp = sc.pl.dotplot(
    adata_mac, var_names=var_dict, groupby='macrophage_subtypes',
    standard_scale='var', cmap='Reds', dot_min=0.0, dot_max=1.0,
    var_group_rotation=0,
    figsize=(12, 3),
    return_fig=True,
)

# ── publication styling ───────────────────────────────────────────────────────
axd = dp.get_axes(); ax = axd['mainplot_ax']
for lbl in ax.get_xticklabels():
    lbl.set_fontsize(14); lbl.set_fontweight('bold'); lbl.set_fontstyle('italic')
for lbl in ax.get_yticklabels():
    lbl.set_fontsize(15); lbl.set_fontweight('bold')
if 'gene_group_ax' in axd:
    for txt in axd['gene_group_ax'].texts:
        txt.set_fontsize(15); txt.set_fontweight('bold')
for key in ("size_legend_ax", "color_legend_ax"):
    if key in axd and axd[key] is not None:
        axd[key].tick_params(labelsize=9)

import os
out_dir = os.path.join(os.getcwd(), 'figures'); os.makedirs(out_dir, exist_ok=True)
dp.savefig(os.path.join(out_dir, 'mac_dotplot_m1m2_burn_sham.pdf'), bbox_inches='tight')
dp.savefig(os.path.join(out_dir, 'mac_dotplot_m1m2_burn_sham.png'), dpi=600, bbox_inches='tight')
dp.show()

