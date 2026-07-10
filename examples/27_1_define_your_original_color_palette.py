"""1. Define your original color palette

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, numpy, re, scanpy, scipy, statsmodels
Key calls: .plot, def add_ct_labels, def conf_ellipse, def score_set, def stars, def style_ax, gp.get_library, plt.savefig, plt.show, plt.subplots, plt.tight_layout, sc.pl.umap
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


# 1. Define your original color palette
major_cell_colors_full = {
   "Keratinocytes":     "#FF7F00",
   "Fibroblasts":       "#33A02C",
   "Neutrophils":       "#E31A1C",
   "Macrophages":       "#1F78B4",
   "Monocytes":  "#E6AB02",
   "Conventional Dendritic Cells":   "#6A3D9A",
   "Endothelial Cells": "#FB9A99",
   "T Cells":           "#A6CEE3",
   "Smooth Muscle Cells":      "#B2DF8A",
   "Sebaceous Gland Cells":         "#720946",
   #"Adipocytes":        "#CAB2D6"
}

# 2. Define the desired order for plotting/categories
desired_order_full = [
   "Keratinocytes",
   "Fibroblasts",
   "Neutrophils",
   "Macrophages",
   "Monocytes",
   "Conventional Dendritic Cells",
   "Endothelial Cells",
   "T Cells",
   "Smooth Muscle Cells",
   "Sebaceous Gland Cells",
   #"Adipocytes"
]

# 3. Map your clusters based on the DEG analysis
cluster_to_celltype_full = {
   "0":  "Neutrophils",        # S100a8/9, Cxcl2, Ccl3, Acod1, Il1b — see caveat below
   "1":  "Smooth Muscle Cells",       # Ttn, Neb, Ryr1, Actn3 — SKELETAL muscle
   "2":  "Fibroblasts",       # Col1a1/2, Col3a1, Dcn, Mfap5, Dpt
   "3":  "Conventional Dendritic Cells",       # Ciita, H2-Aa/Ab1/Eb1, Cd74 (cDC/APC)
   "4":  "Neutrophils",        # Acod1, Hdc, Ccl3/4, Nlrp3, Clec4e, Cd14
   "5":  "Neutrophils",        # Ccl3, Cxcl2, Acod1, Cd274, Hilpda
   "6":  "Macrophages",         # C1qb, Mrc1, Stab1, Pf4, Apoe, Csf1r (resident)
   "7":  "Keratinocytes",        # Lef1, Msx2, Dlx3, Krt25/28, Mki67 (HF matrix)
   "8":  "T Cells",    # Cd3d/e/g, Cd2, Lck, Zap70, Il2rb
   "9":  "Endothelial Cells",      # Pecam1, Cdh5, Kdr, Tie1, Egfl7
   "10": "Keratinocytes",        # Krt5/14, Col17a1, Trp63, Lama3 (basal)
   "11": "Keratinocytes",        # Krt15/17, Lhx2, Bnc2, Cxcl14 (follicle/stem)
   "12": "Neutrophils",        # Cxcr2, Csf3r, Sell, Hp, Ncf2
   "13": "Keratinocytes",        # Krt1/10, Dsg1a/b, Krtdap (suprabasal)
   "14": "Fibroblasts",       # Tnn, Lrrc15, Lox, Tnc, Thbs2 (activated)
   "15": "Macrophages",         # Arg1, Nos2, F13a1, Hilpda (recruited/inflammatory)
   "16": "Sebaceous Gland Cells",        # Awat1, Scd1, Far2, Mgll, Krt79, Cidea
   "17": "Monocytes",        # Msr1, Fcgr1, Lyz1, S100a4, Arg1 (monocyte-derived)
   "18": "Macrophages",         # Adgre1, Mrc1, C1qb, Stab1, Ms4a4a/6c (resident)
   "19": "Smooth Muscle Cells",       # Myh11, Rgs5, Pdgfrb, Notch3 — mural/SMOOTH muscle
   "20": "Neutrophils",        # Cxcr2, Csf3r, Mmp9, Trem1, Hp
   "21": "Keratinocytes",        # Tchh, Krt71/73/25/27, Padi3 (IRS)
   "22": "Keratinocytes",        # Krt31/33/36/83, Krtaps, Foxn1, Hoxc13 (hair shaft)
   "23": "Keratinocytes",        # Lor, Casp14, Aloxe3, Klk7, Cnfn (cornified)
}

# 4. Apply mapping, order categories, and assign colors to your AnnData object
# Ensure the cluster column is string type to match the dictionary keys
adata_full.obs["leiden_res0.8"] = adata_full.obs["leiden_res0.8"].astype(str)

adata_full.obs["cell_types_full"] = (adata_full.obs["leiden_res0.8"]
                                      .map(cluster_to_celltype_full)
                                      .astype("category"))

# Set the factor levels to your preferred order
adata_full.obs["cell_types_full"] = adata_full.obs["cell_types_full"].cat.set_categories(desired_order_full)
# Map colors strictly matching your desired order
adata_full.uns["cell_types_full_colors"] = [major_cell_colors_full[c] for c in desired_order_full]

# 5. QC checks
print("--- Cell Type Counts ---")
print(adata_full.obs["cell_types_full"].value_counts())
print("\n--- QC Check ---")
print("Missing from dataset:", set(desired_order_full) - set(adata_full.obs["cell_types_full"].dropna().unique()))







import re, numpy as np, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

# ── cell-type column + labels ─────────────────────────────────────────────────
CT_COL = next((c for c in ["cell_types_simple", "cell_types_full", "cell_type", "cell_types"]
               if c in adata_full.obs.columns), None)
assert CT_COL, f"set CT_COL manually from {list(adata_full.obs.columns)}"

xy  = adata_full.obsm["X_umap"]
cts = adata_full.obs[CT_COL].astype(str)
centroids = {ct: np.median(xy[cts.values == ct], axis=0) for ct in cts.unique()}

def add_ct_labels(ax, fs=24):
    for ct, (cx, cy) in centroids.items():
        ax.text(cx, cy, ct, fontsize=fs, fontweight="bold", ha="center", va="center",
                color="black", zorder=12,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# ── mac/mono mask (verify printed labels!) ────────────────────────────────────
mask = cts.str.contains(r"macro|mono|mφ|mΦ", case=False, regex=True).values
print("encircled cell types:", sorted(cts[mask].unique()))
pts = xy[mask]

# ── build a dashed outline (xs, ys) around the population ──────────────────────
OUTLINE = "hull"        # "hull" (tight, smoothed)  or  "ellipse" (clean oval)
PAD     = 1.12          # outward expansion factor

ctr = np.median(pts, axis=0)
d   = np.linalg.norm(pts - ctr, axis=1)
core = pts[d <= np.percentile(d, 97)]     # trim outliers for a clean boundary

if OUTLINE == "ellipse":
    c = core.mean(0); cov = np.cov(core.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    theta = np.arctan2(vecs[1, 0], vecs[0, 0])
    n_std = 2.6                                       # ~99% coverage
    w, h = 2 * n_std * np.sqrt(vals)
    t = np.linspace(0, 2 * np.pi, 400)
    ell = np.array([w / 2 * np.cos(t), h / 2 * np.sin(t)])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = (R @ ell).T * PAD + c
    xs, ys = out[:, 0], out[:, 1]
else:  # convex hull, expanded + smoothed
    hull = ConvexHull(core)
    hv   = core[hull.vertices]
    hc   = hv.mean(0)
    hv   = hc + (hv - hc) * PAD
    xh = np.append(hv[:, 0], hv[0, 0]); yh = np.append(hv[:, 1], hv[0, 1])
    try:
        tck, _ = splprep([xh, yh], s=hv.shape[0] * 0.8, per=True)
        xs, ys = splev(np.linspace(0, 1, 400), tck)
    except Exception:
        xs, ys = xh, yh                              # fallback: sharp polygon

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
fig.subplots_adjust(right=0.74)                      # room for the arrow/callout
import scanpy as sc
sc.pl.umap(adata_full, color=CT_COL, palette=list(adata_full.uns["cell_types_full_colors"]),
           ax=ax, show=False, frameon=True, size=8, legend_loc=None, title="")
add_ct_labels(ax)

ax.plot(xs, ys, ls="--", lw=3.5, color="black", zorder=15,
        path_effects=[pe.withStroke(linewidth=5.5, foreground="white")])


ax.set_title("")
ax.set_xlabel("UMAP 1", fontsize=35, fontweight="bold")
ax.set_ylabel("UMAP 2", fontsize=35, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])

FIG = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/figures"
fig.savefig(f"{FIG}/umap_full_macmono_circled_arrow.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG}/umap_full_macmono_circled_arrow.pdf", bbox_inches="tight")
plt.show()


import re, scanpy as sc, gseapy as gp
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import chi2

# ── object with all cell types + sample metadata ──────────────────────────────
adata = adata_full
CT_COL = next((c for c in ["cell_types_simple", "cell_types_full", "cell_type", "cell_types"]
               if c in adata.obs.columns), None)
assert CT_COL, f"set CT_COL manually from {list(adata.obs.columns)}"
for col in ['Type', 'Timepoint']:
    assert col in adata.obs.columns, f"{col} missing from adata_full.obs — use the object carrying sample metadata"

# ── subset to macrophages + monocytes (verify the printed labels!) ────────────
ct   = adata.obs[CT_COL].astype(str)
mask = ct.str.contains(r"MΦ|Mono.", case=False, regex=True)   # hardcode exact labels if needed
print("kept cell types:", sorted(ct[mask].unique()))
adata_mm = adata[mask].copy()
print(f"{adata_mm.n_obs} macrophage/monocyte cells")

# ── Hallmark scores (mouse) on the mac/mono subset ────────────────────────────
h2m  = lambda g: g[0].upper() + g[1:].lower()
hall = gp.get_library('MSigDB_Hallmark_2020', organism='Mouse')
def score_set(ad, key, name):
    genes = [h2m(g) for g in hall[key] if h2m(g) in ad.var_names]
    sc.tl.score_genes(ad, genes, score_name=name, use_raw=False, random_state=0)
score_set(adata_mm, 'Glycolysis',                'Glycolysis_Score')
score_set(adata_mm, 'Oxidative Phosphorylation', 'OXPHOS_Score')
score_set(adata_mm, 'Hypoxia',                   'Hypoxia_Score')

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import chi2, ks_2samp
from statsmodels.stats.multitest import multipletests

TYPE_PAL = {'Burn': '#C0392B', 'Sham': '#2980B9'}
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_STAT = 30, 24, 26, 20, 21
XLAB, YLAB = 'OXPHOS (z)', 'Glycolysis + Hypoxia (z)'
TP_ORDER = sorted(obs['Timepoint'].astype(str).unique(),
                  key=lambda t: int(re.search(r'\d+', t).group()))

LO = float(obs[[XK, YK]].min().min()); HI = float(obs[[XK, YK]].max().max())
pad = 0.05 * (HI - LO); LIM = (LO - pad, HI + pad)

def conf_ellipse(x, y, ax, conf=0.95, **kw):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 5: return
    vals, vecs = np.linalg.eigh(np.cov(x, y))
    order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * np.sqrt(chi2.ppf(conf, df=2)) * np.sqrt(vals)
    ax.add_patch(Ellipse((x.mean(), y.mean()), width=w, height=h, angle=theta, **kw))

def style_ax(ax, show_y=True):
    ax.tick_params(axis='both', labelsize=FS_TICK, width=1.4, length=6)
    ax.tick_params(axis='y', labelleft=show_y)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_linewidth(1.4); ax.spines['bottom'].set_linewidth(1.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

def stars(p):
    if not np.isfinite(p): return 'n/a'
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'

# ── KS test (Burn vs Sham on combined Glyco+Hypoxia axis) per timepoint ───────
ks_rows = []
for tp in TP_ORDER:
    sub = obs[obs['Timepoint'].astype(str) == tp]
    b = sub.loc[sub['Type'] == 'Burn', YK].dropna().values
    s = sub.loc[sub['Type'] == 'Sham', YK].dropna().values
    D, p = ks_2samp(b, s) if len(b) > 2 and len(s) > 2 else (np.nan, np.nan)
    ks_rows.append(dict(tp=tp, D=D, p=p, n_burn=len(b), n_sham=len(s)))
ksdf = pd.DataFrame(ks_rows)
ok = ksdf['p'].notna()
ksdf.loc[ok, 'padj'] = multipletests(ksdf.loc[ok, 'p'], method='fdr_bh')[1]
print(ksdf.to_string(index=False))
ks_lookup = ksdf.set_index('tp')

# ── single-row scatter, one panel per timepoint ──────────────────────────────
ncol = len(TP_ORDER)
fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.6), squeeze=False, sharey=True)
for ci, tp in enumerate(TP_ORDER):
    ax  = axes[0][ci]
    sub = obs[obs['Timepoint'].astype(str) == tp]
    for cond, zo in [('Sham', 1), ('Burn', 2)]:
        c = sub[sub['Type'] == cond]
        ax.scatter(c[XK], c[YK], c=TYPE_PAL[cond], s=10, alpha=0.5,
                   rasterized=True, zorder=zo, edgecolors='none')
        conf_ellipse(c[XK], c[YK], ax, conf=0.95, edgecolor=TYPE_PAL[cond],
                     facecolor=TYPE_PAL[cond], alpha=0.2, lw=2.8, zorder=zo + 3)
    ax.axhline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
    ax.axvline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
    ax.plot([LO, HI], [LO, HI], 'k--', lw=1, alpha=0.6, zorder=0)
    ax.set_xlim(LIM); ax.set_ylim(LIM)

    r = ks_lookup.loc[tp]
    ax.text(0.04, 0.96, f"KS D={r.D:.2f}\n{stars(r.padj)}", transform=ax.transAxes,
            ha='left', va='top', fontsize=FS_STAT, fontweight='bold')

    ax.set_title(tp, fontsize=FS_TITLE, fontweight='bold')
    ax.set_xlabel(XLAB, fontsize=FS_LABEL, fontweight='bold')
    if ci == 0:
        ax.set_ylabel(YLAB, fontsize=FS_LABEL, fontweight='bold')
    style_ax(ax, show_y=(ci == 0))

handles = [Line2D([0], [0], color=TYPE_PAL[c], lw=3, label=c) for c in ['Sham', 'Burn']]
fig.legend(handles=handles, fontsize=FS_LEGEND, loc='upper right', frameon=False,
           bbox_to_anchor=(0.99, 0.90))

plt.tight_layout()
FIG = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/figures"
plt.savefig(f'{FIG}/macmono_oxphos_vs_glycohypoxia_scatter_ks.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG}/macmono_oxphos_vs_glycohypoxia_scatter_ks.png', dpi=300, bbox_inches='tight')
plt.show()

