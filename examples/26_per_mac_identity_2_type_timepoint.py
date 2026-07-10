"""per mac_identity: 2 (Type) × timepoint split UMAPs, colored by subtype

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, numpy, pandas, re, scipy
Key calls: .plot, barplot, def conf_ellipse, def style_ax, def tp_per_cell, gp.get_library, plt.savefig, plt.show, plt.subplots, plt.tight_layout, sc.tl.score_genes, scatter
"""

# ---------- per mac_identity: 2 (Type) × timepoint split UMAPs, colored by subtype ----------
import pandas as pd, numpy as np, re
import matplotlib as mpl, matplotlib.pyplot as plt

ID_COL = "mac_identity"
ident_all = adata_mac.obs[ID_COL].astype(str)
_have = sorted(ident_all.unique())
_find = lambda *ks: next((h for h in _have if any(k in h.lower() for k in ks)), None)
ID_ORDER = [x for x in [_find("inflamm", "mono"), _find("recruit"), _find("resident", "rep")] if x]

# subtype categorical + palette (reuse mac_colors)
if not pd.api.types.is_categorical_dtype(adata_mac.obs["macrophage_subtypes"]):
    adata_mac.obs["macrophage_subtypes"] = adata_mac.obs["macrophage_subtypes"].astype("category")
cats = list(adata_mac.obs["macrophage_subtypes"].cat.categories)

# Type_Timepoint ordering (same as the cell you have)
tt_levels = ["Sham D7","Burn D7","Sham D10","Burn D10","Sham D14","Burn D14","Sham D19","Burn D19"]
tt_present = set(adata_mac.obs["Type_Timepoint_C"].astype(str))
tt_levels  = [t for t in tt_levels if t in tt_present]
type_order = ["Sham", "Burn"]
tp_order   = sorted({tt.split()[1] for tt in tt_levels}, key=lambda t: int(re.sub(r"\D", "", t)))
TYPE_COL   = {"Sham": "#2471A3", "Burn": "#C0392B"}

um = adata_mac.obsm["X_umap"]
xpad = 0.04 * (um[:, 0].max() - um[:, 0].min()); ypad = 0.04 * (um[:, 1].max() - um[:, 1].min())
xlim = (um[:, 0].min() - xpad, um[:, 0].max() + xpad); ylim = (um[:, 1].min() - ypad, um[:, 1].max() + ypad)

tt_str    = adata_mac.obs["Type_Timepoint_C"].astype(str).values
subt      = adata_mac.obs["macrophage_subtypes"].astype(str).values
ident_arr = ident_all.values

for ident in ID_ORDER:
    in_id = ident_arr == ident
    present_sub = [c for c in cats if (in_id & (subt == c)).any()]   # subtypes in THIS identity
    ncols = len(tp_order)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 9), squeeze=False)
    fig.suptitle(ident, fontsize=24, fontweight="bold", y=1.02)
    for r, ty in enumerate(type_order):
        for c, tp in enumerate(tp_order):
            ax = axes[r][c]; tt = f"{ty} {tp}"
            if tt not in tt_levels:
                ax.set_visible(False); continue
            ax.scatter(um[:, 0], um[:, 1], s=4, c="#ECECEC", linewidths=0, rasterized=True)  # context
            sel  = in_id & (tt_str == tt)
            cols = [mac_colors.get(s, "#999999") for s in subt[sel]]
            ax.scatter(um[sel, 0], um[sel, 1], c=cols, s=20, linewidths=0, rasterized=True)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            style_umap_axes(ax, xlabel="UMAP 1" if r == 1 else "", ylabel="UMAP 2" if c == 0 else "",
                            title=f"{tt}  (n = {int(sel.sum()):,})")
            ax.title.set_fontsize(17); ax.title.set_fontweight("bold"); ax.grid(False)
        axes[r][0].annotate(ty, xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
                            ha="center", va="center", fontsize=20, fontweight="bold", color=TYPE_COL[ty])

    try:
        legend_cats = [c for c in desired_order if c in present_sub]
    except NameError:
        legend_cats = present_sub
    handles = [mpl.patches.Patch(facecolor=mac_colors.get(c, "#999999"), label=c) for c in legend_cats]
    leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
                     title="Subtype", fontsize=(PUB["legend_fs"] if "PUB" in globals() else 14))
    for t in leg.get_texts(): t.set_fontweight("bold")
    plt.tight_layout()
    fig.savefig(FIGDIR_MAC / f"umap_split_TypeTimepoint_{_slug(ident)}_by_subtype.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR_MAC / f"umap_split_TypeTimepoint_{_slug(ident)}_by_subtype.pdf", bbox_inches="tight")
    plt.show()


import re, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ID_COL, SUB_COL = "mac_identity", "macrophage_subtypes"
NORM = True   # True = proportions (each bar 100%); False = raw counts

# identities (Inf -> Recruited -> Resident) + short labels
have  = list(pd.unique(adata_mac.obs[ID_COL].astype(str)))
find  = lambda k: next((h for h in have if k in h.lower()), None)
id_order = [x for x in [find("inflamm"), find("recruit"), find("resident")] if x]
short = {x: lbl for x, lbl in
         [(find("inflamm"), "Inf. Mono."), (find("recruit"), "Recruited"), (find("resident"), "Resident")] if x}

# subtype order (mac_colors first)
sub_cats = list(pd.unique(adata_mac.obs[SUB_COL].astype(str)))
subtypes = [s for s in mac_colors if s in sub_cats] + [s for s in sub_cats if s not in mac_colors]

# timepoints
def tp_per_cell(obs):
    for c in obs.columns:
        if "time" in c.lower(): return obs[c].astype(str)
    for c in obs.columns:
        if obs[c].astype(str).str.fullmatch(r"D?\d+").mean() > 0.8: return obs[c].astype(str)
    for c in obs.columns:
        ext = obs[c].astype(str).str.extract(r"(D\d+)")[0]
        if ext.notna().mean() > 0.8: return ext
    raise ValueError("no timepoint column")
tpc = tp_per_cell(adata_mac.obs).astype(str).str.extract(r"(\d+)")[0].radd("D")
timepoints = sorted(pd.unique(tpc.dropna()), key=lambda t: int(re.search(r"\d+", t).group()))

df = adata_mac.obs[[ID_COL, SUB_COL]].astype(str).copy()
df["tp"] = tpc.values
df = df[df[ID_COL].isin(id_order)]

fig, axes = plt.subplots(1, len(timepoints), figsize=(3.6 * len(timepoints), 6), sharey=True)
axes = np.atleast_1d(axes)
for ax, t in zip(axes, timepoints):
    d  = df[df["tp"] == t]
    ct = (pd.crosstab(d[ID_COL], d[SUB_COL], normalize="index" if NORM else False)
            .reindex(id_order).reindex(columns=subtypes).fillna(0))
    x, bottom = np.arange(len(id_order)), np.zeros(len(id_order))
    for s in subtypes:
        ax.bar(x, ct[s].values, bottom=bottom, color=mac_colors.get(s, "#CCCCCC"),
               width=0.8, edgecolor="white", linewidth=0.4)
        bottom += ct[s].values
    ax.set_xticks(x)
    ax.set_xticklabels([short.get(i, i) for i in id_order], rotation=45, ha="right",
                       fontsize=14, fontweight="bold")
    ax.set_title(t, fontsize=20, fontweight="bold")
    if NORM:
        ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Proportion of cells" if NORM else "Number of cells",
                   fontsize=18, fontweight="bold")

handles = [Patch(facecolor=mac_colors.get(s, "#CCCCCC"), label=s) for s in subtypes]
fig.legend(handles=handles, title="Subtype", loc="center left", bbox_to_anchor=(1.0, 0.5),
           frameon=False, fontsize=12, title_fontsize=13)
fig.tight_layout()
fig.savefig(FIGDIR_MAC / "barplot_identity_filled_subtype_by_timepoint.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "barplot_identity_filled_subtype_by_timepoint.pdf", bbox_inches="tight")
plt.show()


import re, scanpy as sc, gseapy as gp
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import chi2

# ── scores via score_genes (HALLMARK, mouse) ──────────────────────────────────
h2m  = lambda g: g[0].upper() + g[1:].lower()
hall = gp.get_library('MSigDB_Hallmark_2020', organism='Mouse')
glyco  = [h2m(g) for g in hall['Glycolysis'] if h2m(g) in adata_mac.var_names]
oxphos = [h2m(g) for g in hall['Oxidative Phosphorylation'] if h2m(g) in adata_mac.var_names]
hypoxia = [h2m(g) for g in hall['Hypoxia'] if h2m(g) in adata_mac.var_names]
sc.tl.score_genes(adata_mac, glyco,  score_name='Glycolysis_Score', use_raw=False, random_state=0)
sc.tl.score_genes(adata_mac, oxphos, score_name='OXPHOS_Score',     use_raw=False, random_state=0)
sc.tl.score_genes(adata_mac, hypoxia, score_name='Hypoxia_Score',   use_raw=False, random_state=0)
adata_mac.obs['Metabolic_Ratio'] = adata_mac.obs['Glycolysis_Score'] - adata_mac.obs['OXPHOS_Score']
obs = adata_mac.obs.copy()

# ── one 95% confidence ellipse per group ──────────────────────────────────────
def conf_ellipse(x, y, ax, conf=0.95, **kw):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 5:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = np.sqrt(chi2.ppf(conf, df=2))          # 2D 95% region
    w, h = 2 * scale * np.sqrt(vals)
    ax.add_patch(Ellipse((x.mean(), y.mean()), width=w, height=h, angle=theta, **kw))

TYPE_PAL = {'Burn': '#C0392B', 'Sham': '#2980B9'}
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 30, 22, 18, 20
TP_ORDER = sorted(obs['Timepoint'].astype(str).unique(),
                  key=lambda t: int(re.search(r'\d+', t).group()))

# axis pairs to show, one per row (x, y, xlabel, ylabel)
PAIRS = [
    ("OXPHOS_Score",  "Glycolysis_Score", "OXPHOS Score",  "Glycolysis Score"),
    ("Hypoxia_Score", "Glycolysis_Score", "Hypoxia Score", "Glycolysis Score"),
]

# shared comparable limits across all three scores
score_cols = ["OXPHOS_Score", "Glycolysis_Score", "Hypoxia_Score"]
LO = float(obs[score_cols].min().min()); HI = float(obs[score_cols].max().max())
pad = 0.05 * (HI - LO); LIM = (LO - pad, HI + pad)

def style_ax(ax):
    ax.tick_params(axis='both', labelsize=FS_TICK, width=1.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_linewidth(1.2); ax.spines['bottom'].set_linewidth(1.2)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

nrow, ncol = len(PAIRS), len(TP_ORDER)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 5.0 * nrow), squeeze=False)

for ri, (xk, yk, xlab, ylab) in enumerate(PAIRS):
    for ci, tp in enumerate(TP_ORDER):
        ax = axes[ri][ci]
        sub_tp = obs[obs['Timepoint'].astype(str) == tp]
        for cond, z in [('Sham', 1), ('Burn', 2)]:
            c = sub_tp[sub_tp['Type'] == cond]
            ax.scatter(c[xk], c[yk], c=TYPE_PAL[cond], s=10, alpha=0.5,
                       rasterized=True, zorder=z, edgecolors='none')
            conf_ellipse(c[xk], c[yk], ax, conf=0.95, edgecolor=TYPE_PAL[cond],
                         facecolor=TYPE_PAL[cond], alpha=0.2, lw=2.8, zorder=z + 3)
        ax.plot([LO, HI], [LO, HI], 'k--', lw=1, alpha=0.8, zorder=0)
        ax.set_xlim(LIM); ax.set_ylim(LIM)
        if ri == 0:
            ax.set_title(tp, fontsize=FS_TITLE, fontweight='bold')
        ax.set_xlabel(xlab, fontsize=FS_LABEL, fontweight='bold')   # per-row x meaning
        if ci == 0:
            ax.set_ylabel(ylab, fontsize=FS_LABEL, fontweight='bold')
        style_ax(ax)

handles = [Line2D([0], [0], color=TYPE_PAL[c], lw=3, label=c) for c in ['Sham', 'Burn']]
fig.legend(handles=handles, fontsize=FS_LEGEND, loc='upper right', frameon=False,
           bbox_to_anchor=(0.99, 0.5))

plt.tight_layout()
FIG = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/figures"
plt.savefig(f'{FIG}/mac_hallmark_statespace_ellipse_hypoxia.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG}/mac_hallmark_statespace_ellipse_hypoxia.png', dpi=300, bbox_inches='tight')
plt.show()

