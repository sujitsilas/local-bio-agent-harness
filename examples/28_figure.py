"""figure

Source: macrophages_resident_recruited.ipynb
Libraries: gseapy, matplotlib, mpl_toolkits, numpy, re, scipy, seaborn
Key calls: def add_ct_labels, def conf_ellipsoid, plt.figure, plt.savefig, plt.show, plt.subplots, plt.tight_layout, sc.pl.umap, sc.tl.score_genes, scatter, sns.kdeplot, umap
"""

# ── figure ────────────────────────────────────────────────────────────────────
COLOR_BY = 'balance'     # 'balance' (gradient + condition ellipses) or 'condition'
vlim    = float(np.nanpercentile(np.abs(obs['MB']), 98))
xlo, xhi = np.nanpercentile(obs[XK], [1, 99])
ylo, yhi = np.nanpercentile(obs[YK], [1, 99])

ncol = len(TP_ORDER)
fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 6.0), squeeze=False,
                         sharex=True, sharey=True)
fig.subplots_adjust(left=0.09, right=0.88, bottom=0.20, top=0.90, wspace=0.07)

mappable = None
for ci, tp in enumerate(TP_ORDER):
    ax  = axes[0][ci]
    sub = obs[obs['Timepoint'].astype(str) == tp]

    if COLOR_BY == 'balance':
        # every cell drawn exactly ONCE; condition shown via ellipse outlines only
        mappable = ax.scatter(sub[XK], sub[YK], c=sub['MB'], cmap='RdBu_r',
                              vmin=-vlim, vmax=vlim, s=7, alpha=0.6,
                              edgecolors='none', rasterized=True, zorder=1)
        for cond in ['Sham', 'Burn']:
            c = sub[sub['Type'] == cond]
            conf_ellipse(c[XK], c[YK], ax, conf=0.95, edgecolor=TYPE_PAL[cond],
                         facecolor='none', lw=3.5, zorder=5)
    else:  # 'condition' — each cell once, colored by its group
        for cond, zo in [('Sham', 1), ('Burn', 2)]:
            c = sub[sub['Type'] == cond]
            ax.scatter(c[XK], c[YK], c=TYPE_PAL[cond], s=7, alpha=0.45,
                       edgecolors='none', rasterized=True, zorder=zo)
            conf_ellipse(c[XK], c[YK], ax, conf=0.95, edgecolor=TYPE_PAL[cond],
                         facecolor='none', lw=3.0, zorder=6)

    ax.axhline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
    ax.axvline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

    r = ks_lookup.loc[tp]
    ax.text(0.04, 0.97, f"KS D={r.D:.2f}\n{stars(r.padj)}", transform=ax.transAxes,
            ha='left', va='top', fontsize=FS_STAT, fontweight='bold')
    ax.set_title(tp, fontsize=FS_TITLE, fontweight='bold', pad=8)
    ax.set_xlabel(XLAB, fontsize=FS_LABEL, fontweight='bold')
    if ci == 0:
        ax.set_ylabel('OXPHOS ↔ Glycolysis', fontsize=FS_LABEL , fontweight='bold', labelpad=8)
    style_ax(ax, show_y=(ci == 0))

# colorbar (balance mode), to the right
if COLOR_BY == 'balance' and mappable is not None:
    cax = fig.add_axes([0.90, 0.30, 0.012, 0.42])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label('OXPHOS  ↔  Glycolysis', fontsize=14, fontweight='bold')
    cb.ax.tick_params(labelsize=11)

# condition legend BELOW the panels (no overlap with titles / y-label)
handles = [Line2D([0], [0], color=TYPE_PAL[c], lw=4, label=c) for c in ['Sham', 'Burn']]
fig.legend(handles=handles, fontsize=FS_LEGEND, loc='lower center',
           bbox_to_anchor=(0.49, -0.02), ncol=2, frameon=False,
           title='', title_fontsize=FS_LEGEND)

FIG = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/figures"
fig.savefig(f'{FIG}/macmono_hypoxia_vs_metabalance_{COLOR_BY}.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG}/macmono_hypoxia_vs_metabalance_{COLOR_BY}.png', dpi=300, bbox_inches='tight')
plt.show()


import seaborn as sns
from matplotlib.lines import Line2D

ncol = len(TP_ORDER)
fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 6.0), squeeze=False,
                         sharex=True, sharey=True)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.90, wspace=0.07)

xlo, xhi = np.nanpercentile(obs[XK], [1, 99])
ylo, yhi = np.nanpercentile(obs[YK], [1, 99])

for ci, tp in enumerate(TP_ORDER):
    ax  = axes[0][ci]
    sub = obs[obs['Timepoint'].astype(str) == tp]
    for cond in ['Sham', 'Burn']:
        c = sub[sub['Type'] == cond]
        # faint points for context (each cell once)
        ax.scatter(c[XK], c[YK], s=10, alpha=0.4, color=TYPE_PAL[cond],
                   edgecolors='none', rasterized=True, zorder=1)
        # filled density + crisp outline = the condition difference
        sns.kdeplot(data=c, x=XK, y=YK, ax=ax, color=TYPE_PAL[cond],
                    levels=5, thresh=0.10, fill=True, alpha=0.2, zorder=2)
        sns.kdeplot(data=c, x=XK, y=YK, ax=ax, color=TYPE_PAL[cond],
                    levels=5, thresh=0.10, linewidths=1.6, alpha=0.2, zorder=3)
    # centroids + Sham->Burn shift arrow
    cs = sub.loc[sub['Type'] == 'Sham', [XK, YK]].mean().values
    cb = sub.loc[sub['Type'] == 'Burn', [XK, YK]].mean().values
    for ctr, cond in [(cs, 'Sham'), (cb, 'Burn')]:
        ax.scatter(*ctr, s=220, color=TYPE_PAL[cond], edgecolor='black',
                   lw=2, zorder=6)
    #ax.annotate("", xy=cb, xytext=cs, zorder=7,
    #            arrowprops=dict(arrowstyle='-|>', lw=2.8, color='black', mutation_scale=24))

    ax.axhline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
    ax.axvline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

    r = ks_lookup.loc[tp]
    ax.text(0.04, 0.97, f"KS D={r.D:.2f}\n{stars(r.padj)}", transform=ax.transAxes,
            ha='left', va='top', fontsize=FS_STAT, fontweight='bold')
    ax.set_title(tp, fontsize=FS_TITLE, fontweight='bold', pad=8)
    ax.set_xlabel(XLAB, fontsize=FS_LABEL, fontweight='bold')
    if ci == 0:
        ax.set_ylabel("OXPHOS ↔ Glycolysis", fontsize=FS_LABEL, fontweight='bold', labelpad=8)
    style_ax(ax, show_y=(ci == 0))

handles = [Line2D([0], [0], color=TYPE_PAL[c], lw=6, label=c) for c in ['Sham', 'Burn']]
fig.legend(handles=handles, fontsize=FS_LEGEND, loc='lower center',
           bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)

FIG = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/figures"
fig.savefig(f'{FIG}/macmono_hypoxia_vs_metabalance_density_arrow.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG}/macmono_hypoxia_vs_metabalance_density_arrow.png', dpi=300, bbox_inches='tight')
plt.show()


import re, numpy as np, matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3d projection)

obs = adata_mac.obs.copy()

# x, y, z = OXPHOS, Hypoxia, Glycolysis
AX = [("OXPHOS_Score", "OXPHOS"), ("Hypoxia_Score", "Hypoxia"), ("Glycolysis_Score", "Glycolysis")]
(xk, xl), (yk, yl), (zk, zl) = AX

TYPE_PAL = {'Burn': '#C0392B', 'Sham': '#2980B9'}
FS_TITLE, FS_LABEL, FS_LEGEND = 26, 15, 20
TP_ORDER = sorted(obs['Timepoint'].astype(str).unique(),
                  key=lambda t: int(re.search(r'\d+', t).group()))

# shared comparable limits across all three scores
score_cols = [xk, yk, zk]
LO = float(obs[score_cols].min().min()); HI = float(obs[score_cols].max().max())
pad = 0.05 * (HI - LO); LIM = (LO - pad, HI + pad)

def conf_ellipsoid(ax, X, conf=0.95, n=24, **kw):
    """95% confidence ellipsoid for an N x 3 cloud."""
    X = np.asarray(X, float)
    if len(X) < 5:
        return
    ctr = X.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(X.T))
    radii = np.sqrt(chi2.ppf(conf, df=3) * np.clip(vals, 0, None))  # 3D 95% region
    u = np.linspace(0, 2 * np.pi, n); v = np.linspace(0, np.pi, n)
    sphere = np.stack([np.outer(np.cos(u), np.sin(v)),
                       np.outer(np.sin(u), np.sin(v)),
                       np.outer(np.ones_like(u), np.cos(v))], axis=-1)
    pts = sphere @ np.diag(radii) @ vecs.T + ctr
    ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], linewidth=0,
                    antialiased=True, shade=True, **kw)

ncol = len(TP_ORDER)
fig = plt.figure(figsize=(6.2 * ncol, 6.4))
for ci, tp in enumerate(TP_ORDER):
    ax = fig.add_subplot(1, ncol, ci + 1, projection='3d')
    sub_tp = obs[obs['Timepoint'].astype(str) == tp]
    for cond in ['Sham', 'Burn']:
        c = sub_tp[sub_tp['Type'] == cond]
        ax.scatter(c[xk], c[yk], c[zk], s=6, alpha=0.35, color=TYPE_PAL[cond],
                   edgecolors='none', depthshade=False, rasterized=True)
        conf_ellipsoid(ax, c[[xk, yk, zk]].values, color=TYPE_PAL[cond], alpha=0.18)
    ax.set_xlim(LIM); ax.set_ylim(LIM); ax.set_zlim(LIM)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(xl, fontsize=FS_LABEL, fontweight='bold', labelpad=10)
    ax.set_ylabel(yl, fontsize=FS_LABEL, fontweight='bold', labelpad=10)
    ax.set_zlabel(zl, fontsize=FS_LABEL, fontweight='bold', labelpad=10)
    ax.set_title(tp, fontsize=FS_TITLE, fontweight='bold', pad=4)
    ax.view_init(elev=20, azim=-60)   # tweak to taste

handles = [Line2D([0], [0], marker='o', linestyle='', markerfacecolor=TYPE_PAL[c],
                  markeredgecolor='none', markersize=12, label=c) for c in ['Sham', 'Burn']]
fig.legend(handles=handles, fontsize=FS_LEGEND, loc='upper right', frameon=False,
           bbox_to_anchor=(0.99, 0.92))

plt.tight_layout()
FIG = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/figures"
plt.savefig(f'{FIG}/mac_hallmark_statespace_3d_ellipsoid.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG}/mac_hallmark_statespace_3d_ellipsoid.png', dpi=300, bbox_inches='tight')
plt.show()


import numpy as np
from gseapy import Msigdb   # pip install gseapy  (if not already installed)

# ── pull Hallmark gene sets straight from MSigDB (mouse-ortholog collection) ──
msig = Msigdb()

# Inspect available builds/categories if the version below errors:
#   print(msig.list_dbver())                       # e.g. 2024.1.Mm, 2023.2.Mm ...
#   print(msig.list_category(dbver="2024.1.Mm"))   # mh.all = mouse Hallmark
DBVER    = "2024.1.Mm"     # mouse MSigDB build  (use 2024.1.Hs + 'h.all' for human)
CATEGORY = "mh.all"        # mouse Hallmark collection
gmt = msig.get_gmt(category=CATEGORY, dbver=DBVER)

WANTED = {
    "Glycolysis_Score": "HALLMARK_GLYCOLYSIS",
    "OXPHOS_Score":     "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "Hypoxia_Score":    "HALLMARK_HYPOXIA",
}

# sanity check the set names resolved
missing = [s for s in WANTED.values() if s not in gmt]
assert not missing, f"Not in {CATEGORY}/{DBVER}: {missing}\nAvailable: {sorted(gmt)[:5]}..."

# ── score each signature (overwrites the old columns) ────────────────────────
for score_name, set_name in WANTED.items():
    genes   = gmt[set_name]
    present = [g for g in genes if g in adata_full.var_names]
    print(f"{set_name:38s}: {len(present):3d}/{len(genes)} genes present in adata")
    sc.tl.score_genes(adata_full, gene_list=present,
                      score_name=score_name, use_raw=False)


import numpy as np
import matplotlib.patheffects as pe

# ── cell-type column for labels ───────────────────────────────────────────────
CT_COL = next((c for c in ["cell_types_simple", "cell_types_full", "cell_type", "cell_types"]
               if c in adata_full.obs.columns), None)
assert CT_COL, f"set CT_COL manually from {list(adata_full.obs.columns)}"
print("labeling by:", CT_COL)

xy  = adata_full.obsm["X_umap"]
cts = adata_full.obs[CT_COL].astype(str)
centroids = {ct: np.median(xy[cts.values == ct], axis=0) for ct in cts.unique()}

def add_ct_labels(ax, fs=24):
    for ct, (cx, cy) in centroids.items():
        ax.text(cx, cy, ct, fontsize=fs, fontweight="bold", ha="center", va="center",
                color="black", zorder=10,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# ── rescale each score to 0–2 ─────────────────────────────────────────────────
disp = {}
for score in ["Glycolysis_Score", "OXPHOS_Score", "Hypoxia_Score"]:
    v = adata_full.obs[score].astype(float).values
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    adata_full.obs[f"{score}_disp"] = (v - vmin) / (vmax - vmin) * 2.0
    disp[score] = f"{score}_disp"

# ── feature plots, stacked VERTICALLY, with cell-type labels on top ───────────
panels = [
    ("Glycolysis_Score", "Glycolysis"),
    ("OXPHOS_Score",     "OXPHOS"),
    ("Hypoxia_Score",    "Hypoxia"),
]
fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 6.8 * len(panels)))
axes = np.atleast_1d(axes)
for ax, (score, title) in zip(axes, panels):
    sc.pl.umap(adata_full, color=disp[score], cmap="RdBu_r",
               vmin=0, vmax=2, sort_order=True, size=6,
               frameon=True, show=False, ax=ax, colorbar_loc="right")
    add_ct_labels(ax)                                    # <- labels on top
    ax.set_title(title, fontsize=40, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=40, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=40, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

# clean 0–2 colorbar ticks (once, after all panels drawn
for cax in fig.axes:
    if cax.get_label() == "<colorbar>":
        cax.set_yticks([0, 0.5, 1, 1.5, 2])
        cax.set_yticklabels(["0", "0.5", "1", "1.5", "2"], fontsize=28, fontweight="bold")

fig.tight_layout()
#fig.savefig(FIGDIR_MAC / "umap_full_glyco_oxphos_hypoxia_hallmark_0to2_labeled.png", dpi=300, bbox_inches="tight")
#fig.savefig(FIGDIR_MAC / "umap_full_glyco_oxphos_hypoxia_hallmark_0to2_labeled.pdf", bbox_inches="tight")
plt.show()


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
    ax.set_title(title, fontsize=30, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=30, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=30, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    # clean 0–2 colorbar ticks
    for cax in fig.axes:
        if cax.get_label() == "<colorbar>":
            cax.set_yticks([0, 0.5, 1, 1.5, 2])

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "umap_full_glyco_oxphos_scores_0to2.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "umap_full_glyco_oxphos_scores_0to2.pdf", bbox_inches="tight")
plt.show()

