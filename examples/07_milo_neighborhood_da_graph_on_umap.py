"""Milo neighborhood DA graph on UMAP — node size = neighborhood size

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, numpy, pandas, pathlib, scanpy, scipy
Key calls: .plot, def _ov, def cliffs, def expr_of, def shorten, plt.show, plt.subplots, sc.tl.score_genes, scatter, sns.kdeplot, sns.violinplot, umap
"""

# ══ Milo neighborhood DA graph on UMAP — node size = neighborhood size ═══════════
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

um = ad.obsm["X_umap"]; pos = um[idx]
xpad = 0.03 * (um[:, 0].max() - um[:, 0].min()); ypad = 0.03 * (um[:, 1].max() - um[:, 1].min())
xlim = (um[:, 0].min() - xpad, um[:, 0].max() + xpad); ylim = (um[:, 1].min() - ypad, um[:, 1].max() + ypad)

# node area ∝ neighborhood size (n cells); smaller overall scale
SIZE_BY = nh_size                       # <- the parameter mapped to circle size
SMIN, SMAX = 5, 80
sz = SMIN + (SMAX - SMIN) * (SIZE_BY / SIZE_BY.max())

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(um[:, 0], um[:, 1], s=3, c="#ECECEC", linewidths=0, rasterized=True, zorder=1)
ov = (N.T @ N).tocoo()
seg = [[pos[i], pos[j]] for i, j, v in zip(ov.row, ov.col, ov.data) if i < j and v >= 5]
if seg:
    ax.add_collection(LineCollection(seg, colors="0.8", linewidths=0.25, alpha=0.35, zorder=2))
ax.scatter(pos[~sig, 0], pos[~sig, 1], s=sz[~sig], color="0.85", edgecolor="0.55", linewidth=0.25, zorder=3)
sca = ax.scatter(pos[sig, 0], pos[sig, 1], s=sz[sig], c=clip(lfc[sig]), cmap=cmap, norm=norm,
                 edgecolor="black", linewidth=0.4, zorder=4)

ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1", fontsize=PUB["axis_label_fs"], fontweight="bold")
ax.set_ylabel("UMAP 2", fontsize=PUB["axis_label_fs"], fontweight="bold")
ax.set_title("", fontsize=18, fontweight="bold")
for spn in ax.spines.values(): spn.set_linewidth(PUB["border_lw"])

# colorbar (Burn top / Sham bottom)
cb = fig.colorbar(sca, ax=ax, fraction=0.035, pad=0.02)
cb.set_ticks([]); cb.outline.set_linewidth(0.8)
cb.ax.text(0.5, 1.03, "", transform=cb.ax.transAxes, ha="center", va="bottom",
           fontsize=14, fontweight="bold", color="#C0392B")
cb.ax.text(0.5, -0.03, "", transform=cb.ax.transAxes, ha="center", va="top",
           fontsize=14, fontweight="bold", color="#2980B9")

# size legend = neighborhood size (n cells)
qs = np.quantile(SIZE_BY, [0.1, 0.5, 0.9]).round().astype(int)
handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.6", markeredgecolor="0.3",
                  markersize=np.sqrt(SMIN + (SMAX - SMIN) * (q / SIZE_BY.max())), label=str(q))
           for q in qs]
#ax.legend(handles=handles, title="Nhood size (cells)", loc="lower right", frameon=False,
#          fontsize=11, title_fontsize=12, labelspacing=1.4, borderpad=0.9, handletextpad=1.2)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / f"milo_nhood_graph{tag}.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / f"milo_nhood_graph{tag}.pdf", bbox_inches="tight", facecolor="white")
plt.show()


import numpy as np
from matplotlib.patches import Patch

# panels in order: Inf. Mono -> Recruited -> Resident (auto-match the real labels)
have = set(adata_mac.obs["mac_identity"].astype(str))
find = lambda k: next((h for h in sorted(have) if k in h.lower()), None)
identities = [x for x in [find("inflamm"), find("recruit"), find("resident")] if x]
print("panels:", identities)
BORDER_LW = 1.6

xy    = adata_mac.obsm["X_umap"]
ident = adata_mac.obs["mac_identity"].astype(str).values
subt  = adata_mac.obs["macrophage_subtypes"].astype(str).values
subtype_cats = list(adata_mac.obs["macrophage_subtypes"].cat.categories)

pad  = (xy[:, 0].max() - xy[:, 0].min()) * 0.03
xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)

fig, axes = plt.subplots(1, len(identities), figsize=(7 * len(identities), 6.5))
axes = np.atleast_1d(axes)
for ax, idn in zip(axes, identities):
    in_id = ident == idn
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="#E8E8E8", linewidths=0, rasterized=True)  # context
    for st in subtype_cats:
        m = in_id & (subt == st)
        ax.scatter(xy[m, 0], xy[m, 1], s=12, c=mac_colors.get(st, "#999999"),
                   linewidths=0, rasterized=True)
    ax.set_title(idn, fontsize=26, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=26, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=26, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(BORDER_LW)
        ax.spines[side].set_color("black")
    ax.set_aspect("equal", adjustable="box")

handles = [Patch(facecolor=mac_colors.get(st, "#999999"), label=st) for st in subtype_cats]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
           fontsize=15, frameon=False, title="Subtype", title_fontsize=16)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "umap_subtypes_split_inf_recruited_resident.png", dpi=600,
            bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "umap_subtypes_split_inf_recruited_resident.pdf",
            bbox_inches="tight", facecolor="white")
plt.show()


import numpy as np
from matplotlib.patches import Patch

have = set(adata_mac.obs["mac_identity"].astype(str))
find = lambda k: next((h for h in sorted(have) if k in h.lower()), None)
identities = [x for x in [find("inflamm"), find("resident")] if x]     # Inf. Mono + Resident
types = ["Sham", "Burn"]
TYPE_PAL, BORDER_LW = {"Sham": "#2980B9", "Burn": "#C0392B"}, 1.6

xy    = adata_mac.obsm["X_umap"]
ident = adata_mac.obs["mac_identity"].astype(str).values
typ   = adata_mac.obs["Type"].astype(str).values
subt  = adata_mac.obs["macrophage_subtypes"].astype(str).values
subtype_cats = list(adata_mac.obs["macrophage_subtypes"].cat.categories)

pad  = (xy[:, 0].max() - xy[:, 0].min()) * 0.03
xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad); ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)

fig, axes = plt.subplots(len(identities), len(types),
                         figsize=(7 * len(types), 6.5 * len(identities)), squeeze=False)
for ri, idn in enumerate(identities):
    for ci, ty in enumerate(types):
        ax = axes[ri][ci]; sel = (ident == idn) & (typ == ty)
        ax.scatter(xy[:, 0], xy[:, 1], s=4, c="#E8E8E8", linewidths=0, rasterized=True)   # context
        for st in subtype_cats:
            m = sel & (subt == st)
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=12, c=mac_colors.get(st, "#999999"),
                           linewidths=0, rasterized=True)
        if ri == 0: ax.set_title(ty, fontsize=26, fontweight="bold", color=TYPE_PAL[ty], pad=10)
        if ci == 0: ax.set_ylabel(idn, fontsize=19, fontweight="bold")
        ax.set_xlabel("UMAP 1" if ri == len(identities) - 1 else "", fontsize=19, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(xlim); ax.set_ylim(ylim)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True); ax.spines[side].set_linewidth(BORDER_LW); ax.spines[side].set_color("black")
        ax.set_aspect("equal", adjustable="box")

handles = [Patch(facecolor=mac_colors.get(st, "#999999"), label=st) for st in subtype_cats]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
           fontsize=15, frameon=False, title="Subtype", title_fontsize=16)
fig.tight_layout()
fig.savefig(FIGDIR_MAC / "umap_subtypes_inf_resident_split_burn_sham.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "umap_subtypes_inf_resident_split_burn_sham.pdf", bbox_inches="tight", facecolor="white")
plt.show()


import scanpy as sc, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import mannwhitneyu

TYPE_PAL, SUB_COL = {"Sham": "#2980B9", "Burn": "#C0392B"}, "macrophage_subtypes"
EARLY = next((s for s in adata_mac.obs[SUB_COL].astype(str).unique()
              if "early" in s.lower() and "mdm" in s.lower()), "Early MDM")

# programs (self-renewing/embryonic-resident TLF markers + resident TFs)
mono_genes = ["Ly6c2", "Ccr2", "Plac8", "Chil3", "Vcan"]
res_genes  = ["Timd4", "Lyve1", "Folr2", "Cd163", "Gas6", "Mrc1", "Maf", "Mafb"]
mono_genes = [g for g in mono_genes if g in adata_mac.var_names]
res_genes  = [g for g in res_genes  if g in adata_mac.var_names]
print("monocyte program:", mono_genes, "\nresident program:", res_genes)

sc.tl.score_genes(adata_mac, mono_genes, score_name="mono_score", use_raw=False)
sc.tl.score_genes(adata_mac, res_genes,  score_name="res_score",  use_raw=False)

em = adata_mac.obs.loc[adata_mac.obs[SUB_COL].astype(str) == EARLY,
                       ["Type", "mono_score", "res_score"]].copy()
em["diff"] = em["res_score"] - em["mono_score"]          # >0 = resident-biased
THR = 0.0
print(em.groupby("Type").size())
print("resident-biased fraction:\n", em.assign(rb=em["diff"] > THR).groupby("Type")["rb"].mean().round(3))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
lim = [em[["mono_score", "res_score"]].min().min(), em[["mono_score", "res_score"]].max().max()]
for ax, ty in zip(axes[:2], ["Sham", "Burn"]):
    d = em[em.Type == ty]
    ax.scatter(d["mono_score"], d["res_score"], s=10, alpha=0.4, color=TYPE_PAL[ty],
               edgecolor="none", rasterized=True)
    ax.plot(lim, lim, ls="--", color="0.5", lw=1.2)      # y=x : above = resident-biased
    ax.text(0.04, 0.96, f"{ty}\nresident-biased {100*(d['diff']>THR).mean():.1f}%\nn={len(d)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold", color=TYPE_PAL[ty])
    ax.set_xlabel("Monocyte program score", fontsize=15, fontweight="bold")
    ax.set_ylabel("Resident program score", fontsize=15, fontweight="bold")
    ax.set_xlim(lim); ax.set_ylim(lim)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

axC = axes[2]
for ty in ["Sham", "Burn"]:
    sns.kdeplot(em.loc[em.Type == ty, "diff"], ax=axC, color=TYPE_PAL[ty], fill=True,
                alpha=0.25, lw=3, label=ty)
axC.axvline(THR, ls="--", color="0.5")
p = mannwhitneyu(em.loc[em.Type == "Burn", "diff"], em.loc[em.Type == "Sham", "diff"]).pvalue
axC.set_title(f"Resident bias within Early MDM  (MWU p={p:.1e})", fontsize=13, fontweight="bold")
axC.set_xlabel("Resident − Monocyte score", fontsize=15, fontweight="bold")
axC.legend(frameon=False, fontsize=13, title="Condition")
for s in ("top", "right"): axC.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIGDIR_MAC / "earlyMDM_lineage_score_burn_vs_sham.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "earlyMDM_lineage_score_burn_vs_sham.pdf", bbox_inches="tight", facecolor="white")
plt.show()


import scanpy as sc, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import mannwhitneyu, rankdata

TYPE_PAL, SUB_COL = {"Sham": "#2980B9", "Burn": "#C0392B"}, "macrophage_subtypes"
EARLY = next((s for s in adata_mac.obs[SUB_COL].astype(str).unique()
              if "early" in s.lower() and "mdm" in s.lower()), "Early MDM")

# three orthogonal programs: monocyte-immaturity vs inflammation vs maturation
signatures = {
  "Classical monocyte":      ["Ly6c2","Ly6c1","Ccr2","Plac8","Chil3","Vcan","Sell","F13a1","Hp","Gngt2"],
  "Inflammatory activation": ["Il1b","Tnf","Nos2","Ptgs2","Ccl2","Ccl3","Ccl4","Cxcl2","Cxcl3","Il1a","Nfkbia"],
  "Mature MDM/macrophage":   ["Cebpb","Mafb","Adgre1","C1qa","C1qb","Apoe","Mrc1","Fcgr1","Csf1r"],
}
for name, gl in signatures.items():
    gl = [g for g in gl if g in adata_mac.var_names]
    sc.tl.score_genes(adata_mac, gl, score_name=name, use_raw=False)
    print(f"{name:24s} -> {gl}")

em = adata_mac.obs.loc[adata_mac.obs[SUB_COL].astype(str) == EARLY, ["Type"] + list(signatures)].copy()

def cliffs(a, b):                                  # effect size, not n-inflated p
    a, b = np.asarray(a), np.asarray(b); n1, n2 = len(a), len(b)
    U = rankdata(np.concatenate([a, b]))[:n1].sum() - n1*(n1+1)/2
    return 2*U/(n1*n2) - 1
print(f"\n{EARLY}: Burn vs Sham (+ = up in Burn)")
for name in signatures:
    b, s = em.loc[em.Type=="Burn", name], em.loc[em.Type=="Sham", name]
    print(f"  {name:24s} Δmedian={b.median()-s.median():+.3f}  Cliff δ={cliffs(b,s):+.2f}  p={mannwhitneyu(b,s).pvalue:.1e}")

# transcriptomic proximity to the Inf. Mono population (PCA space): closer = more monocyte-like
rep = "X_pca_harmony" if "X_pca_harmony" in adata_mac.obsm else "X_pca"
P = adata_mac.obsm[rep]
inf_c = P[adata_mac.obs["mac_identity"].astype(str).str.contains("Inflamm").values].mean(0)
emm = (adata_mac.obs[SUB_COL].astype(str) == EARLY).values
dd = pd.DataFrame({"Type": adata_mac.obs["Type"].astype(str).values[emm],
                   "dist": np.linalg.norm(P[emm] - inf_c, axis=1)})
b, s = dd.loc[dd.Type=="Burn","dist"], dd.loc[dd.Type=="Sham","dist"]
print(f"\nDistance to Inf.Mono centroid (smaller=more monocyte-like): "
      f"Burn={b.median():.2f}  Sham={s.median():.2f}  δ={cliffs(b,s):+.2f}  p={mannwhitneyu(b,s).pvalue:.1e}")

# violins
fig, axes = plt.subplots(1, len(signatures), figsize=(5*len(signatures), 5))
for ax, name in zip(axes, signatures):
    sns.violinplot(data=em, x="Type", y=name, order=["Sham","Burn"], hue="Type",
                   palette=TYPE_PAL, legend=False, cut=0, inner="quartile", ax=ax)
    ax.set_title(name, fontsize=14, fontweight="bold"); ax.set_xlabel("")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.suptitle(f"{EARLY}: program scores, Burn vs Sham", fontsize=16, fontweight="bold", y=1.03)
fig.tight_layout(); plt.show()


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

id_palette = {
    "Inflammatory Monocytes":      "#D62728",   # red
    "Recruited Macrophages":  "#F39C12",   # orange
    "Resident Macrophages":   "#2CA02C",   # green
}

xy       = adata_mac.obsm["X_umap"]
mac_id   = adata_mac.obs["mac_identity"].astype(str).values
type_ser = adata_mac.obs["Type"].astype(str)

if str(adata_mac.obs["Type"].dtype) == "category":
    types = list(adata_mac.obs["Type"].cat.categories)
else:
    types = ["Sham", "Burn"] if {"Sham", "Burn"} <= set(type_ser.unique()) else sorted(type_ser.unique())
type_arr = type_ser.values

pad  = (xy[:, 0].max() - xy[:, 0].min()) * 0.03
xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)

fig, axes = plt.subplots(1, len(types), figsize=(10, 5))
axes = np.atleast_1d(axes)
for ax, t in zip(axes, types):
    in_t = type_arr == t
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="#E8E8E8", linewidths=0, rasterized=True)   # all cells grey
    for lab, col in id_palette.items():
        m = in_t & (mac_id == lab)
        ax.scatter(xy[m, 0], xy[m, 1], s=10, c=col, linewidths=0, rasterized=True)
    ax.set_title(t, fontsize=35, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=35, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=35, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

handles = [Line2D([0], [0], marker='o', linestyle='', markersize=13,
                  markerfacecolor=c, markeredgecolor='none', label=l)
           for l, c in id_palette.items()]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
           fontsize=17, frameon=False, title="Identity", title_fontsize=18)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "umap_mac_identity_split_by_type.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "umap_mac_identity_split_by_type.pdf", bbox_inches="tight")
plt.show()


import pandas as pd
from matplotlib.ticker import PercentFormatter

TWO = ["Recruited Macrophages", "Resident Macrophages"]
short = {"Recruited Macrophages": "Recruited", "Resident Macrophages": "Resident"}

# auto-detect the subtype column from mac_colors keys (excluding mac_identity itself)
def _ov(c):
    try:
        return len(set(adata_mac.obs[c].astype(str).unique()) & set(mac_colors.keys()))
    except Exception:
        return 0
subtype_col = max([c for c in adata_mac.obs.columns if c != "mac_identity"], key=_ov)
assert _ov(subtype_col) > 0, "No obs column matches mac_colors keys — set subtype_col manually."
print(f"Subtype column: {subtype_col!r}")

d = adata_mac.obs[adata_mac.obs["mac_identity"].isin(TWO)][["mac_identity", subtype_col]].copy()
prop2 = pd.crosstab(d["mac_identity"], d[subtype_col], normalize="index").reindex(TWO)

# column order follows mac_colors; any extras appended
subtypes = [s for s in mac_colors.keys() if s in prop2.columns]
subtypes += [s for s in prop2.columns if s not in subtypes]
prop2 = prop2[subtypes]
print(prop2.round(3))

fig, ax = plt.subplots(figsize=(7, 6))
x = np.arange(len(prop2.index)) * 0.6           # bring the 2 bars closer
bottom = np.zeros(len(prop2.index))
for st in subtypes:
    ax.bar(x, prop2[st].values, bottom=bottom, width=0.45,
           color=mac_colors.get(st, "#CCCCCC"), edgecolor="white", linewidth=0.8, label=st)
    bottom += prop2[st].values



ax.set_xticks(x)
ax.set_xticklabels([short.get(s, s) for s in prop2.index], fontsize=24, fontweight="bold", rotation = 45)
ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
ax.set_ylabel("Proportion of cells", fontsize=24, fontweight="bold", labelpad=8)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.tick_params(axis="y", labelsize=18, width=1.5, length=6)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(title="Macrophage subtype", fontsize=13, title_fontsize=14,
          loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "subtype_composition_recruited_vs_resident.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "subtype_composition_recruited_vs_resident.pdf", bbox_inches="tight")
plt.show()


import pandas as pd
import numpy as np

# three identities in order: Inf. Mono -> Recruited -> Resident (auto-match real labels)
have = set(adata_mac.obs["mac_identity"].astype(str))
find = lambda k: next((h for h in sorted(have) if k in h.lower()), None)
GROUPS = [x for x in [find("inflamm"), find("recruit"), find("resident")] if x]
type_order = ["Sham", "Burn"]

def shorten(s):
    sl = s.lower()
    return ("Inf. Mono." if "inflamm" in sl else
            "Recruited"  if "recruit" in sl else
            "Resident"   if "resident" in sl else s)
short = {g: shorten(g) for g in GROUPS}
print("groups:", GROUPS)

# subtype column (auto-detected from mac_colors keys)
def _ov(c):
    try:
        return len(set(adata_mac.obs[c].astype(str).unique()) & set(mac_colors.keys()))
    except Exception:
        return 0
subtype_col = max([c for c in adata_mac.obs.columns if c != "mac_identity"], key=_ov)
assert _ov(subtype_col) > 0, "No obs column matches mac_colors keys — set subtype_col manually."
print(f"Subtype column: {subtype_col!r}")

# RAW counts of subtypes within each (identity, Type)
d = adata_mac.obs[adata_mac.obs["mac_identity"].isin(GROUPS)][["mac_identity", "Type", subtype_col]].copy()
d["Type"] = d["Type"].astype(str)
ct = pd.crosstab([d["mac_identity"], d["Type"]], d[subtype_col])
row_order = [(idn, tp) for idn in GROUPS for tp in type_order if (idn, tp) in ct.index]
ct = ct.reindex(row_order)

subtypes = [s for s in mac_colors.keys() if s in ct.columns]
subtypes += [s for s in ct.columns if s not in subtypes]
ct = ct[subtypes]
print(ct)

# ── grouped x positions: Sham|Burn within each identity, gap between identities ─
within, gap, bar_w = 0.5, 0.55, 0.44
positions, group_xs = [], {}
x = 0.0; prev = None
for (idn, tp) in row_order:
    if prev is not None and idn != prev:
        x += gap
    positions.append(x); group_xs.setdefault(idn, []).append(x)
    x += within; prev = idn
positions = np.array(positions)

fig, ax = plt.subplots(figsize=(12, 7.5))          # wider for 6 bars
bottom = np.zeros(len(row_order))
for st in subtypes:
    ax.bar(positions, ct[st].values, bottom=bottom, width=bar_w,
           color=mac_colors.get(st, "#CCCCCC"), edgecolor="white", linewidth=0.8, label=st)
    bottom += ct[st].values

ax.set_xticks(positions)
ax.set_xticklabels([tp for (idn, tp) in row_order], fontsize=25, fontweight="bold",
                   rotation=45, ha="right", rotation_mode="anchor")
ax.tick_params(axis="x", length=0)

for idn in GROUPS:
    xc = float(np.mean(group_xs[idn]))
    ax.text(xc, -0.22, short[idn], transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=28, fontweight="bold")

ax.set_xlim(positions[0] - 0.45, positions[-1] + 0.45)
ax.set_ylabel("Number of cells", fontsize=35, fontweight="bold", labelpad=8)
ax.tick_params(axis="y", labelsize=22, width=1.5, length=6)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(title="Macrophage subtype", fontsize=0, title_fontsize=14,
          loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "subtype_counts_3identities_by_type.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "subtype_counts_3identities_by_type.pdf", bbox_inches="tight")
plt.show()


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

TWO = ["Recruited Macrophages", "Resident Macrophages"]
type_order = ["Sham", "Burn"]
short = {"Recruited Macrophages": "Recruited", "Resident Macrophages": "Resident"}

# subtype column (auto-detected from mac_colors keys)
def _ov(c):
    try:
        return len(set(adata_mac.obs[c].astype(str).unique()) & set(mac_colors.keys()))
    except Exception:
        return 0
subtype_col = max([c for c in adata_mac.obs.columns if c != "mac_identity"], key=_ov)
assert _ov(subtype_col) > 0, "No obs column matches mac_colors keys — set subtype_col manually."
print(f"Subtype column: {subtype_col!r}")

# ── data: panel A = proportions by identity; panel B = counts by identity × Type ─
base = adata_mac.obs[adata_mac.obs["mac_identity"].isin(TWO)].copy()
base["Type"] = base["Type"].astype(str)

propA = pd.crosstab(base["mac_identity"], base[subtype_col], normalize="index").reindex(TWO)
ctB   = pd.crosstab([base["mac_identity"], base["Type"]], base[subtype_col])
row_order = [(idn, tp) for idn in TWO for tp in type_order if (idn, tp) in ctB.index]
ctB = ctB.reindex(row_order)

subtypes = [s for s in mac_colors.keys() if s in propA.columns]
subtypes += [s for s in propA.columns if s not in subtypes]
propA = propA[subtypes]
ctB   = ctB[[s for s in subtypes if s in ctB.columns]]
print(propA.round(3)); print(ctB)

# ── figure ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 7.5),
                               gridspec_kw={"width_ratios": [1, 1.4]})

# ===== Panel A: proportions, Recruited vs Resident =====
xA = np.arange(len(propA.index)) * 0.6
bottom = np.zeros(len(propA.index))
for st in subtypes:
    axA.bar(xA, propA[st].values, bottom=bottom, width=0.45,
            color=mac_colors.get(st, "#CCCCCC"), edgecolor="white", linewidth=0.8)
    bottom += propA[st].values
axA.set_xticks(xA)
axA.set_xticklabels([short.get(s, s) for s in propA.index], fontsize=22, fontweight="bold",
                    rotation=45, ha="right", rotation_mode="anchor")
axA.set_xlim(xA[0] - 0.4, xA[-1] + 0.4)
axA.set_ylim(0, 1)
axA.yaxis.set_major_formatter(PercentFormatter(1.0))
#axA.set_ylabel("Proportion of cells", fontsize=24, fontweight="bold", labelpad=8)
axA.tick_params(axis="y", labelsize=18, width=1.5, length=6)
axA.tick_params(axis="x", length=0)
axA.set_title("", fontsize=22, fontweight="bold", pad=10)
for s in ["top", "right"]:
    axA.spines[s].set_visible(False)

# ===== Panel B: counts, Recruited/Resident split by Sham/Burn =====
within, gap, bar_w = 0.5, 0.55, 0.44
positions, group_xs = [], {}
x = 0.0; prev = None
for (idn, tp) in row_order:
    if prev is not None and idn != prev:
        x += gap
    positions.append(x); group_xs.setdefault(idn, []).append(x)
    x += within; prev = idn
positions = np.array(positions)

bottom = np.zeros(len(row_order))
for st in subtypes:
    if st not in ctB.columns:
        continue
    axB.bar(positions, ctB[st].values, bottom=bottom, width=bar_w,
            color=mac_colors.get(st, "#CCCCCC"), edgecolor="white", linewidth=0.8)
    bottom += ctB[st].values
axB.set_xticks(positions)
axB.set_xticklabels([tp for (idn, tp) in row_order], fontsize=20, fontweight="bold",
                    rotation=45, ha="right", rotation_mode="anchor")
axB.tick_params(axis="x", length=0)
for idn in TWO:
    xc = float(np.mean(group_xs[idn]))
    axB.text(xc, -0.20, short[idn], transform=axB.get_xaxis_transform(),
             ha="center", va="top", fontsize=24, fontweight="bold")
axB.set_xlim(positions[0] - 0.45, positions[-1] + 0.45)
axB.set_ylabel("Number of cells", fontsize=24, fontweight="bold", labelpad=8)
axB.tick_params(axis="y", labelsize=18, width=1.5, length=6)
#axB.set_title("Cell number by condition", fontsize=22, fontweight="bold", pad=10)
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)

# ===== shared subtype legend =====
handles = [Patch(facecolor=mac_colors.get(st, "#CCCCCC"), label=st) for st in subtypes]
fig.legend(handles=handles, title="Macrophage subtype", fontsize=13, title_fontsize=15,
           loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "panel_subtype_composition_and_counts.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "panel_subtype_composition_and_counts.pdf", bbox_inches="tight")
plt.show()


import numpy as np
import scipy.sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)

genes = ["Arg1", "Nos2"]
CMAP, NA_GREY = "Reds", "#E8E8E8"

# columns = the three identities (auto-match real labels): Inf. Mono -> Recruited -> Resident
have = set(adata_mac.obs["mac_identity"].astype(str))
find = lambda k: next((h for h in sorted(have) if k in h.lower()), None)
idents = [x for x in [find("inflamm"), find("recruit"), find("resident")] if x]
print("columns:", idents)

missing = [g for g in genes if g not in adata_mac.var_names]
assert not missing, f"Genes not found in adata_mac: {missing}"

xy    = adata_mac.obsm["X_umap"]
ident = adata_mac.obs["mac_identity"].astype(str).values

def expr_of(g):
    x = adata_mac[:, g].X
    return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()

mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

pad  = (xy[:, 0].max() - xy[:, 0].min()) * 0.03
xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)

nrow, ncol = len(genes), len(idents)
fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.9 * nrow))
axes = np.atleast_2d(axes)

for r, g in enumerate(genes):
    e = expr_of(g)
    vmax = np.percentile(e[e > 0], 99) if (e > 0).any() else 1.0   # shared scale across the columns
    norm = Normalize(vmin=0, vmax=vmax)
    sc = None
    for c, idn in enumerate(idents):
        ax = axes[r, c]
        m = ident == idn                                           # <- identity only, no Type
        ax.scatter(xy[:, 0], xy[:, 1], s=4, c=NA_GREY, linewidths=0, rasterized=True)  # context
        order = np.argsort(e[m])                                                        # high on top
        sc = ax.scatter(xy[m, 0][order], xy[m, 1][order], s=12, c=e[m][order],
                        cmap=CMAP, norm=norm, linewidths=0, rasterized=True)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(idn, fontsize=28, fontweight="bold", pad=14)
        if c == 0:
            ax.set_ylabel(g, fontsize=35, fontweight="bold", fontstyle="italic", labelpad=14)
    cbar = fig.colorbar(sc, ax=axes[r, :].tolist(), fraction=0.022, pad=0.012)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(f"{g} Expr.", fontsize=25, fontweight="bold")

fig.savefig(FIGDIR_MAC / "featureplot_arg1_nos2_by_identity.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "featureplot_arg1_nos2_by_identity.pdf", bbox_inches="tight")
plt.show()

