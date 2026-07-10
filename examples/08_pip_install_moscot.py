"""pip install moscot

Source: macrophages_resident_recruited.ipynb
Libraries: adjustText, cellrank, matplotlib, moscot, numpy, pathlib, scipy
Key calls: adjust_text, cellrank.estimators, cellrank.kernels, plt.show, plt.subplots, umap
"""

# pip install moscot
import moscot as mt
from moscot.problems.time import TemporalProblem

# numeric time axis (D7→7, D10→10, …)
adata_mac.obs["time_num"] = adata_mac.obs["Timepoint"].astype(str).str.extract(r"(\d+)").astype(float)

transitions = {}
for cond in ["Sham", "Burn"]:
    ad = adata_mac[adata_mac.obs["Type"] == cond].copy()
    tp = TemporalProblem(ad).prepare(time_key="time_num")     # uses PCA joint space by default
    tp = tp.solve(epsilon=1e-2, tau_a=0.97, tau_b=0.97)       # unbalanced -> allows growth/influx/death

    # consecutive-timepoint cluster transition matrices
    tps = sorted(ad.obs["time_num"].unique())
    for s, t in zip(tps[:-1], tps[1:]):
        M = tp.cell_transition(
            source=s, target=t,
            source_groups="mac_identity",     # or "macrophage_subtypes"
            target_groups="mac_identity",
            forward=True,                      # rows = source mass distributed to target
        )
        transitions[(cond, s, t)] = M
        print(f"\n{cond}  {int(s)}→{int(t)}"); print(M.round(2))


import numpy as np
import matplotlib.pyplot as plt

SHORT = {"Inflammatory Monocytes": "Inf.\nMono.",
         "Recruited Macrophages":  "Recr.",
         "Resident Macrophages":   "Resid."}
conds = ["Sham", "Burn"]
pairs = sorted({(s, t) for (_, s, t) in transitions}, key=lambda p: p[0])

fig, axes = plt.subplots(len(conds), len(pairs),
                         figsize=(3.6 * len(pairs), 3.6 * len(conds)))
axes = np.atleast_2d(axes)
im = None
for i, cond in enumerate(conds):
    for j, (s, t) in enumerate(pairs):
        ax = axes[i, j]
        M = transitions[(cond, s, t)]
        im = ax.imshow(M.values, cmap="magma_r", vmin=0, vmax=1)
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                v = M.values[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.55 else "black",
                        fontsize=13, fontweight="bold")
        ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
        ax.set_xticklabels([SHORT.get(c, c) for c in M.columns], fontsize=11)
        ax.set_yticklabels([SHORT.get(r, r) for r in M.index], fontsize=11)
        if i == 0: ax.set_title(f"{int(s)}→{int(t)}", fontsize=17, fontweight="bold")
        if j == 0: ax.set_ylabel(cond, fontsize=20, fontweight="bold", labelpad=12)

fig.suptitle("Forward OT transition probability  (row = from, col = to)",
             fontsize=18, fontweight="bold", y=1.02)
cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
cbar.set_label("transition prob.", fontsize=14, fontweight="bold")
fig.savefig(FIGDIR_MAC / "ot_pool_transitions_heatmap.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "ot_pool_transitions_heatmap.pdf", bbox_inches="tight")
plt.show()


from moscot.problems.time import TemporalProblem
adata_mac.obs["time_num"] = adata_mac.obs["Timepoint"].astype(str).str.extract(r"(\d+)").astype(float)

problems = {}
for cond in ["Sham", "Burn"]:
    ad = adata_mac[adata_mac.obs["Type"] == cond].copy()
    problems[cond] = (TemporalProblem(ad).prepare(time_key="time_num")
                                          .solve(epsilon=1e-2, tau_a=0.97, tau_b=0.97))


import numpy as np, pandas as pd
import cellrank as cr
from cellrank.kernels import RealTimeKernel
from cellrank.estimators import GPCCA

drivers, fate = {}, {}
for cond in ["Sham", "Burn"]:
    tmk = RealTimeKernel.from_moscot(problems[cond])
    tmk.compute_transition_matrix(self_transitions="all", conn_weight=0.2, threshold="auto")

    g = GPCCA(tmk)
    g.compute_macrostates(n_states=6, cluster_key="macrophage_subtypes")
    macs = list(g.macrostates.cat.categories)
    print(cond, "macrostates:", macs)

    # choose terminal ENDPOINTS explicitly (need >1 so fate probs are a real competition)
    term = [s for s in macs if any(k in s for k in ["Res", "Act", "IFN", "LAM", "Rep"])]
    if not term:
        term = macs                      # fallback: treat all macrostates as endpoints
    g.set_terminal_states(term)
    g.compute_fate_probabilities()

    res_name = next((s for s in g.terminal_states.cat.categories if "Res" in s), None)
    if res_name:
        fp = g.fate_probabilities
        fate[cond] = pd.Series(np.asarray(fp[res_name].X).ravel(),
                               index=g.adata.obs_names, name=f"{cond}_resident_fate")
        drivers[cond] = g.compute_lineage_drivers(lineages=res_name,
                                                  cluster_key="macrophage_subtypes")
    else:
        print(f"  ⚠ no resident-like macrostate found in {cond} — check `macs` above")
    g.plot_fate_probabilities(same_plot=False)


import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)

xy   = adata_mac.obsm["X_umap"]
typ  = adata_mac.obs["Type"].astype(str).values
subt = adata_mac.obs["macrophage_subtypes"].astype(str).values
sub_cats = [s for s in adata_mac.obs["macrophage_subtypes"].cat.categories
            if (subt == s).any()]

# subtype label positions = median UMAP coord (same on both panels)
centroids = {s: np.median(xy[subt == s], axis=0) for s in sub_cats}

# evaluation grid over the full embedding
pad = 0.6
xmin, xmax = xy[:, 0].min() - pad, xy[:, 0].max() + pad
ymin, ymax = xy[:, 1].min() - pad, xy[:, 1].max() + pad
xx, yy = np.mgrid[xmin:xmax:220j, ymin:ymax:220j]
grid = np.vstack([xx.ravel(), yy.ravel()])

conds = ["Sham", "Burn"]
title_color = {"Sham": "#3B5BDB", "Burn": "#E8412A"}

# per-condition KDE
dens = {}
for cond in conds:
    pts = xy[typ == cond].T
    dens[cond] = gaussian_kde(pts)(grid).reshape(xx.shape)

# SHARED = True -> same color scale across panels (quantitative); False -> per-panel (matches the look)
SHARED = False
vmax = max(d.max() for d in dens.values()) if SHARED else None
from adjustText import adjust_text

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
for ax, cond in zip(axes, conds):
    z = dens[cond]
    levels = np.linspace(0, vmax if SHARED else z.max(), 100)
    ax.contourf(xx, yy, z, levels=levels, cmap="viridis")
    ax.set_title(cond, fontsize=30, fontweight="bold", color=title_color[cond], pad=12)
    ax.set_xlabel("UMAP 1", fontsize=24, fontweight="bold")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.tick_params(labelsize=18)
    axes[0].set_ylabel("UMAP 2", fontsize=28, fontweight="bold")
    ax.grid(False)

box   = dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9)
arrow = dict(arrowstyle="-", color="white", lw=1.0, alpha=0.9)

# 1) de-overlap on the LEFT panel
texts = [axes[0].text(cx, cy, s, fontsize=18, fontweight="bold",
                      ha="center", va="center", bbox=box)
         for s, (cx, cy) in centroids.items()]

# 2) mirror the resolved positions onto the RIGHT panel (identical placement)
final_pos = {t.get_text(): t.get_position() for t in texts}
for s, (cx, cy) in centroids.items():
    tx, ty = final_pos[s]
    axes[1].annotate(s, xy=(cx, cy), xytext=(tx, ty), ha="center", va="center",
                     fontsize=18, fontweight="bold", bbox=box, arrowprops=arrow)



fig.tight_layout()
fig.savefig(FIGDIR_MAC / "umap_density_burn_vs_sham.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "umap_density_burn_vs_sham.pdf", bbox_inches="tight")
plt.show()

