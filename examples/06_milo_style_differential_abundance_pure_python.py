"""Milo-style differential abundance (pure Python) — Burn vs Sham macrophages

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, numpy, pathlib, scanpy, scipy, statsmodels
Key calls: def _style, plt.cm, plt.get_cmap, plt.show, plt.subplots, sc.pp.neighbors, scatter
"""

# ══════════════════════════════════════════════════════════════════════════════
# Milo-style differential abundance (pure Python) — Burn vs Sham macrophages
#   Blocked design ~ C(Timepoint) + Type ; test Type (Burn vs Sham).
#   Quasi-Poisson GLM w/ shared dispersion + BH FDR ; beeswarm + nhood UMAP graph.
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib as mpl, matplotlib.pyplot as plt
import scanpy as sc, statsmodels.api as sm
from scipy.stats import norm as _norm
from statsmodels.stats.multitest import multipletests
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ── params ──────────────────────────────────────────────────────────────────────
SUB_COL, TYPE_COL = "macrophage_subtypes", "Type"
PROP, MIX_THRESH, ALPHA_FDR, SEED = 0.10, 0.70, 0.10, 0
STRATIFY_IDENTITY = None     # e.g. "Inflammatory Monocytes" to run DA within one mac_identity; None = all
FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
PUB = globals().get("PUB", dict(axis_label_fs=24, border_lw=1.6))
rng = np.random.default_rng(SEED)

ad = adata_mac if STRATIFY_IDENTITY is None else \
     adata_mac[adata_mac.obs["mac_identity"].astype(str) == STRATIFY_IDENTITY].copy()
tag = "" if STRATIFY_IDENTITY is None else f"_{STRATIFY_IDENTITY.split()[0].lower()}"

# ── replicate + timepoint columns ───────────────────────────────────────────────
SAMP_CANDS = ["Sample","sample","orig.ident","orig_ident","SampleID","sample_id",
              "library","Library","mouse","Mouse","replicate","Replicate"]
samp_col = next((c for c in SAMP_CANDS if c in ad.obs.columns), None)
assert samp_col, f"Set samp_col manually from {list(ad.obs.columns)}"
tp_series = (ad.obs["Timepoint"].astype(str) if "Timepoint" in ad.obs
             else ad.obs["Type_Timepoint_C"].astype(str).str.split().str[-1])
print(f"replicate = {samp_col!r} | {ad.obs[samp_col].nunique()} samples")

# ── KNN graph ────────────────────────────────────────────────────────────────────
rep = next((r for r in ["X_pca_harmony","X_pca","X_umap"] if r in ad.obsm), None)
if "distances" not in ad.obsp:
    sc.pp.neighbors(ad, use_rep=rep, n_neighbors=30)
#A = (ad.obsp["distances"] > 0).astype(np.int8).tolil(); A.setdiag(1); A = A.tocsr()
A = (ad.obsp["distances"] > 0).astype(np.int8)
A = (A + A.T).tolil(); A.setdiag(1)          # union of directed kNN → undirected
A = (A.tocsr() > 0).astype(np.int8)          # binarize (2's where mutual)

R = ad.obsm[rep]

# ── make_nhoods ──────────────────────────────────────────────────────────────────
cand = rng.choice(ad.n_obs, size=max(1, int(PROP * ad.n_obs)), replace=False)
idx = []
for c in cand:
    nb = A[c].indices
    idx.append(int(nb[np.argmin(((R[nb] - R[nb].mean(0))**2).sum(1))]) if nb.size else int(c))
idx = np.unique(idx)
N = A[idx].T.tocsr(); nh_size = np.asarray(N.sum(0)).ravel()
print(f"{len(idx)} neighborhoods (median size {np.median(nh_size):.0f})")

# ── counts per neighborhood per sample ───────────────────────────────────────────
samp = ad.obs[samp_col].astype(str).values
samples = sorted(np.unique(samp))
S = sp.csr_matrix((np.ones(ad.n_obs), (np.arange(ad.n_obs), [samples.index(s) for s in samp])),
                  shape=(ad.n_obs, len(samples)))
counts = (N.T @ S).toarray()
cond_of = np.array([ad.obs.loc[samp == s, TYPE_COL].astype(str).mode().iloc[0] for s in samples])
tp_of   = np.array([tp_series[samp == s].mode().iloc[0] for s in samples])
keep_s  = counts.sum(0) > 0
counts, burn, tp_keep = counts[:, keep_s], (cond_of[keep_s] == "Burn").astype(float), tp_of[keep_s]
assert burn.min() == 0 and burn.max() == 1, "need both Burn and Sham samples"
offset = np.log(counts.sum(0))

# ── design: ~ C(Timepoint) + Burn  (block on timepoint, test Type) ───────────────
tp_levels = sorted(set(tp_keep), key=lambda t: int("".join(filter(str.isdigit, t)) or 0))
Dtp = pd.get_dummies(pd.Categorical(tp_keep, categories=tp_levels), drop_first=True).to_numpy(float)
X = np.column_stack([np.ones(burn.size), Dtp, burn]); bcol = X.shape[1] - 1
if np.linalg.matrix_rank(X) < X.shape[1]:                    # timepoint × type collinear → drop block
    X = np.column_stack([np.ones(burn.size), burn]); bcol = 1
    print("⚠ Timepoint collinear with Type — falling back to ~ Type")
else:
    print(f"design: ~ C(Timepoint){tp_levels} + Type   (n_samples={burn.size})")

# ── per-neighborhood quasi-Poisson GLM, shared dispersion ───────────────────────
coef = np.full(len(idx), np.nan); se1 = np.full(len(idx), np.nan); phi = np.full(len(idx), np.nan)
for j in range(len(idx)):
    if counts[j].sum() < 5: continue
    try:
        r = sm.GLM(counts[j], X, family=sm.families.Poisson(), offset=offset).fit()
        coef[j] = r.params[bcol]; se1[j] = np.sqrt(r.cov_params()[bcol, bcol])
        phi[j]  = r.pearson_chi2 / max(r.df_resid, 1)
    except Exception:
        pass
ok = np.isfinite(coef) & np.isfinite(se1) & (se1 > 0)
phi_common = max(1.0, float(np.nanmedian(phi[ok])))          # shrink to common dispersion
z = coef / (se1 * np.sqrt(phi_common)); lfc = coef / np.log(2)
pval = np.ones(len(idx)); pval[ok] = 2 * _norm.sf(np.abs(z[ok]))
fdr  = np.ones(len(idx)); fdr[ok] = multipletests(pval[ok], method="fdr_bh")[1]
sig  = ok & (fdr < ALPHA_FDR)
print(f"dispersion φ={phi_common:.2f} | significant: {int(sig.sum())} "
      f"(Burn-up {int((sig&(lfc>0)).sum())}, Sham-up {int((sig&(lfc<0)).sum())})")

# ── annotate neighborhoods by majority macrophage_subtype ────────────────────────
subs = ad.obs[SUB_COL].astype(str).values
subcats = [s for s in mac_colors if s in set(subs)] + [s for s in sorted(set(subs)) if s not in mac_colors]
SUB1 = sp.csr_matrix((np.ones(ad.n_obs), (np.arange(ad.n_obs), [subcats.index(s) for s in subs])),
                     shape=(ad.n_obs, len(subcats)))
nsub = (N.T @ SUB1).toarray(); frac = nsub.max(1) / np.maximum(nsub.sum(1), 1)
annot = np.array([subcats[i] for i in nsub.argmax(1)], dtype=object); annot[frac < MIX_THRESH] = "Mixed"

# ── robust color / axis scale (ignore separation outliers) ───────────────────────
dmax = float(np.nanpercentile(np.abs(lfc[ok]), 98)) or 1.0
norm, cmap = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax), plt.get_cmap("RdBu_r")
clip = lambda v: np.clip(v, -dmax, dmax)
def _style(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_linewidth(PUB["border_lw"])

# ══ FIGURE 1 — beeswarm ══════════════════════════════════════════════════════════
#groups = [g for g in subcats if (annot == g).any()] + (["Mixed"] if (annot == "Mixed").any() else [])
groups = [g for g in subcats if (annot == g).any()]        # exclude "Mixed"
groups = sorted(groups, key=lambda g: np.nanmedian(lfc[(annot == g) & ok]) if ((annot == g) & ok).any() else 0)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.axvline(0, color="0.4", lw=1.4, ls="--", zorder=1)
for i, g in enumerate(groups):
    ns = (annot == g) & ok & ~sig; ss = (annot == g) & sig
    ax.scatter(clip(lfc[ns]), i + rng.uniform(-.34, .34, ns.sum()), s=18, color="0.82", zorder=2)
    ax.scatter(clip(lfc[ss]), i + rng.uniform(-.34, .34, ss.sum()), s=34, c=clip(lfc[ss]),
               cmap=cmap, norm=norm, edgecolor="black", linewidth=0.3, zorder=3)
ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=17, fontweight="bold")
ax.set_xlim(-dmax * 1.15, dmax * 1.15)
ax.set_xlabel("log$_2$FC  (Sham ← → Burn)", fontsize=16, fontweight="bold")
ax.set_title(f"Differential abundance (SpatialFDR<{ALPHA_FDR})", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", labelsize=17); _style(ax)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.02)
cb.set_ticks([]); cb.outline.set_linewidth(0.8)              # drop numeric ticks
cb.ax.text(0.5, 1.03, "Burn", transform=cb.ax.transAxes, ha="center", va="bottom",
           fontsize=15, fontweight="bold", color="#C0392B")   # red (top) end
cb.ax.text(0.5, -0.03, "Sham", transform=cb.ax.transAxes, ha="center", va="top",
           fontsize=15, fontweight="bold", color="#2980B9")   # blue (bottom) end

fig.tight_layout()
fig.savefig(FIGDIR_MAC / f"milo_beeswarm_burn_vs_sham{tag}.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / f"milo_beeswarm_burn_vs_sham{tag}.pdf", bbox_inches="tight", facecolor="white")
plt.show()
