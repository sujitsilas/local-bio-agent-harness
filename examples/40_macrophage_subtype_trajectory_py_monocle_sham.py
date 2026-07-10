"""Macrophage subtype trajectory (py-monocle) — Sham vs Burn, rooted on Inf. Mono

Source: macrophages_resident_recruited.ipynb
Libraries: collections, matplotlib, numpy, pathlib, py_monocle, scanpy, scipy
Key calls: .plot, def _chaikin, def _chains_from_edges, def _fit_cond, def _p, def _pick_inf_root, def _star, def _top, def _vec, def fit_trajectory, def plot_trajectory, dendrogram
"""

# ══════════════════════════════════════════════════════════════════════════════
# Macrophage subtype trajectory (py-monocle) — Sham vs Burn, rooted on Inf. Mono.
#   ONE CELL: fits the principal graph on macrophage_subtypes, anchors the root at
#   a PURE Inf. Mono. tip, and plots the UMAP colored by macrophage_subtypes
#   (smooth trajectory, boxed subtype labels, prominent root star).
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patheffects as pe
import scipy.sparse as sp
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from pathlib import Path
from scipy.spatial.distance import cdist
try:
    from py_monocle import learn_graph, order_cells
except ImportError as e:
    raise ImportError("Run:  pip install git+https://github.com/bioturing/py-monocle.git") from e

# ── config ────────────────────────────────────────────────────────────────────
P_THRESHOLD, N_CENTROIDS = 14, 20
FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)
TYPE_PAL = {"Sham": "#2980B9", "Burn": "#C0392B"}
conds    = ["Sham", "Burn"]

# ── macrophage_subtypes drive everything (clusters, node dominance, root, colors) ─
ident_all = adata_mac.obs["macrophage_subtypes"].astype(str)
sub_all   = adata_mac.obs["macrophage_subtypes"].astype(str)
_have  = set(ident_all.unique())
_find  = lambda *ks: next((h for h in sorted(_have) if any(k in h.lower() for k in ks)), None)
INF    = _find("inf. mono", "mono")        # "Inf. Mono."  -> ROOT anchor
RES    = _find("res/rep", "resident")      # "MΦ-Res/Rep"  -> resident endpoint
assert INF and RES, f"anchor subtypes not found among {sorted(_have)}"

LINEAGE_ORDER = ["Inf. Mono.", "Early MDM", "MΦ-Inf", "MΦ-Act", "MΦ-IFN/AS DCs",
                 "LAM-I", "LAM-II", "LAM", "MΦ-Res/Rep"]
KEEP_SUBS = [s for s in LINEAGE_ORDER if s in _have] + [s for s in sorted(_have) if s not in LINEAGE_ORDER]
XY       = adata_mac.obsm["X_umap"]
TIME_ALL = adata_mac.obs["Timepoint"].astype(str).str.extract(r"(\d+)").astype(float).iloc[:, 0].values

# subtype color palette (reuse notebook mac_colors; tab20 fallback for any extras)
_base_pal = mac_colors if "mac_colors" in globals() else {}
SUB_PAL   = {s: _base_pal.get(s, plt.get_cmap("tab20")(i % 20)) for i, s in enumerate(KEEP_SUBS)}

# ── root picker: PURE Inf.Mono. tip (majority Inf.Mono cells), time as tie-break ──
def _pick_inf_root(centroids, mst, nn_node, ident_sub, time_sub, ids_present):
    K = len(centroids)
    deg = np.asarray(((mst + mst.T) != 0).sum(1)).ravel()
    counts   = np.vstack([np.bincount(nn_node[ident_sub == i], minlength=K) for i in ids_present])
    total    = counts.sum(0).astype(float)
    inf_i    = ids_present.index(INF)
    inf_frac = counts[inf_i] / np.maximum(total, 1)
    dom      = counts.argmax(0)
    
    # 1. Primary candidates: majority or near-majority Inf. Mono.
    cand = np.where((total > 0) & (inf_frac >= 0.35))[0]
    if cand.size == 0: cand = np.where(total > 0)[0]
    
    # 2. Prefer leaf nodes, but fallback to any candidate
    leaves = cand[deg[cand] == 1]
    pool   = leaves if leaves.size else cand
    
    # 3. Tie-break: purity + early timepoints (as before)
    early = time_sub == np.nanmin(time_sub)
    n_tot = np.bincount(nn_node, minlength=K).astype(float)
    n_erl = np.bincount(nn_node[early], minlength=K).astype(float)
    frac_early = n_erl / np.maximum(n_tot, 1)
    
    root = int(pool[np.argmax(inf_frac[pool] + 0.25 * frac_early[pool])])
    
    # 4. SAFETY FALLBACK: if root has 0 Inf. Mono. cells, force to closest Inf. Mono. centroid
    if inf_frac[root] == 0:
        inf_centers = np.where(counts[inf_i] > 0)[0]
        root = int(inf_centers[np.argmin(cdist(centroids[root:root+1], centroids[inf_centers])[0])])
        
    return root, deg, float(inf_frac[root])


# ── fit one condition ─────────────────────────────────────────────────────────
def _fit_cond(xy, clu, ident_sub, time_sub, ids_present, n_centroids):
    projected_points, mst, centroids = learn_graph(
        matrix=xy, clusters=clu, n_centroids=n_centroids, prune=True, p_threshold=P_THRESHOLD)
    nn_node = cKDTree(centroids).query(xy)[1]
    root, deg, purity = _pick_inf_root(centroids, mst, nn_node, ident_sub, time_sub, ids_present)
    pt = np.asarray(order_cells(xy, centroids, mst=mst, projected_points=projected_points,
                                root_pr_cells=root), dtype=float)
    pt[np.isinf(pt)] = np.nan
    fin = np.isfinite(pt); ptn = np.full_like(pt, np.nan)
    if fin.any():
        lo, hi = np.nanmin(pt), np.nanmax(pt); ptn[fin] = (pt[fin] - lo) / (hi - lo + 1e-12)
    ii, jj = sp.triu(mst + mst.T, k=1).nonzero()
    rho_t = spearmanr(ptn[fin], time_sub[fin])[0] if fin.sum() > 2 else np.nan
    return dict(xy=xy, ident=ident_sub, centroids=centroids, edges=np.column_stack([ii, jj]),
                root=root, ptn=ptn, rho_time=rho_t, root_purity=purity, is_leaf=bool(deg[root] == 1))

def fit_trajectory(keep_ids, n_centroids=N_CENTROIDS):
    ids_present = [i for i in keep_ids if i in _have]
    mono = {}
    for cond in conds:
        m = ((adata_mac.obs["Type"].astype(str) == cond).values & ident_all.isin(ids_present).values)
        d = _fit_cond(XY[m], pd.factorize(sub_all.values[m])[0],
                      ident_all.values[m], TIME_ALL[m], ids_present, n_centroids)
        mono[cond] = d
        print(f"{cond}: n={int(m.sum())} root={INF} purity={d['root_purity']:.2f} "
              f"leaf={d['is_leaf']} rho(pt,time)={d['rho_time']:+.2f}")
    return mono, ids_present

# ── trajectory smoothing helpers ────────────────────────────────────────────────
def _chains_from_edges(edges):
    """Stitch the MST into continuous chains (break only at leaves/forks)."""
    adj = defaultdict(list)
    for a, b in edges:
        a, b = int(a), int(b); adj[a].append(b); adj[b].append(a)
    deg = {n: len(v) for n, v in adj.items()}
    eid = lambda u, v: (u, v) if u < v else (v, u)
    seen, chains = set(), []
    for e in [n for n in adj if deg[n] != 2]:
        for nb in adj[e]:
            if eid(e, nb) in seen: continue
            seen.add(eid(e, nb)); path = [e, nb]; prev, cur = e, nb
            while deg.get(cur, 0) == 2:
                nxt = next((x for x in adj[cur] if x != prev), None)
                if nxt is None or eid(cur, nxt) in seen: break
                seen.add(eid(cur, nxt)); path.append(nxt); prev, cur = cur, nxt
            chains.append(path)
    for a, b in edges:
        a, b = int(a), int(b)
        if eid(a, b) not in seen: seen.add(eid(a, b)); chains.append([a, b])
    return chains

def _chaikin(pts, iters=3):
    """Corner-cutting smoothing that keeps endpoints (no overshoot)."""
    pts = np.asarray(pts, float)
    for _ in range(iters):
        if len(pts) < 3: break
        new = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            new += [0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q]
        new.append(pts[-1]); pts = np.array(new)
    return pts

# ── figure: UMAP colored by macrophage_subtypes (Sham | Burn) ───────────────────
def plot_trajectory(mono, ids_present, desc, fname, lw_graph=5.0, label_fs=26):
    keep_all = ident_all.isin(ids_present).values
    ctx = XY[keep_all]; sub_ctx = sub_all.values[keep_all]
    pad  = (ctx[:, 0].max() - ctx[:, 0].min()) * 0.04
    xlim = (ctx[:, 0].min() - pad, ctx[:, 0].max() + pad)
    ylim = (ctx[:, 1].min() - pad, ctx[:, 1].max() + pad)
    label_pos = {s: np.median(ctx[sub_ctx == s], axis=0)
                 for s in ids_present if (sub_ctx == s).any()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    if desc: fig.suptitle(desc, fontsize=22, fontweight="bold", y=1.03)
    for k, (ax, cond) in enumerate(zip(axes, conds)):
        d = mono[cond]; C = d["centroids"]; sub_c = np.asarray(d["ident"])
        ax.scatter(ctx[:, 0], ctx[:, 1], s=4, c="#ECECEC", linewidths=0, rasterized=True, zorder=1)
        # color cells by macrophage_subtypes
        for s in ids_present:
            cm = sub_c == s
            if cm.any():
                ax.scatter(d["xy"][cm, 0], d["xy"][cm, 1], s=9, color=SUB_PAL[s],
                           linewidths=0, rasterized=True, zorder=2)
        # smooth continuous trajectory (white halo underlay → black on top)
        smooths = [_chaikin(C[np.asarray(ch, int)], 3) for ch in _chains_from_edges(d["edges"])]
        for sm in smooths:
            ax.plot(sm[:, 0], sm[:, 1], color="white", lw=lw_graph + 4.0,
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)
        for sm in smooths:
            ax.plot(sm[:, 0], sm[:, 1], color="black", lw=lw_graph,
                    solid_capstyle="round", solid_joinstyle="round", zorder=4)
        # boxed subtype labels
        for s, (lx, ly) in label_pos.items():
            ax.text(lx, ly, s, fontsize=19, fontweight="bold", ha="center", va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.82),
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        # prominent root star — always on top, haloed so it can't disappear
        ax.scatter(*C[d["root"]], s=950, marker="*", c="#E8412A", edgecolor="black", lw=2.2,
                   zorder=12, label=f"root ({INF})",
                   path_effects=[pe.withStroke(linewidth=4.0, foreground="white")])
        ax.set_title(cond, fontsize=30, fontweight="bold", color=TYPE_PAL[cond], pad=10)
        ax.set_xlabel("UMAP 1", fontsize=label_fs, fontweight="bold")
        ax.set_ylabel("UMAP 2" if k == 0 else "", fontsize=label_fs, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.legend(fontsize=13, frameon=True, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGDIR_MAC / f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR_MAC / f"{fname}.pdf", bbox_inches="tight")
    plt.show()

# ── run: fit + plot ─────────────────────────────────────────────────────────────
mono_full, ids_present = fit_trajectory(KEEP_SUBS, N_CENTROIDS)
plot_trajectory(mono_full, ids_present, "", "monocle_trajectory_bysubtype_colored")


import scanpy as sc, numpy as np, matplotlib.pyplot as plt

ident_all = adata_mac.obs["mac_identity"].astype(str)
_have = sorted(ident_all.unique()); _find = lambda *ks: next((h for h in _have if any(k in h.lower() for k in ks)), None)
INF, RECR = _find("inflamm", "mono"), _find("recruit")
KEEP = [INF, RECR]
keep_mask = ident_all.isin(KEEP).values
adata_ir = adata_mac[keep_mask].copy()
print("subset:", adata_ir.n_obs, "cells |", {k: int((ident_all[keep_mask] == k).sum()) for k in KEEP})

import scanpy as sc, numpy as np
try:
    import scanpy.external as sce
except Exception as e:
    raise ImportError("pip install harmonypy") from e

# technical batch key — sample/library/mouse, NOT Type/Timepoint (those are the biology)
BATCH_CANDS = ["Sample", "sample", "orig.ident", "orig_ident", "SampleID", "sample_id",
               "library", "Library", "mouse", "Mouse", "replicate", "Replicate", "batch"]
BATCH = next((c for c in BATCH_CANDS if c in adata_ir.obs.columns), None)
assert BATCH, f"set BATCH manually from {list(adata_ir.obs.columns)}"
print(f"Harmony batch key = {BATCH}  ({adata_ir.obs[BATCH].nunique()} levels)")

sc.pp.highly_variable_genes(adata_ir, n_top_genes=2000, flavor="seurat")
emb = adata_ir[:, adata_ir.var.highly_variable].copy()
sc.pp.scale(emb, max_value=10)
sc.tl.pca(emb, n_comps=30, svd_solver="arpack")
sce.pp.harmony_integrate(emb, BATCH, basis="X_pca", adjusted_basis="X_pca_harmony", max_iter_harmony=20)
adata_ir.obsm["X_pca"]         = emb.obsm["X_pca"]
adata_ir.obsm["X_pca_harmony"] = emb.obsm["X_pca_harmony"]

sc.pp.neighbors(adata_ir, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony")
sc.tl.umap(adata_ir, min_dist=0.3)
try:
    sc.tl.leiden(adata_ir, resolution=0.5, key_added="ir_leiden", flavor="igraph", n_iterations=2, directed=False)
except TypeError:
    sc.tl.leiden(adata_ir, resolution=0.5, key_added="ir_leiden")
print("clusters:", adata_ir.obs["ir_leiden"].nunique(),
      "| Burn frac:", round((adata_ir.obs['Type'].astype(str) == 'Burn').mean(), 2))


import scanpy as sc, matplotlib.pyplot as plt
sc.settings.figdir = str(FIGDIR_MAC)   # so save= lands in your figures dir

# (1) Leiden clusters, labels on data
sc.pl.umap(adata_ir, color="ir_leiden", size=14, alpha=0.9, frameon=False,
           legend_loc="on data", legend_fontsize=12, legend_fontweight="bold",
           title="Re-subclustered Inf+Recruited — Leiden (Harmony)", save="_umap_ir_leiden.png")

# (2) subtypes by Type × Timepoint — identical to your grid (X_umap is now Harmony-corrected)
import numpy as np, re, matplotlib as mpl
sub_ir = adata_ir.obs["macrophage_subtypes"].astype(str); present = set(sub_ir.unique())
cats = [c for c in (desired_order if "desired_order" in globals() else sorted(present)) if c in present]
um = adata_ir.obsm["X_umap"]
ty_arr = adata_ir.obs["Type"].astype(str).values
tp_arr = adata_ir.obs["Timepoint"].astype(str).str.extract(r"(\d+)")[0].radd("D").values
subt = sub_ir.values
type_order = ["Sham", "Burn"]; TYPE_COL = {"Sham": "#2471A3", "Burn": "#C0392B"}
tp_order = sorted(set(tp_arr), key=lambda t: int(re.sub(r"\D", "", t)))
px = 0.04 * (um[:, 0].max() - um[:, 0].min()); py = 0.04 * (um[:, 1].max() - um[:, 1].min())
xlim = (um[:, 0].min() - px, um[:, 0].max() + px); ylim = (um[:, 1].min() - py, um[:, 1].max() + py)
ncols = len(tp_order)
fig, axes = plt.subplots(2, ncols, figsize=(4.4 * ncols, 9), squeeze=False)
fig.suptitle("Re-subclustered Inf+Recruited (Harmony) — subtypes by Type × Timepoint", fontsize=18, fontweight="bold", y=1.01)
for r, ty in enumerate(type_order):
    for c, tp in enumerate(tp_order):
        ax = axes[r][c]
        ax.scatter(um[:, 0], um[:, 1], s=4, c="#ECECEC", linewidths=0, rasterized=True)
        sel = (ty_arr == ty) & (tp_arr == tp)
        ax.scatter(um[sel, 0], um[sel, 1], c=[mac_colors.get(s, "#999999") for s in subt[sel]],
                   s=14, linewidths=0, rasterized=True)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.03, 0.97, f"n={int(sel.sum()):,}", transform=ax.transAxes, ha="left", va="top",
                fontsize=11, fontweight="bold")
        if r == 0: ax.set_title(tp, fontsize=20, fontweight="bold")
        if c == 0: ax.set_ylabel(ty, fontsize=20, fontweight="bold", color=TYPE_COL[ty])
handles = [mpl.patches.Patch(facecolor=mac_colors.get(c, "#999999"), label=c) for c in cats]
leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
                 fontsize=13, title="Subtype", title_fontsize=14)
for t in leg.get_texts(): t.set_fontweight("bold")
fig.tight_layout()
fig.savefig(FIGDIR_MAC / "umap_ir_subtypes_by_type_timepoint_harmony.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "umap_ir_subtypes_by_type_timepoint_harmony.pdf", bbox_inches="tight")
plt.show()


import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, dendrogram as _scdendro

SPREAD_MIN, PCT_MIN = 0.55, 0.10     # keep if min group is <45% of max group AND ≥10% cells somewhere



gene_list = [
    # ── Glycolysis ───────────────────────────────
    "Hk2","Pfkfb3","Aldoa","Pkm","Ldha","Slc2a1",
    # ── Hypoxia / HIF ────────────────────────────
    "Egln3","Hilpda","Vegfa","Fam162a",
    # ── Redox / heme stress ──────────────────────
    "Sod2","Hmox1",
    # ── Inflammatory / cytokine ──────────────────
    "Il1b","Nos2","Saa3","Cxcl2","Nfkbia","Cebpb","Il1rn","Csf3r","Ncf2",
    # ── ER-stress / proteostasis / lysosomal ─────
    "Ero1l","Furin","Ndfip1","Atp6v0c","Ctsb","Ctsd","Ctss",
    # ── Reparative lipid / resolution (Sham) ─────
    "Spp1","Gpnmb","Lpl","Apoe","Plin2","Tgm2","Thbs1","Timp2",
    # ── Immediate-early / AP-1 ───────────────────
    "Fos","Egr1","Junb","Jdp2","Dusp1","Nr4a3",
]



gene_list = [
    # ── Glycolysis ───────────────────────────────
    "Hk2",  # Glucose-6-phosphate isomerase
    "Pfkfb3",  # Phosphofructokinase-1, fructose-bisphosphate
    "Aldoa",  # Fructose-1,6-bisphosphate aldolase
    "Pkm",  # Pyruvate kinase
    "Ldha",  # Lactate dehydrogenase
    "Slc2a1",  # Facilitated glucose transporter (GLUT1),
    # ── Hypoxia / HIF ────────────────────────────
    "Egln3",  # Endothelin-binding protein
    "Hilpda",  # Hypoxia-inducible factor 1A, dominant negative variant
    "Vegfa",  # Vascular endothelial growth factor A (VEGF-A)
    "Fam162a",  # Family with sequence similarity 162 member A
    # ── Redox / heme stress ──────────────────────
    "Sod2",  # Superoxide dismutase 2 (Mn-dependent)
    "Hmox1",  # Heme oxygenase 1
    # ── Inflammatory / cytokine ──────────────────
    "Il1b",  # Interleukin-1 beta
    "Arg1",
    "Nos2",  # Nitric oxide synthase 2 (inducible)
    "Saa3",  # Serum albumin A3
    "Cxcl2",  # C-X-C motif chemokine ligand 2 (IL-8)
    "Nfkbia",  # Nuclear factor kappa B inhibitor alpha
    "Cebpb",  # CCAAT/enhancer-binding protein beta (CEBPA)
    "Il1rn",  # Interleukin-1 receptor antagonist
    "Csf3r",  # Colony-stimulating factor 3 receptor
    "Ncf2",  # Nuclear factor of activated T-cells 2
    # ── ER-stress / proteostasis / lysosomal ─────
    "Ero1l",  # ERp57
    "Furin",  # Proprotein convertase Furin
    "Ndfip1",  # Nuclear DNA binding protein 1-like
    "Atp6v0c",  # ATP-synthase V0 component subunit C (lysosomal)
    "Ctsb",  # Cathepsin B
    "Ctsd",  # Cathepsin D
    "Ctss",  # Cathepsin S
    # ── Reparative lipid / resolution (Sham) ─────
    "Spp1",  # Sterile alpha-matrix protein 1
    "Gpnmb",  # Gastric phosphatase (GPNAc)
    "Lpl",  # Lipoprotein lipase
    "Apoe",  # Apolipoprotein E
    "Plin2",  # Phospholipid-binding protein 2 (PLIN2)
    "Tgm2",  # Transglutaminase 2
    "Thbs1",  # Thrombospondin 1
    "Timp2",  # Type I transmembrane proteoglycan 2
    # ── Immediate-early / AP-1 ───────────────────
    "Fos",  # Fos-related antigen 1 (FRA-1)
    "Egr1",  # Early growth response protein 1
    "Junb",  # Jun B proto-oncogene product
    "Jdp2",  # Jun-D domain-containing protein 2
    "Dusp1",  # Dual specificity phosphatase 1 (DUSP1)
    "Nr4a3",  # Nuclear receptor subfamily 4, group A, member 3 (NR4A3)
]



genes = [g for g in gene_list if g in ad.var_names]

X = ad[:, genes].X; X = X.toarray() if sp.issparse(X) else np.asarray(X)
grp = ad.obs["Type_TP"].values
M = pd.DataFrame(X, columns=genes).assign(g=grp).groupby("g").mean().reindex(order)       # mean expr
P = pd.DataFrame(X > 0, columns=genes).assign(g=grp).groupby("g").mean().reindex(order)   # frac expressing

mx, mn = M.max(0), M.min(0)
spread = (mx - mn) / (mx + 1e-9)                       # fractional dynamic range across groups
keep = [g for g in genes if spread[g] >= SPREAD_MIN and P[g].max() >= PCT_MIN]
drop = [g for g in genes if g not in keep]
print(f"kept {len(keep)}/{len(genes)}  |  dropped (flat/low): {drop}")

# cluster the kept genes by their z-scored profile
Mz = (M[keep] - M[keep].mean(0)) / (M[keep].std(0) + 1e-9)
Z = linkage(Mz.T.values, method="ward")
genes_clustered = [keep[i] for i in _scdendro(Z, no_plot=True)["leaves"]]
import scanpy as sc

# ── KNN + Leiden clustering of the kept genes (features = z-scored profile over Type×TP) ──
Mz = ((M[keep] - M[keep].mean(0)) / (M[keep].std(0) + 1e-9)).T      # genes × groups
gad = sc.AnnData(Mz.values.astype("float32"))
gad.obs_names = list(Mz.index); gad.var_names = list(M.index)

sc.pp.neighbors(gad, n_neighbors=min(10, len(keep) - 1), use_rep="X", metric="correlation")
try:
    sc.tl.leiden(gad, resolution=1.0, key_added="module", flavor="igraph", n_iterations=2, directed=False)
except TypeError:
    sc.tl.leiden(gad, resolution=1.0, key_added="module")
mod = gad.obs["module"].astype(str)

# order modules (and genes within) by the Type×TP group where they peak → temporal read top→bottom
gpeak   = {g: int(Mz.loc[g].values.argmax()) for g in keep}
modpeak = {m: float(np.mean([gpeak[g] for g in keep if mod[g] == m])) for m in mod.unique()}
modules = {f"M{i+1}": sorted([g for g in keep if mod[g] == m], key=lambda g: gpeak[g])
           for i, m in enumerate(sorted(mod.unique(), key=lambda m: modpeak[m]))}
print({k: v for k, v in modules.items()})

with plt.rc_context({"font.size": 12, "axes.titlesize": 12, "legend.fontsize": 12}):
    dp = sc.pl.dotplot(
        ad, modules, groupby="Type_TP", categories_order=order,
        standard_scale="var", cmap="Reds", swap_axes=True, dendrogram=False,
        colorbar_title="Mean expression\nin group", size_title="Fraction of cells\nin group (%)",
        figsize=(5.5, max(1, 0.34 * len(keep))), return_fig=True,
    )
    dp.legend(width=1.8)
    dp.make_figure()
    ax = dp.ax_dict["mainplot_ax"]
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=16, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontsize=16, fontweight="bold")
    dp.fig.savefig(FIGDIR_MAC / "dotplot_temporal_knn_modules.png", dpi=300, bbox_inches="tight")
    dp.fig.savefig(FIGDIR_MAC / "dotplot_temporal_knn_modules.pdf", bbox_inches="tight")
    plt.show()


import scipy.sparse as sp, gseapy as gp, numpy as np, pandas as pd, scanpy as sc, matplotlib.pyplot as plt

# drivers of the panels where centroids separate (recruited lineage, mid-late timepoints)
sep = (adata_mac.obs["mac_identity"].astype(str).isin([find("inflamm"), find("recruit")]) &
       adata_mac.obs["Timepoint"].astype(str).isin(["D10", "D14", "D19"]))
A = adata_mac[sep.values].copy()
# … then build Type_TP, run _top(...) and the dotplot exactly as before

A = adata_mac                                  # all 3 mac compartments (the plot's cells)
typ = A.obs["Type"].astype(str).values

# Type × Timepoint axis
tpA = A.obs["Timepoint"].astype(str).str.extract(r"(\d+)")[0].radd("D")
A.obs["Type_TP"] = (A.obs["Type"].astype(str) + " " + tpA).astype("category")
order = [f"{t} {d}" for t in ["Sham","Burn"] for d in ["D7","D10","D14","D19"]]
order = [o for o in order if o in set(A.obs["Type_TP"])]
A.obs["Type_TP"] = A.obs["Type_TP"].cat.set_categories(order)

# the 4 gene sets that BUILD the two axes
h2m  = lambda g: g[0].upper() + g[1:].lower()
hall = gp.get_library("MSigDB_Hallmark_2020", organism="Mouse")
glyco  = [h2m(g) for g in hall["Glycolysis"]]
oxphos = [h2m(g) for g in hall["Oxidative Phosphorylation"]]
try:
    recr_genes = ["Ccr2","Ly6c2","Arg1", "Nos2","Ero1l","Slpi", "Fn1","Spp1","Trem2","Gpnmb","Cd9","Ms4a7","Itgax"]
    res_genes  = ["Adgre1","Mertk","Timd4","Cd163","Mrc1","Folr2","Lyve1","Gas6","Selenop","C1qa","C1qb","C1qc","Pf4","Maf"]
except NameError:
    recr_genes = ["Ccr2","Ly6c2","Arg1", "Nos2","Fn1", "Ero1l","Slpi", "Spp1","Trem2","Gpnmb","Cd9","Ms4a7","Itgax"]
    res_genes  = ["Adgre1","Mertk","Timd4","Cd163","Mrc1","Folr2","Lyve1","Gas6","Selenop","C1qa","C1qb","C1qc","Pf4","Maf"]

def _top(genes, sham_up, n=8, min_expr=0.10):
    genes = [g for g in dict.fromkeys(genes) if g in A.var_names]
    if not genes: return []
    Xg = A[:, genes].X; Xg = Xg.toarray() if sp.issparse(Xg) else np.asarray(Xg)
    df = pd.DataFrame(Xg, columns=genes)
    mb, ms = df[typ == "Burn"].mean(), df[typ == "Sham"].mean()
    diff = (ms - mb) if sham_up else (mb - ms)          # +ve = drives the centroid that way
    ok = [g for g in diff.index if max(mb[g], ms[g]) >= min_expr and diff[g] > 0]
    return diff[ok].sort_values(ascending=False).head(n).index.tolist()

groups = {
    "Glycolysis": _top(glyco,      sham_up=False),   # y-axis: Burn glycolytic
    "OXPHOS":     _top(oxphos,     sham_up=True),    # y-axis: Sham OXPHOS
    "Recruited":  _top(recr_genes, sham_up=False),   # x-axis: Burn recruited
    "Resident":   _top(res_genes,  sham_up=True),    # x-axis: Sham resident
}
groups = {g: gl for g, gl in groups.items() if gl}
print({g: gl for g, gl in groups.items()})

with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "legend.fontsize": 10}):
    dp = sc.pl.dotplot(
        A, groups, groupby="Type_TP", categories_order=order,
        standard_scale="var", cmap="Reds", swap_axes=True, dendrogram=False,
        colorbar_title="Mean expression\nin group", size_title="Fraction of cells\nin group (%)",
        figsize=(5.5, max(5, 0.34 * sum(len(v) for v in groups.values()))), return_fig=True,
    )
    dp.legend(width=1.8); dp.make_figure()
    axm = dp.ax_dict["mainplot_ax"]
    plt.setp(axm.get_xticklabels(), rotation=45, ha="right", fontsize=16, fontweight="bold")
    plt.setp(axm.get_yticklabels(), fontsize=16, fontweight="bold")
    dp.fig.savefig(FIGDIR_MAC / "dotplot_centroid_drivers.png", dpi=300, bbox_inches="tight")
    dp.fig.savefig(FIGDIR_MAC / "dotplot_centroid_drivers.pdf", bbox_inches="tight")
    plt.show()


sc.pl.umap(adata_mac,color=["C1qa", "C1qb", "Apoe", "Csf1r"], palette="tab20", wspace=0.4)

sc.pl.umap(adata_full,color=["C1qa", "C1qb", "Bst2", "Il3ra"], palette="tab20", wspace=0.4)

sc.pl.umap(adata_mac,color=["C1qa", "C1qb", "Bst2"], palette="tab20", wspace=0.4)

sc.pl.umap(adata_mac,color=["C1qa", "C1qb", "Bst2"], palette="tab20", wspace=0.4)

import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
import scanpy as sc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

m1_genes      = ['Nos2', 'Arg1']
genes_to_plot = [g for g in m1_genes if g in adata_mac.var_names]
types         = ['Sham', 'Burn']                       # Sham top, Burn bottom
ID_COL        = 'mac_identity'

# identity columns (biological order first)
id_all   = adata_mac.obs[ID_COL].astype(str)
pref     = ["Inflammatory Monocytes", "MΦ-Recruited", "MΦ-Resident/Repair"]
identities = [i for i in pref if i in set(id_all)] + [i for i in sorted(set(id_all)) if i not in pref]
SHORT = {"Inflammatory Monocytes": "Inf. Mono.", "MΦ-Recruited": "MΦ-Recr", "MΦ-Resident/Repair": "MΦ-Res/Rep"}

colors = ["#D3D3D3", "#3F00FF"]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("lightgrey_blue", colors, N=256)

# per-gene vmax + expression vectors
def _vec(g):
    v = adata_mac[:, g].X
    return v.toarray().ravel() if hasattr(v, "toarray") else np.asarray(v).ravel()
expr   = {g: _vec(g) for g in genes_to_plot}
vmaxes = {g: (float(np.percentile(v[v > 0], 99)) if (v > 0).any() else 1.0) for g, v in expr.items()}

# fixed UMAP extent (same framing in every panel)
um = adata_mac.obsm['X_umap']
xpad = 0.03 * (um[:, 0].max() - um[:, 0].min()); ypad = 0.03 * (um[:, 1].max() - um[:, 1].min())
xlim = (um[:, 0].min() - xpad, um[:, 0].max() + xpad)
ylim = (um[:, 1].min() - ypad, um[:, 1].max() + ypad)

typ = adata_mac.obs['Type'].astype(str).values
idv = id_all.values

n_rows, n_cols = len(genes_to_plot) * len(types), len(identities)
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(2.6 * n_cols, 1.9 * n_rows),
                         constrained_layout=True)
axes = np.atleast_2d(axes)

for gi, gene in enumerate(genes_to_plot):
    vmax, ev = vmaxes[gene], expr[gene]
    for ti, wtype in enumerate(types):
        r = gi * len(types) + ti
        for c, idn in enumerate(identities):
            ax   = axes[r, c]
            mask = (typ == wtype) & (idv == idn)
            ax.scatter(um[:, 0], um[:, 1], s=6, c="#ECECEC", linewidths=0, rasterized=True)  # context
            ax.scatter(um[mask, 0], um[mask, 1], c=ev[mask], cmap=custom_cmap,
                       vmin=0, vmax=vmax, s=10, linewidths=0, rasterized=True)               # gene
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
            if r == 0:
                ax.set_title(SHORT.get(idn, idn), fontsize=18, fontweight='bold', pad=4)
            if c == 0:                                          # bold Sham / Burn label
                ax.text(-0.14, 0.5, wtype, transform=ax.transAxes, rotation=90,
                        ha='center', va='center', fontsize=22, fontweight='bold')

fig.canvas.draw()                                              # finalize positions before colorbars

for gi, gene in enumerate(genes_to_plot):
    top_ax = axes[gi * len(types), -1]
    bot_ax = axes[gi * len(types) + 1, -1]
    y0, y1 = bot_ax.get_position().y0, top_ax.get_position().y1
    cax = fig.add_axes([1.01, y0, 0.012, y1 - y0])
    sm  = ScalarMappable(cmap=custom_cmap, norm=Normalize(vmin=0, vmax=vmaxes[gene]))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(gene, fontsize=22, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=15)

plt.show()


import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

genes    = [g for g in ['Arg1','Nos2', ] if g in adata_mac.var_names]
types    = ['Sham', 'Burn']
TYPE_PAL = {'Sham': '#2980B9', 'Burn': '#C0392B'}
tp_order = ['D7', 'D10', 'D14', 'D19']
tps      = [t for t in tp_order if t in adata_mac.obs['Timepoint'].astype(str).unique()]
tp_day   = {t: int(t[1:]) for t in tps}
TEST     = 't'                        # 't' = Welch t-test | 'mwu' = Mann-Whitney

ID_COL = 'mac_identity'
pref   = ["Inflammatory Monocytes", "MΦ-Recruited"]
ids    = [i for i in pref if i in set(adata_mac.obs[ID_COL].astype(str))]
SHORT  = {"Inflammatory Monocytes": "Inf. Mono.", "MΦ-Recruited": "MΦ-Recr", "MΦ-Resident/Repair": "MΦ-Res/Rep"}

# sample (mouse) column
SAMP_CANDS = ["Sample","sample","orig.ident","orig_ident","SampleID","sample_id",
              "library","Library","mouse","Mouse","replicate","Replicate"]
samp_col = next((c for c in SAMP_CANDS if c in adata_mac.obs.columns), None)
assert samp_col, f"Set samp_col manually from {list(adata_mac.obs.columns)}"
print(f"aggregating by {samp_col!r}")

def _vec(g):
    v = adata_mac[:, g].X
    return v.toarray().ravel() if hasattr(v, "toarray") else np.asarray(v).ravel()
df = pd.DataFrame({"Sample": adata_mac.obs[samp_col].astype(str).values,
                   "Type":   adata_mac.obs["Type"].astype(str).values,
                   "tp":     adata_mac.obs["Timepoint"].astype(str).values,
                   "id":     adata_mac.obs[ID_COL].astype(str).values})
for g in genes: df[g] = _vec(g)

# ── pseudobulk: mean expression per mouse × identity (× its tp/Type) ─────────────
pb = df.groupby(["Sample", "id", "tp", "Type"], observed=True)[genes].mean().reset_index()

def _p(b, s):
    if len(b) < 2 or len(s) < 2: return np.nan
    return (mannwhitneyu(b, s, alternative="two-sided").pvalue if TEST == "mwu"
            else ttest_ind(b, s, equal_var=False).pvalue)
def _star(p):
    return "" if np.isnan(p) else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

rng = np.random.default_rng(0)
nr, nc = len(genes), len(ids)
fig, axes = plt.subplots(nr, nc, figsize=(10,8), squeeze=False, sharex=True)
for gi, g in enumerate(genes):
    ymax = 0
    for ci, idn in enumerate(ids):
        ax = axes[gi][ci]; means = {}
        for ty in types:
            sub = pb[(pb.id == idn) & (pb.Type == ty)]
            grp = sub.groupby("tp")[g].agg(["mean", "sem"]).reindex(tps); means[ty] = grp
            jit = 0.12 * (1 if ty == "Burn" else -1)
            for t in tps:                                   # individual mice
                v = sub[sub.tp == t][g].values
                if len(v):
                    ax.scatter(np.full(len(v), tp_day[t] + jit) + rng.uniform(-.04, .04, len(v)),
                               v, s=22, color=TYPE_PAL[ty], alpha=0.45, edgecolor="none", zorder=2)
                    ymax = max(ymax, v.max())
            ax.errorbar([tp_day[t] for t in tps], grp["mean"], yerr=grp["sem"], fmt="-o",
                        color=TYPE_PAL[ty], lw=3, ms=9, capsize=4, markeredgecolor="white",
                        markeredgewidth=1.2, label=ty, zorder=3)
        # Burn vs Sham stat per timepoint (sample-level)
        for t in tps:
            b = pb[(pb.id == idn) & (pb.Type == "Burn") & (pb.tp == t)][g].values
            s = pb[(pb.id == idn) & (pb.Type == "Sham") & (pb.tp == t)][g].values
            st = _star(_p(b, s))
            if st:
                yt = np.nanmax([means[ty].loc[t, "mean"] + means[ty].loc[t, "sem"] for ty in types])
                ax.text(tp_day[t], yt, st, ha="center", va="bottom", fontsize=20,
                        fontweight="bold", color="0.15")
        if gi == 0: ax.set_title(SHORT.get(idn, idn), fontsize=25, fontweight="bold", pad=8)
        if ci == 0: ax.set_ylabel(f"{g} Expr.", fontsize=22, fontweight="bold")
        ax.set_xticks([tp_day[t] for t in tps]); ax.set_xticklabels(tps, fontsize=20, fontweight="bold")
        ax.tick_params(axis="y", labelsize=18)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for ci in range(nc):
        axes[gi][ci].set_ylim(0, ymax * 1.18)               # shared y per gene row

for ci in range(nc): axes[-1][ci].set_xlabel("", fontsize=15, fontweight="bold")
axes[0][-1].legend(title="Condition", bbox_to_anchor=(1.02, 1.0), loc="upper left",
                   frameon=False, fontsize=15, title_fontsize=15)
fig.tight_layout()
fig.savefig(FIGDIR_MAC / "nos2_arg1_burn_vs_sham_by_identity_stats.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "nos2_arg1_burn_vs_sham_by_identity_stats.pdf", bbox_inches="tight", facecolor="white")
plt.show()

