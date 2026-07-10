"""Lightweight CellChat-style crosstalk: Inf.Mono / Recruited / Resident macrophage

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, numpy, pathlib
Key calls: def _edge, def _selfloop, def comm_matrix, def draw_net, dendrogram, plt.rcParams, plt.savefig, plt.show, plt.subplots, sc.pl.dotplot
"""

# ══════════════════════════════════════════════════════════════════════════════
# Lightweight CellChat-style crosstalk: Inf.Mono / Recruited / Resident macrophages
#   score(S→R) = Σ_LR  mean(ligand|S) × mean(receptor|R)   (frac-expressing gated)
#   Fig 1: circle-plot grid (Burn/Sham × timepoint).  Fig 2: resident chemokine source.
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity"); FIGDIR_MAC.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
MIN_FRAC = 0.10                                     # ligand/receptor must be in ≥10% of the group

# ── curated mouse L–R pairs (inflammation + monocyte recruitment) ────────────────
LR_RAW = [
    ("Ccl2", ["Ccr2"]), ("Ccl7", ["Ccr2"]), ("Ccl8", ["Ccr2"]), ("Ccl12", ["Ccr2"]),
    ("Ccl3", ["Ccr1", "Ccr5"]), ("Ccl4", ["Ccr5"]), ("Ccl5", ["Ccr1", "Ccr5"]),
    ("Cx3cl1", ["Cx3cr1"]), ("Cxcl1", ["Cxcr2"]), ("Cxcl2", ["Cxcr2"]), ("Cxcl3", ["Cxcr2"]),
    ("Cxcl9", ["Cxcr3"]), ("Cxcl10", ["Cxcr3"]), ("Cxcl16", ["Cxcr6"]),
    ("Il1b", ["Il1r1"]), ("Il1a", ["Il1r1"]), ("Tnf", ["Tnfrsf1a", "Tnfrsf1b"]),
    ("Il6", ["Il6ra"]), ("Osm", ["Osmr"]), ("Il18", ["Il18r1"]), ("Csf1", ["Csf1r"]),
    ("Csf2", ["Csf2rb"]), ("Spp1", ["Cd44"]), ("Tgfb1", ["Tgfbr1"]), ("Igf1", ["Igf1r"]),
]

# ── identities + expression tables ──────────────────────────────────────────────
ID_COL = "mac_identity"
idv = adata_mac.obs[ID_COL].astype(str)
_have = sorted(idv.unique()); _find = lambda *k: next((h for h in _have if any(x in h.lower() for x in k)), None)
INF, RECR, RESI = _find("inflamm", "mono"), _find("recruit"), _find("resident")
IDENTS = [x for x in [INF, RECR, RESI] if x]
present = set(adata_mac.var_names)
LR = [(l, [r for r in rs if r in present]) for l, rs in LR_RAW if l in present]
LR = [(l, rs) for l, rs in LR if rs]
genes = sorted({g for l, rs in LR for g in [l] + rs})

X = adata_mac[:, genes].X; X = X.toarray() if sp.issparse(X) else np.asarray(X)
key = (adata_mac.obs["Type"].astype(str).values + "|" +
       adata_mac.obs["Timepoint"].astype(str).values + "|" + idv.values)
E = pd.DataFrame(X, columns=genes); 
meanE = E.groupby(key).mean(); fracE = (E > 0).groupby(key).mean()

conds = ["Burn", "Sham"]
tps   = [t for t in ["D7", "D10", "D14", "D19"] if t in set(adata_mac.obs["Timepoint"].astype(str))]

def comm_matrix(cond, tp):
    M = pd.DataFrame(0.0, index=IDENTS, columns=IDENTS)
    for S in IDENTS:
        ks = f"{cond}|{tp}|{S}"
        if ks not in meanE.index: continue
        for R in IDENTS:
            kr = f"{cond}|{tp}|{R}"
            if kr not in meanE.index: continue
            tot = 0.0
            for lig, recs in LR:
                if fracE.at[ks, lig] < MIN_FRAC: continue
                rp = [r for r in recs]
                if max(fracE.at[kr, r] for r in rp) < MIN_FRAC: continue
                tot += meanE.at[ks, lig] * max(meanE.at[kr, r] for r in rp)
            M.at[S, R] = tot
    return M

Ms = {(c, t): comm_matrix(c, t) for c in conds for t in tps}
vmax = max(M.values.max() for M in Ms.values()) or 1.0

# ══ FIGURE 1 — circle-plot grid ══════════════════════════════════════════════════
ID_PAL = {INF: "#8E44AD", RECR: "#E67E22", RESI: "#16A085"}       # not red/blue (Burn/Sham)
SHORT  = {INF: "Inf.\nMono", RECR: "Recr.\nMΦ", RESI: "Res.\nMΦ"}
POS    = {INF: (0, 1), RECR: (-0.9, -0.7), RESI: (0.9, -0.7)}
TCOL   = {"Burn": "#C0392B", "Sham": "#2980B9"}

def _edge(ax, p0, p1, w, color):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=0.16", arrowstyle="-|>",
                                 mutation_scale=11, lw=w, color=color, alpha=0.85,
                                 shrinkA=15, shrinkB=15, zorder=2))
def _selfloop(ax, p, w, color):
    x, y = p
    ax.add_patch(FancyArrowPatch((x - 0.16, y + 0.10), (x + 0.16, y + 0.10),
                                 connectionstyle="arc3,rad=2.6", arrowstyle="-|>",
                                 mutation_scale=9, lw=w, color=color, alpha=0.85,
                                 shrinkA=2, shrinkB=2, zorder=2))
def draw_net(ax, M):
    for S in IDENTS:
        for R in IDENTS:
            w = M.at[S, R]
            if w <= 0: continue
            lw = 0.6 + 6.5 * (w / vmax)
            (_selfloop if S == R else _edge)(ax, POS[S], *( (POS[S], lw, ID_PAL[S]) if S == R else (POS[R], lw, ID_PAL[S]) )[1:] ) if False else None
            if S == R: _selfloop(ax, POS[S], lw, ID_PAL[S])
            else:      _edge(ax, POS[S], POS[R], lw, ID_PAL[S])
    for n in IDENTS:
        ax.add_patch(Circle(POS[n], 0.17, color=ID_PAL[n], ec="black", lw=1.5, zorder=3))
        ax.text(*POS[n], SHORT[n], ha="center", va="center", fontsize=7.5, fontweight="bold", color="white", zorder=4)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.4, 1.7); ax.set_aspect("equal"); ax.axis("off")

fig, axes = plt.subplots(len(conds), len(tps), figsize=(3.2 * len(tps), 3.4 * len(conds)))
for i, c in enumerate(conds):
    for j, t in enumerate(tps):
        ax = axes[i][j]; M = Ms[(c, t)]; draw_net(ax, M)
        ax.set_title(f"{t}   (Σ={M.values.sum():.0f})", fontsize=12, fontweight="bold")
    axes[i][0].text(-0.12, 0.5, c, transform=axes[i][0].transAxes, rotation=90, ha="center", va="center",
                    fontsize=20, fontweight="bold", color=TCOL[c])
handles = [Line2D([0], [0], marker="o", linestyle="", markersize=12, markerfacecolor=ID_PAL[n],
                  markeredgecolor="black", label=n) for n in IDENTS]
handles += [Line2D([0], [0], color="0.4", lw=w, label=lab) for w, lab in [(1, "weak"), (6, "strong")]]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False, fontsize=12)
fig.suptitle("Macrophage inflammatory crosstalk (ligand→receptor strength)", fontsize=16, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(FIGDIR_MAC / "cellchat_crosstalk_grid_burn_vs_sham.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "cellchat_crosstalk_grid_burn_vs_sham.pdf", bbox_inches="tight", facecolor="white")
plt.show()

# ══ FIGURE 2 — resident macrophages as chemokine/cytokine SOURCE ══════════════════
recruit_lig = [g for g in ["Ccl2","Ccl7","Ccl8","Ccl12","Ccl3","Ccl4","Ccl5","Cxcl1","Cxcl2",
                           "Cxcl3","Cxcl9","Cxcl10","Cxcl16","Cx3cl1","Il1b","Il1a","Tnf","Il6",
                           "Osm","Csf1","Spp1"] if g in adata_mac.var_names]
tt = [f"{ty} {d}" for ty in ["Sham", "Burn"] for d in ["D7","D10","D14","D19"]
      if f"{ty} {d}" in set(adata_mac.obs["Type_Timepoint_C"].astype(str))]
adata_mac.obs["Type_Timepoint_C"] = pd.Categorical(adata_mac.obs["Type_Timepoint_C"].astype(str),
                                                    categories=tt, ordered=True)
res = adata_mac[idv.values == RESI].copy()
sc.pl.dotplot(res, recruit_lig, groupby="Type_Timepoint_C", categories_order=tt,
              standard_scale="var", cmap="Reds", dendrogram=False,
              title="Resident MΦ — monocyte-recruiting chemokines/cytokines (source)",
              figsize=(0.4 * len(recruit_lig) + 3, 0.5 * len(tt) + 2), show=False)
plt.savefig(FIGDIR_MAC / "cellchat_resident_chemokine_source.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(FIGDIR_MAC / "cellchat_resident_chemokine_source.pdf", bbox_inches="tight", facecolor="white")
plt.show()

