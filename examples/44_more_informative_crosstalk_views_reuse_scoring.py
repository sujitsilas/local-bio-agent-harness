"""More informative crosstalk views (reuse scoring from the CellChat-style cell)

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, numpy, pathlib, scipy, statsmodels
Key calls: def _st, def _vec, def coexp_stats, def render_table, heatmap, plt.Normalize, plt.cm, plt.show, plt.subplots, scatter
"""

# ══════════════════════════════════════════════════════════════════════════════
# More informative crosstalk views (reuse scoring from the CellChat-style cell):
#   FIG A: differential signaling heatmap (Burn − Sham), sender→receiver × timepoint
#   FIG B: resident MΦ → monocyte/recruited L–R bubble plot, Sham vs Burn over time
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, matplotlib.pyplot as plt

SHORT2 = {INF: "Inf.Mono", RECR: "Recr.MΦ", RESI: "Res.MΦ"}

# tidy per-interaction table
rec = []
for c in conds:
    for t in tps:
        for S in IDENTS:
            ks = f"{c}|{t}|{S}"
            if ks not in meanE.index: continue
            for R in IDENTS:
                kr = f"{c}|{t}|{R}"
                if kr not in meanE.index: continue
                for lig, recs in LR:
                    if fracE.at[ks, lig] < MIN_FRAC: continue
                    rbest = max(recs, key=lambda r: meanE.at[kr, r])
                    if fracE.at[kr, rbest] < MIN_FRAC: continue
                    s = meanE.at[ks, lig] * meanE.at[kr, rbest]
                    if s > 0:
                        rec.append((c, t, S, R, f"{lig}→{rbest}", s))
comm = pd.DataFrame(rec, columns=["cond", "tp", "sender", "receiver", "pair", "strength"])

# ── FIG A: differential heatmap (Burn − Sham) ────────────────────────────────────
agg = comm.groupby(["cond", "tp", "sender", "receiver"])["strength"].sum().reset_index()
agg["sr"] = agg["sender"].map(SHORT2) + " → " + agg["receiver"].map(SHORT2)
order_sr = [f"{SHORT2[s]} → {SHORT2[r]}" for s in IDENTS for r in IDENTS]
B = agg[agg.cond == "Burn"].pivot_table("strength", "sr", "tp").reindex(index=order_sr, columns=tps).fillna(0)
Sh = agg[agg.cond == "Sham"].pivot_table("strength", "sr", "tp").reindex(index=order_sr, columns=tps).fillna(0)
D = B - Sh
vlim = np.nanmax(np.abs(D.values)) or 1.0

fig, ax = plt.subplots(figsize=(1.1 * len(tps) + 3.5, 0.5 * len(order_sr) + 2))
im = ax.imshow(D.values, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
ax.set_xticks(range(len(tps))); ax.set_xticklabels(tps, fontsize=13, fontweight="bold")
ax.set_yticks(range(len(order_sr))); ax.set_yticklabels(order_sr, fontsize=12)
for i in range(len(order_sr)):
    for j in range(len(tps)):
        v = D.values[i, j]
        ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=9, fontweight="bold",
                color="white" if abs(v) > 0.6 * vlim else "black")
ax.set_title("Differential signaling  (Burn − Sham)\nred = stronger in Burn", fontsize=14, fontweight="bold")
ax.set_xlabel("Timepoint", fontsize=13, fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Δ strength", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGDIR_MAC / "cellchat_differential_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "cellchat_differential_heatmap.pdf", bbox_inches="tight", facecolor="white")
plt.show()

# ── FIG B: resident MΦ → monocyte/recruited L–R bubble plot ──────────────────────
sub = comm[(comm.sender == RESI) & (comm.receiver.isin([INF, RECR]))]
p = sub.groupby(["cond", "tp", "pair"])["strength"].sum().reset_index()
p["ttc"] = p["cond"] + " " + p["tp"]
cols = [f"{c} {t}" for c in ["Sham", "Burn"] for t in tps]
top = p.groupby("pair")["strength"].sum().sort_values(ascending=False).head(15).index.tolist()
mat = p[p.pair.isin(top)].pivot_table("strength", "pair", "ttc").reindex(index=top, columns=cols).fillna(0)
smax = mat.values.max() or 1.0

fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 3.5, 0.45 * len(top) + 2))
for i, pair in enumerate(top):
    for j, col in enumerate(cols):
        v = mat.loc[pair, col]
        if v > 0:
            ax.scatter(j, i, s=30 + 420 * (v / smax), c=[v], cmap="Reds", vmin=0, vmax=smax,
                       edgecolor="black", linewidths=0.4)
ax.axvline(len(tps) - 0.5, color="0.5", ls="--", lw=1.2)          # Sham | Burn divider
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=11, fontweight="bold")
for lbl in ax.get_xticklabels():
    lbl.set_color("#C0392B" if lbl.get_text().startswith("Burn") else "#2980B9")
ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=11)
ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(-0.5, len(top) - 0.5); ax.invert_yaxis()
ax.set_title("Resident MΦ → monocyte/recruited signaling", fontsize=14, fontweight="bold")
sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, smax)); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02); cb.set_label("interaction strength", fontsize=11, fontweight="bold")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIGDIR_MAC / "cellchat_resident_to_mono_bubble.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "cellchat_resident_to_mono_bubble.pdf", bbox_inches="tight", facecolor="white")
plt.show()


import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.sparse as sp
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
G1, G2 = "Arg1", "Nos2"
assert G1 in adata_mac.var_names and G2 in adata_mac.var_names
COL = {"neg": "#D5D8DC", "a": "#27AE60", "n": "#C724B1", "dbl": "#F39C12"}  # green Arg1, magenta Nos2, gold double
TYPE_PAL = {"Sham": "#2980B9", "Burn": "#C0392B"}

def _vec(ad, g):
    x = ad[:, g].X
    return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
a, n = _vec(adata_mac, G1), _vec(adata_mac, G2)
pa, pn = a > 0, n > 0
obs = adata_mac.obs; typ = obs["Type"].astype(str).values

SAMP_CANDS = ["Sample", "sample", "orig.ident", "orig_ident", "SampleID", "sample_id",
              "library", "Library", "mouse", "Mouse", "replicate", "Replicate", "batch"]
samp_col = next((c for c in SAMP_CANDS if c in obs.columns), None)

# per-sample % double-positive → Burn vs Sham test
dbl = pa & pn
psamp = (pd.DataFrame({"sample": obs[samp_col].astype(str).values, "Type": typ, "dbl": dbl.astype(int)})
         .groupby(["sample", "Type"])["dbl"].mean().mul(100).reset_index())
b = psamp.loc[psamp.Type == "Burn", "dbl"]; s = psamp.loc[psamp.Type == "Sham", "dbl"]
pval = ttest_ind(b, s, equal_var=False).pvalue if len(b) >= 2 and len(s) >= 2 else np.nan
stars = "n.s." if not np.isfinite(pval) else ("****" if pval<1e-4 else "***" if pval<1e-3 else "**" if pval<1e-2 else "*" if pval<0.05 else "n.s.")

rng, JIT = np.random.default_rng(0), 0.07
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharex=True, sharey=True)
for ax, ty in zip(axes, ["Sham", "Burn"]):
    m  = typ == ty
    aa = a[m] + rng.normal(0, JIT, m.sum()); nn = n[m] + rng.normal(0, JIT, m.sum())
    cats = [((~pa[m]) & (~pn[m]), COL["neg"], "—", 1, 6, 0.0),
            (pa[m] & (~pn[m]),    COL["a"],  f"{G1}+", 2, 9, 0.0),
            ((~pa[m]) & pn[m],    COL["n"],  f"{G2}+", 2, 9, 0.0),
            (pa[m] & pn[m],       COL["dbl"], f"{G1}+ {G2}+", 3, 16, 0.4)]
    for msk, c, lab, z, ss, ew in cats:
        ax.scatter(aa[msk], nn[msk], s=ss, c=c, alpha=0.65, linewidths=ew,
                   edgecolors="black" if ew else "none", rasterized=True, zorder=z)
    pct = (pa[m] & pn[m]).mean() * 100
    ax.text(0.97, 0.97, f"{G1}+{G2}+\n{pct:.1f}%", transform=ax.transAxes, ha="right", va="top",
            fontsize=15, fontweight="bold", color="#B8860B")
    ax.axhline(0, color="0.8", lw=0.8, zorder=0); ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_title(ty, fontsize=18, fontweight="bold", color=TYPE_PAL[ty])
    ax.set_xlabel(f"{G1} (log-norm)", fontsize=14, fontweight="bold")
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
axes[0].set_ylabel(f"{G2} (log-norm)", fontsize=14, fontweight="bold")

handles = [Line2D([0],[0], marker="o", linestyle="none", ms=9, mfc=COL[k], mec="black" if k=="dbl" else "none",
                  label=l) for k, l in [("a", f"{G1}+ only"), ("n", f"{G2}+ only"), ("dbl", f"{G1}+ {G2}+ (double)")]]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.06))
fig.suptitle(f"Arg1 / Nos2 co-expression — double+ Burn vs Sham: {stars} (p={pval:.1e})",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIGDIR_MAC / "F_arg1_nos2_coexpression_scatter.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "F_arg1_nos2_coexpression_scatter.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()


import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.sparse as sp
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_ind
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
G1, G2 = "Arg1", "Nos2"
COL = {"neg": "#D5D8DC", "a": "#CD6161", "dbl": "#730A0A", "n": "#DA2D1A"}
TYPE_PAL = {"Sham": "#2980B9", "Burn": "#C0392B"}

def _vec(ad, g):
    x = ad[:, g].X
    v = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
    return np.nan_to_num(v.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

a, n = _vec(adata_mac, G1), _vec(adata_mac, G2)
pa, pn = a > 0, n > 0
obs = adata_mac.obs; typ = obs["Type"].astype(str).values
SAMP_CANDS = ["Sample","sample","orig.ident","orig_ident","SampleID","sample_id",
              "library","Library","mouse","Mouse","replicate","Replicate","batch"]
samp_col = next((c for c in SAMP_CANDS if c in obs.columns), None)

dbl = pa & pn
psamp = (pd.DataFrame({"sample": obs[samp_col].astype(str).values, "Type": typ, "dbl": dbl.astype(int)})
         .groupby(["sample", "Type"])["dbl"].mean().mul(100).reset_index())
b = psamp.loc[psamp["Type"] == "Burn", "dbl"]; s = psamp.loc[psamp["Type"] == "Sham", "dbl"]
pval = ttest_ind(b, s, equal_var=False).pvalue if len(b) >= 2 and len(s) >= 2 else np.nan
stars = "n.s." if not np.isfinite(pval) else ("****" if pval<1e-4 else "***" if pval<1e-3 else "**" if pval<1e-2 else "*" if pval<0.05 else "n.s.")

xhi = max(float(np.nanpercentile(a, 99.5)), 0.5)
yhi = max(float(np.nanpercentile(n, 99.5)), 0.5)
_top = int(np.floor(min(xhi, yhi)))                      # highest integer inside BOTH ranges
TICKS = list(range(0, _top + 1)) if _top >= 1 else None  # same ticks on X and Y
rng, JIT = np.random.default_rng(0), 0.07
BOX = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharex=True, sharey=True)
for ax, ty in zip(axes, ["Sham", "Burn"]):
    m  = typ == ty
    aa = a[m] + rng.normal(0, JIT, m.sum()); nn = n[m] + rng.normal(0, JIT, m.sum())
    for msk, c, z, ss, ew in [((~pa[m])&(~pn[m]), COL["neg"], 1, 6, 0.0),
                              (pa[m]&(~pn[m]),     COL["a"],  2, 9, 0.0),
                              ((~pa[m])&pn[m],     COL["n"],  2, 9, 0.0),
                              (pa[m]&pn[m],        COL["dbl"],3, 16, 0.4)]:
        ax.scatter(aa[msk], nn[msk], s=ss, c=c, alpha=0.65, linewidths=ew,
                   edgecolors="black" if ew else "none", rasterized=True, zorder=z)
    lbl = f"{(pa[m]&pn[m]).mean()*100:.1f}% Arg1+ Nos2+"
    ax.text(0.26, 0.97, lbl, transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold", color=COL["dbl"], bbox=BOX, zorder=10)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0); ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlim(-0.3, xhi * 1.1); ax.set_ylim(-0.3, yhi * 1.1)
    if TICKS is not None:                                 # equal tick COUNT on X and Y
        ax.set_xticks(TICKS); ax.set_yticks(TICKS)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4)); ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_title(ty, fontsize=18, fontweight="bold", color=TYPE_PAL[ty])
    ax.set_xlabel(f"{G1} (log-norm)", fontsize=20, fontweight="bold")
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
axes[0].set_ylabel(f"{G2} (log-norm)", fontsize=20, fontweight="bold")

handles = [Line2D([0],[0], marker="o", linestyle="none", ms=9, mfc=COL[k], mec="black" if k=="dbl" else "none",
                  label=l) for k, l in [("a", f"{G1}+ only"), ("n", f"{G2}+ only"), ("dbl", f"{G1}+ {G2}+")]]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIGDIR_MAC / "F_arg1_nos2_coexpression_scatter.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "F_arg1_nos2_coexpression_scatter.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()


import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.sparse as sp
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
G1, G2, MIN_CELLS = "Arg1", "Nos2", 10

def _vec(ad, g):
    x = ad[:, g].X
    v = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
    return np.nan_to_num(v.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

obs = adata_mac.obs
dblcell = (_vec(adata_mac, G1) > 0) & (_vec(adata_mac, G2) > 0)     # Arg1+ Nos2+ per cell
typ = obs["Type"].astype(str).values
SAMP = ["Sample","sample","orig.ident","orig_ident","SampleID","sample_id",
        "library","Library","mouse","Mouse","replicate","Replicate","batch"]
samp = obs[next(c for c in SAMP if c in obs.columns)].astype(str).values

ORDERS = {
    "mac_identity": [c for c in (MAC_IDENTITIES if "MAC_IDENTITIES" in globals() else []) ],
    "macrophage_subtypes": [c for c in (mac_colors if "mac_colors" in globals() else [])],
}
def _st(p): return "n/a" if not np.isfinite(p) else ("****" if p<1e-4 else "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<0.05 else "n.s.")

def coexp_stats(group_col):
    g = obs[group_col].astype(str).values
    d = pd.DataFrame({"sample": samp, "Type": typ, "grp": g, "dbl": dblcell.astype(int)})
    d = d[~d["grp"].isin(["nan", "None", ""])]
    per = d.groupby(["grp", "sample", "Type"])["dbl"].agg(frac="mean", ncell="size").reset_index()
    per = per[per["ncell"] >= MIN_CELLS]
    order = [x for x in ORDERS.get(group_col, []) if x in set(per["grp"])] or sorted(per["grp"].unique())
    rows = []
    for grp in order:
        sub = per[per["grp"] == grp]
        sh = sub.loc[sub.Type == "Sham", "frac"].values * 100
        bu = sub.loc[sub.Type == "Burn", "frac"].values * 100
        p = ttest_ind(bu, sh, equal_var=False).pvalue if len(bu) >= 2 and len(sh) >= 2 else np.nan
        f = lambda v: f"{np.mean(v):.1f}±{(np.std(v, ddof=1)/np.sqrt(len(v))):.1f}" if len(v) > 1 else (f"{v[0]:.1f}" if len(v) else "—")
        rows.append({"Population": grp,
                     "Sham (%±SEM)": f"{f(sh)} (n={len(sh)})",
                     "Burn (%±SEM)": f"{f(bu)} (n={len(bu)})",
                     "Δ (pp)": (f"{np.mean(bu)-np.mean(sh):+.1f}" if len(bu) and len(sh) else "—"),
                     "p": p})
    df = pd.DataFrame(rows)
    ok = df["p"].notna()
    df["FDR"] = np.nan
    if ok.any(): df.loc[ok, "FDR"] = multipletests(df.loc[ok, "p"], method="fdr_bh")[1]
    df["sig"] = df["FDR"].map(_st)
    df["p"]   = df["p"].map(lambda v: f"{v:.2g}" if np.isfinite(v) else "n/a")
    df["FDR"] = df["FDR"].map(lambda v: f"{v:.2g}" if np.isfinite(v) else "n/a")
    return df

def render_table(df, title, stem):
    cols = ["Population", "Sham (%±SEM)", "Burn (%±SEM)", "Δ (pp)", "p", "FDR", "sig"]
    disp = df[cols]; nrow, ncol = disp.shape
    fig, ax = plt.subplots(figsize=(1.35 * ncol + 1.0, 0.5 * nrow + 1.1)); ax.axis("off")
    tbl = ax.table(cellText=disp.values, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.5)
    tbl.auto_set_column_width(col=list(range(ncol)))
    for j in range(ncol):
        c = tbl[0, j]; c.set_facecolor("#34495E"); c.set_text_props(color="white", fontweight="bold")
    for i in range(nrow):
        if str(df.iloc[i]["sig"]) not in ("n.s.", "n/a"):
            for j in range(ncol): tbl[i + 1, j].set_facecolor("#FDEDEC")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIGDIR_MAC / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGDIR_MAC / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

for col, ttl, stem in [("mac_identity", "", "table_coexp_mac_identity"),
                       ("macrophage_subtypes", "Arg1+ Nos2+ (% of population) — Burn vs Sham\nby macrophage subtype", "table_coexp_macrophage_subtypes")]:
    res = coexp_stats(col)
    res.to_csv(FIGDIR_MAC / f"{stem}.csv", index=False)
    print(f"\n=== {col} ===\n", res.to_string(index=False))
    render_table(res, ttl, stem)


import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.sparse as sp
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from pathlib import Path

FIGDIR_MAC = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
                  "burn_sham_scrnaseq_macs_20260608/figures/mac_identity")
G1, G2 = "Arg1", "Nos2"
COL = {"neg": "#D5D8DC", "a": "#27AE60", "n": "#8E44AD", "dbl": "#C0392B"}   # green / purple / red
TYPE_PAL = {"Sham": "#2980B9", "Burn": "#C0392B"}

def _vec(ad, g):
    x = ad[:, g].X
    v = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
    return np.nan_to_num(v.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

a, n = _vec(adata_mac, G1), _vec(adata_mac, G2)
pa, pn = a > 0, n > 0
obs = adata_mac.obs; typ = obs["Type"].astype(str).values
tp_all = (obs["Timepoint"].astype(str).values if "Timepoint" in obs.columns
          else pd.Series(obs["Type_Timepoint_C"].astype(str)).str.split().str[1].values)
TP_ORDER = [d for d in ["D7", "D10", "D14", "D19"] if d in set(tp_all)]
SAMP_CANDS = ["Sample","sample","orig.ident","orig_ident","SampleID","sample_id",
              "library","Library","mouse","Mouse","replicate","Replicate","batch"]
samp_col = next((c for c in SAMP_CANDS if c in obs.columns), None)

# per-timepoint Burn-vs-Sham on per-sample % double-positive (FDR across timepoints)
sdf = pd.DataFrame({"sample": obs[samp_col].astype(str).values, "Type": typ,
                    "Timepoint": tp_all, "dbl": (pa & pn).astype(int)})
psamp = sdf.groupby(["sample", "Type", "Timepoint"])["dbl"].mean().mul(100).reset_index()
rawp = {}
for tp in TP_ORDER:
    d = psamp[psamp["Timepoint"] == tp]
    bb = d.loc[d["Type"] == "Burn", "dbl"]; ss = d.loc[d["Type"] == "Sham", "dbl"]
    rawp[tp] = ttest_ind(bb, ss, equal_var=False).pvalue if len(bb) >= 2 and len(ss) >= 2 else np.nan
ok = [tp for tp in TP_ORDER if np.isfinite(rawp[tp])]
padj = {tp: np.nan for tp in TP_ORDER}
if ok: padj.update(dict(zip(ok, multipletests([rawp[tp] for tp in ok], method="fdr_bh")[1])))
def _st(p): return "n.s." if not np.isfinite(p) else ("****" if p<1e-4 else "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<0.05 else "n.s.")

xhi = max(float(np.nanpercentile(a, 99.5)), 0.5)
yhi = max(float(np.nanpercentile(n, 99.5)), 0.5)
rng, JIT = np.random.default_rng(0), 0.07
BOX = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.9)

nrow, ncol = 2, len(TP_ORDER)
fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.3 * nrow),
                         sharex=True, sharey=True, squeeze=False)
for r, ty in enumerate(["Sham", "Burn"]):
    for ci, tp in enumerate(TP_ORDER):
        ax = axes[r][ci]
        m = (typ == ty) & (tp_all == tp)
        aa = a[m] + rng.normal(0, JIT, m.sum()); nn = n[m] + rng.normal(0, JIT, m.sum())
        for msk, c, z, ss, ew in [((~pa[m])&(~pn[m]), COL["neg"], 1, 5, 0.0),
                                  (pa[m]&(~pn[m]),     COL["a"],  2, 8, 0.0),
                                  ((~pa[m])&pn[m],     COL["n"],  2, 8, 0.0),
                                  (pa[m]&pn[m],        COL["dbl"],3, 14, 0.4)]:
            ax.scatter(aa[msk], nn[msk], s=ss, c=c, alpha=0.65, linewidths=ew,
                       edgecolors="black" if ew else "none", rasterized=True, zorder=z)
        pct = (pa[m] & pn[m]).mean() * 100 if m.sum() else 0.0
        lbl = f"{pct:.1f}% dbl+"
        if ty == "Burn":
            lbl += f"\nvs Sham {_st(padj[tp])}"
        ax.text(0.04, 0.97, lbl, transform=ax.transAxes, ha="left", va="top",
                fontsize=11, fontweight="bold", color=COL["dbl"], bbox=BOX, zorder=10)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0); ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlim(-0.3, xhi * 1.1); ax.set_ylim(-0.3, yhi * 1.1)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
        if r == 0:
            ax.set_title(tp, fontsize=17, fontweight="bold", pad=8)
        if r == nrow - 1:
            ax.set_xlabel(f"{G1} (log-norm)", fontsize=13, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"{G2} (log-norm)", fontsize=13, fontweight="bold")
            ax.annotate(ty, xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90,
                        ha="center", va="center", fontsize=17, fontweight="bold", color=TYPE_PAL[ty])

handles = [Line2D([0],[0], marker="o", linestyle="none", ms=9, mfc=COL[k], mec="black" if k=="dbl" else "none",
                  label=l) for k, l in [("a", f"{G1}+ only"), ("n", f"{G2}+ only"), ("dbl", f"{G1}+ {G2}+")]]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIGDIR_MAC / "F_arg1_nos2_coexpression_bytimepoint.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(FIGDIR_MAC / "F_arg1_nos2_coexpression_bytimepoint.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()


rng = np.random.default_rng(0)
subs = [x for x in (mac_colors if "mac_colors" in globals() else []) if x in set(obs["macrophage_subtypes"].astype(str))]
subs = subs or sorted(set(obs["macrophage_subtypes"].astype(str)))

co = pd.DataFrame({"sample": obs[samp_col].astype(str).values, "Type": typ,
                   "subtype": obs["macrophage_subtypes"].astype(str).values, "dbl": (pa & pn).astype(int)})
g  = co.groupby(["sample", "Type", "subtype"])
sd = g["dbl"].mean().mul(100).reset_index(name="pct"); sd["ncell"] = g["dbl"].size().values
sd = sd[sd["ncell"] >= 10]
subs = [c for c in subs if c in set(sd["subtype"])]
assert not sd.empty and subs, "min-cell filter emptied the table — lower the 10-cell cutoff"

import statsmodels.stats.multitest as mt
pv = {}
for c in subs:
    d = sd[sd["subtype"] == c]                                   # bracket access, not .sub
    bb = d.loc[d["Type"] == "Burn", "pct"]; ssv = d.loc[d["Type"] == "Sham", "pct"]
    pv[c] = ttest_ind(bb, ssv, equal_var=False).pvalue if len(bb) >= 2 and len(ssv) >= 2 else np.nan
ok = [c for c in subs if np.isfinite(pv[c])]
padj = {c: np.nan for c in subs}
if ok: padj.update(dict(zip(ok, mt.multipletests([pv[c] for c in ok], method="fdr_bh")[1])))
def _st(p): return "" if not np.isfinite(p) else ("***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "")

x = np.arange(len(subs)); w = 0.38
fig, ax = plt.subplots(figsize=(1.05*len(subs)+1.5, 4.2))
for k, cond in enumerate(["Sham", "Burn"]):
    xs = x + (k-0.5)*w; means, sems = [], []
    for j, c in enumerate(subs):
        v = sd[(sd["subtype"] == c) & (sd["Type"] == cond)]["pct"].values
        means.append(float(np.mean(v)) if len(v) else 0.0)
        sems.append(float(v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else 0.0)
        ax.scatter(np.full(len(v), xs[j]) + rng.uniform(-.06, .06, len(v)), v, s=16, color="black", zorder=5)
    ax.bar(xs, means, width=w, color=TYPE_PAL[cond], edgecolor="black", lw=1, label=cond, zorder=2)
    ax.errorbar(xs, means, yerr=sems, fmt="none", ecolor="black", capsize=3, lw=1.2, zorder=4)

ymax = max(float(sd["pct"].max()), 1.0); ax.set_ylim(0, ymax * 1.25)
for j, c in enumerate(subs):
    if _st(padj[c]): ax.text(x[j], ymax*1.05, _st(padj[c]), ha="center", va="bottom", fontsize=15, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(subs, rotation=30, ha="right", fontsize=12, fontweight="bold")
ax.set_ylabel(f"% {G1}+ {G2}+ (of subtype)", fontsize=14, fontweight="bold")
ax.legend(frameon=False, fontsize=12)
for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
fig.tight_layout()
fig.savefig(FIGDIR_MAC / "F_arg1_nos2_doublepos_by_subtype.pdf", bbox_inches="tight", facecolor="white")
plt.show()

