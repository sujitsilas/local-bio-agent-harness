"""FOUR-WAY plots of Burn-vs-Sham DE across consecutive timepoints, per identity

Source: macrophages_resident_recruited.ipynb
Libraries: adjustText, matplotlib, numpy, re, scipy
Key calls: .plot, adjust_text, def _de_bvs, def _fourway_tp, def _top, plt.close, plt.show, plt.subplots, rank_genes_groups, sc.tl.rank_genes_groups, scatter
"""

# ══════════════════════════════════════════════════════════════════════════════
# FOUR-WAY plots of Burn-vs-Sham DE across consecutive timepoints, per identity.
#   Axes = Log2FC(Burn/Sham) at two timepoints;  diagonal = stable Burn/Sham diff.
#   x = earlier tp, y = later tp.  Pairs: D7→D10, D10→D14, D14→D19.
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, scanpy as sc, matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.lines import Line2D

TPS      = ["D7", "D10", "D14", "D19"]
TP_PAIRS = [("D7", "D10"), ("D10", "D14"), ("D14", "D19")]   # (earlier=x, later=y)
MUST_LABEL = ["Arg1", "Nos2", "Ccr2"]
# point color = WHEN the Burn/Sham difference is significant (fixed, not red/blue → reserved for quadrants)
CAT2 = {"ns": "#D9D9D9", "burn": "#C0392B", "sham": "#2980B9", "other": "#111111"}
SHORT_ID = {"Inflammatory Monocytes": "Inflam. Mono."}   # shorten long identity names in titles

import re
RBC_RE = re.compile(r"^(Hb[ab]|Hbq|Alas2|Bpgm|Gypa|Slc4a1|Rhag)", re.I)   # ambient RBC / hemoglobin


BURN, SHAM, NEUT = "#C0392B", "#2980B9", "#9E9E9E"

_idv  = adata_mac.obs["mac_identity"].astype(str)
_have = sorted(_idv.unique()); _find = lambda *ks: next((h for h in _have if any(k in h.lower() for k in ks)), None)
INF, RECR, RESI= _find("inflamm", "mono"), _find("recruit"), _find("resident")
IDENTS = [x for x in [INF, RECR, RESI] if x]

def _de_bvs(ident, tp, min_pct=0.20):
    """Burn vs Sham DE within one identity at one timepoint."""
    sel = ((_idv == ident).values & (adata_mac.obs["Timepoint"].astype(str) == tp).values)
    a = adata_mac[sel].copy()
    nb = int((a.obs["Type"].astype(str) == "Burn").sum()); ns = int((a.obs["Type"].astype(str) == "Sham").sum())
    if nb < 3 or ns < 3:
        print(f"[{ident} | {tp}] skip (Burn={nb} Sham={ns})"); return None
    a = prepare_celltype_for_DE(a, "Macrophages", normalize=False)
    a.obs["Type"] = pd.Categorical(a.obs["Type"].astype(str))
    sc.tl.rank_genes_groups(a, "Type", groups=["Burn"], reference="Sham",
                            method="wilcoxon", use_raw=False, pts=True, key_added="rgg")
    df = sc.get.rank_genes_groups_df(a, group="Burn", key="rgg").dropna(subset=["logfoldchanges", "pvals_adj"])
    pct = [c for c in df.columns if c.startswith("pct_nz")] or [c for c in df.columns if "pct" in c.lower()]
    if len(pct) >= 2:                                       # detection filter (optional)
        df = df[df[pct].max(axis=1) >= min_pct]
    df = df.rename(columns={"names": "gene", "logfoldchanges": "lfc", "pvals_adj": "padj"})
    n0 = len(df); df = df[~df["gene"].str.match(RBC_RE)]    # strip ambient RBC/Hb only
    if len(df) < n0:
        print(f"    [{ident} | {tp}] removed {n0 - len(df)} ambient RBC/Hb genes")
    return df




def _fourway_tp(ax, de_e, de_l, earlier, later, ident, AXLIM=20.0):
    m = de_e[["gene", "lfc", "padj"]].merge(de_l[["gene", "lfc", "padj"]], on="gene", suffixes=("_e", "_l"))
    for c in ("lfc_e", "lfc_l"):
        m[c] = m[c].replace([np.inf, -np.inf], np.nan)
    m = m.dropna(subset=["lfc_e", "lfc_l"])
    se = (m.padj_e < FDR_THRESH) & (m.lfc_e.abs() > LFC_THRESH)     # sig Burn/Sham at earlier
    sl = (m.padj_l < FDR_THRESH) & (m.lfc_l.abs() > LFC_THRESH)     # sig Burn/Sham at later
    sig = se | sl
    # color by quadrant: burn = up both, sham = down both, other = flips, ns = not significant
    m["grp"] = np.where(~sig, "ns",
               np.where((m.lfc_e > 0) & (m.lfc_l > 0), "burn",
               np.where((m.lfc_e < 0) & (m.lfc_l < 0), "sham", "other")))

    from scipy.stats import pearsonr, spearmanr
    r  = pearsonr(m.lfc_e, m.lfc_l)[0]
    rs = spearmanr(m.lfc_e, m.lfc_l)[0]
    print(f"    corr(LFC {earlier}, LFC {later}): Pearson={r:+.2f}  Spearman={rs:+.2f}")

    # quadrant shading = Burn/Sham direction consistency
    ax.axhspan(0,  AXLIM, xmin=0.5, xmax=1.0, color=BURN, alpha=0.06, zorder=0)   # UR Burn↑ both
    ax.axhspan(-AXLIM, 0, xmin=0.0, xmax=0.5, color=SHAM, alpha=0.06, zorder=0)   # LL Sham↑ both
    ax.axhspan(0,  AXLIM, xmin=0.0, xmax=0.5, color=NEUT, alpha=0.05, zorder=0)   # UL flip → Burn
    ax.axhspan(-AXLIM, 0, xmin=0.5, xmax=1.0, color=NEUT, alpha=0.05, zorder=0)   # LR flip → Sham
    ax.text(0.985, 0.985, "Burn↑", transform=ax.transAxes, ha="right", va="top",
            fontsize=18, fontweight="bold", color=BURN, zorder=2)
    ax.text(0.015, 0.015, "Sham↑", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=18, fontweight="bold", color=SHAM, zorder=2)
    ax.text(0.015, 0.985, "Emerging Burn↑", transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontstyle="italic", color=NEUT, zorder=2)
    ax.text(0.985, 0.015, "Transient Burn↑", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=14, fontstyle="italic", color=NEUT, zorder=2)

    for c in ("ns", "other", "sham", "burn"):                       # burn/sham drawn on top
        d = m[m.grp == c]
        ax.scatter(d.lfc_e.clip(-AXLIM, AXLIM), d.lfc_l.clip(-AXLIM, AXLIM),
                   s=6 if c == "ns" else 15, c=CAT2[c], alpha=0.30 if c == "ns" else 0.85,
                   linewidths=0, rasterized=True, zorder=1 if c == "ns" else 3)
    ax.axhline(0, color="grey", lw=0.9, ls=":"); ax.axvline(0, color="grey", lw=0.9, ls=":")
    ax.plot([-AXLIM, AXLIM], [-AXLIM, AXLIM], color="grey", lw=0.9, ls="--", alpha=0.6)
    ax.set_xlim(-AXLIM, AXLIM); ax.set_ylim(-AXLIM, AXLIM)

    def _top(mask, key, n):
        d = m[mask]; return d.loc[d[key].abs().sort_values(ascending=False).index[:n]]
    lab = pd.concat([_top(m.grp == "burn", "lfc_l", 6), _top(m.grp == "sham", "lfc_e", 5),
                     _top(m.grp == "other", "lfc_l", 3), m[m.gene.isin(MUST_LABEL)]]).drop_duplicates("gene")
    texts = [ax.text(np.clip(r["lfc_e"], -AXLIM, AXLIM), np.clip(r["lfc_l"], -AXLIM, AXLIM), r["gene"],
                     fontsize=20, fontweight="bold", ha="center", color=CAT2[r["grp"]], zorder=6,
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
             for _, r in lab.iterrows()]
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#7F8C8D", lw=0.5),
                    expand=(1.9, 2.3), force_text=(1.2, 1.6), force_static=(0.5, 0.7),
                    force_explode=(0.5, 0.7), force_pull=(0.01, 0.01),
                    max_move=40, min_arrow_len=6,
                    only_move={"text": "xy", "static": "xy", "explode": "xy"},
                    ensure_inside_axes=True, time_lim=15.0)


    print(f"[{ident} | {earlier}->{later}]  Burn-specific={int((m.grp=='burn').sum())}  "
          f"Sham-specific={int((m.grp=='sham').sum())}  Other(flip)={int((m.grp=='other').sum())}  "
          f"n.s.={int((m.grp=='ns').sum())}")
    ax.set_xlabel(f"Log$_2$FC Burn/Sham  ({earlier})", fontsize=23, fontweight="bold")
    ax.set_ylabel(f"Log$_2$FC Burn/Sham  ({later})", fontsize=23, fontweight="bold")
    ax.set_title(f"{SHORT_ID.get(ident, ident)} — {earlier} → {later}", fontsize=23, fontweight="bold")


    return m



for ident in IDENTS:
    de = {tp: _de_bvs(ident, tp) for tp in TPS}
    valid = [(e, l) for (e, l) in TP_PAIRS if de.get(e) is not None and de.get(l) is not None]
    if not valid:
        continue
    fig, axes = plt.subplots(1, len(valid), figsize=(7.2 * len(valid), 7), squeeze=False)
    for ax, (e, l) in zip(axes[0], valid):
        m = _fourway_tp(ax, de[e], de[l], e, l, ident)
        m.assign(identity=ident, x_tp=e, y_tp=l).to_csv(
            FIGDIR_MAC / f"fourway_bvs_{_slug(ident)}_{e}_vs_{l}.csv", index=False)
        handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CAT2[c], markeredgecolor="none",
                      markersize=10, label=lb) for c, lb in
               [("burn", "Burn-specific"), ("sham", "Sham-specific"),
                ("other", "Other (direction flips)"), ("ns", "n.s.")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=13)
    fig.tight_layout()
    fn = f"fourway_bvs_temporal_{_slug(ident)}"
    fig.savefig(FIGDIR_MAC / f"{fn}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR_MAC / f"{fn}.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

