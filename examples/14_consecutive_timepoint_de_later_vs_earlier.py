"""Consecutive-timepoint DE (LATER vs EARLIER), per identity × condition

Source: macrophages_resident_recruited.ipynb
Libraries: adjustText, numpy
Key calls: adjust_text, def _run_tp, def _volcano_tp, gp.enrichr, plt.close, plt.show, plt.subplots, rank_genes_groups, sc.tl.rank_genes_groups, scatter, volcano
"""

# ══════════════════════════════════════════════════════════════════════════════
# Consecutive-timepoint DE (LATER vs EARLIER), per identity × condition:
#   D10 vs D7, D14 vs D10, D19 vs D14  for Inflammatory Monocytes & MΦ-Recruited,
#   in Burn and Sham.  Volcano + GO tile colored by TIMEPOINT (no Burn/Sham red/blue).
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, scanpy as sc, matplotlib.pyplot as plt
from adjustText import adjust_text

TP_PAL = {"D7": "#E69F00", "D10": "#009E73", "D14": "#CC79A7", "D19": "#0072B2"}  # temporal, not red/blue
PAIRS  = [("D10", "D7"), ("D14", "D10"), ("D19", "D14")]      # (later, earlier)
MUST_LABEL = ["Arg1", "Nos2", "Ccr2"]
N_LABEL_DRAW = min(N_LABEL, 12)

_idv  = adata_full.obs["mac_identity"].astype(str)
_have = sorted(_idv.unique()); _find = lambda *ks: next((h for h in _have if any(k in h.lower() for k in ks)), None)
INF, RECR = _find("inflamm", "mono"), _find("recruit")
IDENTS, CONDS = [x for x in [INF, RECR] if x], ["Sham", "Burn"]

def _run_tp(ident, cond, later, earlier):
    sel = ((_idv == ident).values &
           (adata_full.obs["Type"].astype(str) == cond).values &
           (adata_full.obs["Timepoint"].astype(str).isin([later, earlier])).values)
    a = adata_full[sel].copy()
    n_l = int((a.obs["Timepoint"].astype(str) == later).sum())
    n_e = int((a.obs["Timepoint"].astype(str) == earlier).sum())
    print(f"[{ident} | {cond}] {later} vs {earlier}:  {later}={n_l}  {earlier}={n_e}")
    if n_l < 3 or n_e < 3:
        print("  skip — too few cells"); return None
    a = prepare_celltype_for_DE(a, "Macrophages", normalize=False)
    a.obs["Timepoint"] = pd.Categorical(a.obs["Timepoint"].astype(str))
    sc.tl.rank_genes_groups(a, "Timepoint", groups=[later], reference=earlier,
                            method="wilcoxon", use_raw=False, pts=True, key_added="rgg")
    df = (sc.get.rank_genes_groups_df(a, group=later, key="rgg")
            .dropna(subset=["logfoldchanges", "pvals_adj"])
            .rename(columns={"names": "gene", "logfoldchanges": "lfc", "pvals_adj": "padj"}))
    df["nlp"] = -np.log10(df["padj"].clip(lower=1e-300))
    return df

# color-blind-safe (Okabe-Ito) timepoint palette — distinct under all CVD types, no burn/sham red-blue

def _volcano_tp(ax, df, later, earlier, title):
    cu, cd = TP_PAL.get(later, "#444"), TP_PAL.get(earlier, "#999")
    up = (df.padj < FDR_THRESH) & (df.lfc >  LFC_THRESH)      # higher at LATER tp  → right side
    dn = (df.padj < FDR_THRESH) & (df.lfc < -LFC_THRESH)      # higher at EARLIER tp → left side
    ax.scatter(df.lfc[~up & ~dn], df.nlp[~up & ~dn], c=NS_COL, s=5, alpha=0.4, linewidths=0, rasterized=True)
    ax.scatter(df.lfc[dn], df.nlp[dn], c=cd, s=11, alpha=0.85, linewidths=0, rasterized=True)
    ax.scatter(df.lfc[up], df.nlp[up], c=cu, s=11, alpha=0.85, linewidths=0, rasterized=True)
    ax.axhline(-np.log10(FDR_THRESH), color="#7F8C8D", lw=0.9, ls="--", alpha=0.6)
    ax.axvline( LFC_THRESH, color="#7F8C8D", lw=0.9, ls="--", alpha=0.6)
    ax.axvline(-LFC_THRESH, color="#7F8C8D", lw=0.9, ls="--", alpha=0.6)

    dsig = df[(df.padj < FDR_THRESH) & (df.lfc.abs() > LFC_THRESH)]
    nd = 8                                                    # fewer labels → less crowding
    lab = pd.concat([pick_labels(dsig[dsig.lfc > 0], nd, ascending_lfc=False),
                     pick_labels(dsig[dsig.lfc < 0], nd, ascending_lfc=True),
                     df[df.gene.isin(MUST_LABEL)]]).drop_duplicates("gene")

    #ax.set_xlim(-25, 25); ax.set_ylim(-0.5, max(df.nlp.max() * 1.28, 1.0))   # extra top headroom
    _lf  = df["lfc"].values[np.isfinite(df["lfc"].values)]            # ignore ±inf lfc
    xabs = max(np.abs(_lf).max() * 1.05, 1.0) if _lf.size else 10.0   # symmetric, 5% pad, floor at 1
    ax.set_xlim(-xabs, xabs); ax.set_ylim(-0.5, max(df.nlp.max() * 1.28, 1.0))


    # DEG counts on their matching sides
    t_dn = ax.text(0.03, 0.97, f"{earlier}↑ {int(dn.sum())}", transform=ax.transAxes, fontsize=16,
                   va="top", ha="left",  color=cd, fontweight="bold")        # earlier = left
    t_up = ax.text(0.97, 0.97, f"{later}↑ {int(up.sum())}",  transform=ax.transAxes, fontsize=16,
                   va="top", ha="right", color=cu, fontweight="bold")        # later = right

    texts = [ax.text(r.lfc, r.nlp, r.gene, fontsize=14, color="black", fontweight="bold", ha="center")
             for _, r in lab.iterrows()]
    if texts:
        adjust_text(texts, ax=ax, objects=[t_up, t_dn],
                    arrowprops=dict(arrowstyle="-", color="#7F8C8D", lw=0.6),
                    expand=(1.4, 1.7), force_text=(0.7, 1.1), force_static=(0.25, 0.4),
                    force_pull=(0.01, 0.01), max_move=25, min_arrow_len=4,
                    only_move={"text": "xy", "static": "xy", "explode": "xy"},
                    ensure_inside_axes=True, time_lim=6.0)

    ax.set_xlabel("Log$_2$ FC  (later / earlier)", fontsize=24, fontweight="bold")
    ax.set_ylabel("$-$Log$_{10}$(padj)", fontsize=24, fontweight="bold")
    ax.set_title(title, fontsize=22, fontweight="bold"); ax.grid(False)


# ── per identity × condition: 1×3 volcano grid + a GO tile per pair ───────────
for ident in IDENTS:
    for cond in CONDS:
        des   = {p: _run_tp(ident, cond, *p) for p in PAIRS}
        valid = [p for p in PAIRS if des[p] is not None]
        if not valid:
            continue
        fig, axes = plt.subplots(1, len(valid), figsize=(7 * len(valid), 7), squeeze=False)
        for ax, (later, earlier) in zip(axes[0], valid):
            _volcano_tp(ax, des[(later, earlier)], later, earlier, f"{ident} · {cond}\n{later} vs {earlier}")
        fig.tight_layout()
        fn = f"volcano_tp_{_slug(ident)}_{_slug(cond)}_consecutive"
        fig.savefig(FIGDIR_MAC / f"{fn}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(FIGDIR_MAC / f"{fn}.png", dpi=300, bbox_inches="tight")
        plt.show(); plt.close(fig)

        for later, earlier in valid:
            df  = des[(later, earlier)]; sig = df[df.padj < FDR_THRESH]
            gene_sets = {earlier: sig[sig.lfc < -LFC_THRESH].gene.tolist(),   # earlier column first (time L→R)
                         later:   sig[sig.lfc >  LFC_THRESH].gene.tolist()}
            rows = []
            for direction, glist in gene_sets.items():
                print(f"  [{ident} | {cond}] {direction}↑: {len(glist)} genes")
                if len(glist) < 5:
                    continue
                try:
                    res = gp.enrichr(gene_list=glist, gene_sets=GO_LIB, organism="mouse", outdir=None, verbose=False).res2d.sort_values("Adjusted P-value").copy()
                    res["term_clean"] = res["Term"].apply(lambda t: str(t).split("(")[0].strip().replace("_", " "))
                    res["term_clean"] = res["term_clean"].apply(lambda t: t[0].upper() + t[1:] if t else t)
                    res = res[~res["term_clean"].apply(contains_excluded)]
                    for _, row in res.head(N_TERMS).iterrows():
                        k, n = (int(x) for x in str(row.get("Overlap", "1/1")).split("/"))
                        rows.append({"pathway_clean": row["term_clean"], "padj": float(row["Adjusted P-value"]),
                                     "FoldEnrichment": round((k / len(glist)) / (n / M_BG), 1) if n else 1.0,
                                     "Count": k, "directionality": direction})
                except Exception as e:
                    print("    ERROR:", e)
            make_enrichment_tile(pd.DataFrame(rows),
                                 output_name=FIGDIR_MAC / f"enrichment_tile_tp_{_slug(ident)}_{_slug(cond)}_{_slug(later)}_vs_{_slug(earlier)}.pdf",
                                 title=f"{ident} · {cond}\nGO BP — {later} vs {earlier}", tile_size=0.7)

