"""MΦ-Recruited only:  per macrophage_subtypes × Timepoint  DE + Volcano + GO

Source: macrophages_resident_recruited.ipynb
Libraries: -
Key calls: gp.enrichr, plt.close, plt.show, plt.subplots, rank_genes_groups, sc.pp.log1p, sc.pp.normalize_total, sc.tl.rank_genes_groups, volcano
"""

# ══════════════════════════════════════════════════════════════════════════════
# MΦ-Recruited only:  per macrophage_subtypes × Timepoint  DE + Volcano + GO
#   (Burn vs Sham)
# Subsets adata_mac to mac_identity == "MΦ-Recruited", RE-NORMALIZES the subset
# from raw counts, then reuses draw_volcano / make_enrichment_tile / pick_labels /
# prepare_celltype_for_DE / _slug / contains_excluded / thresholds / colors.
# ══════════════════════════════════════════════════════════════════════════════
RECRUITED_IDENT = "Inflammatory Monocytes"
SUB_COL         = "macrophage_subtypes"
assert "mac_identity" in adata_mac.obs.columns
assert SUB_COL in adata_mac.obs.columns

FIGDIR_RECR = OUT / "figures" / "mac_recruited_subtypes"
FIGDIR_RECR.mkdir(parents=True, exist_ok=True)

# ── subset to the MΦ-Recruited compartment ────────────────────────────────────
recr_mask = (adata_mac.obs["mac_identity"].astype(str) == RECRUITED_IDENT).values
adata_recr = adata_mac[recr_mask].copy()
print(f"Inflammatory Monocytes subset: {adata_recr.n_obs} cells")

# ── re-normalize the subset from raw counts (prefer .raw, then 'counts' layer) ─
if adata_recr.raw is not None and adata_recr.raw.X.max() > 50:
    adata_recr.X = adata_recr.raw[:, adata_recr.var_names].X.copy()
    print("Restored counts from .raw")
elif "counts" in adata_recr.layers and adata_recr.layers["counts"].max() > 50:
    adata_recr.X = adata_recr.layers["counts"].copy()
    print("Restored counts from layers['counts']")
elif adata_recr.X.max() > 50:
    print("X already looks like counts — normalizing as-is")
else:
    print("WARNING: no raw counts found; X already log-normalized — skipping re-normalization")

if adata_recr.X.max() > 50:                       # only (re)normalize true counts
    sc.pp.normalize_total(adata_recr, target_sum=1e4)
    sc.pp.log1p(adata_recr)
    print("Re-normalized: normalize_total(1e4) + log1p")

# ── subtypes actually present in the recruited compartment ────────────────────
_present = [s for s in adata_recr.obs[SUB_COL].astype(str).unique()
            if s.lower() not in ("nan", "none", "")]
if hasattr(adata_recr.obs[SUB_COL], "cat"):       # honor categorical order if set
    _order = [s for s in adata_recr.obs[SUB_COL].cat.categories if s in _present]
    SUBTYPES = _order + [s for s in _present if s not in _order]
else:
    SUBTYPES = _present
print("subtypes:", SUBTYPES)
print("timepoints:", ALL_TIMEPOINTS)

# ── 1. DE per (subtype, timepoint) ────────────────────────────────────────────
de_recr_sub = {sub: {} for sub in SUBTYPES}
for sub in SUBTYPES:
    for tp in ALL_TIMEPOINTS:
        sel = ((adata_recr.obs["Timepoint"].astype(str) == tp).values &
               (adata_recr.obs[SUB_COL].astype(str)      == sub).values)
        adata_sub = adata_recr[sel].copy()
        n_b = (adata_sub.obs["Type"] == "Burn").sum()
        n_s = (adata_sub.obs["Type"] == "Sham").sum()
        print(f"\n[{sub} | {tp}] {adata_sub.n_obs} cells  Burn={n_b}  Sham={n_s}")
        if n_b < 3 or n_s < 3:
            print("  Skipping — too few cells."); de_recr_sub[sub][tp] = None; continue

        adata_sub = prepare_celltype_for_DE(adata_sub, "Macrophages", normalize=False)
        sc.tl.rank_genes_groups(adata_sub, groupby="Type", groups=["Burn"], reference="Sham",
                                method="wilcoxon", use_raw=False, pts=True, key_added="rgg")
        df = sc.get.rank_genes_groups_df(adata_sub, group="Burn", key="rgg")
        df = df.dropna(subset=["logfoldchanges", "pvals_adj"])
        df = df.rename(columns={"names": "gene", "logfoldchanges": "lfc", "pvals_adj": "padj"})
        df["nlp"] = -np.log10(df["padj"].clip(lower=1e-300))
        de_recr_sub[sub][tp] = df

        nb = ((df.padj < FDR_THRESH) & (df.lfc >  LFC_THRESH)).sum()
        ns = ((df.padj < FDR_THRESH) & (df.lfc < -LFC_THRESH)).sum()
        print(f"  Burn↑: {nb}   Sham↑: {ns}")
        df.to_csv(FIGDIR_RECR / f"de_recruited_{_slug(sub)}_{_slug(tp)}_burn_vs_sham.csv", index=False)

# ── 2. Volcano grid — one figure per subtype (cols = timepoints) ──────────────
for sub in SUBTYPES:
    valid = [tp for tp in ALL_TIMEPOINTS if de_recr_sub[sub].get(tp) is not None]
    if not valid:
        print(f"No valid timepoints for {sub} — skipping volcano."); continue
    ncols = min(4, len(valid)); nrows = int(np.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    for i, tp in enumerate(valid):
        draw_volcano(axes_flat[i], de_recr_sub[sub][tp], f"Inflammatory Monocytes · {sub}\n{tp}")
    for ax in axes_flat[len(valid):]:
        ax.set_visible(False)
    fig.tight_layout(); fig.canvas.draw()
    fig.savefig(FIGDIR_RECR / f"volcano_recruited_{_slug(sub)}_timepoint_burn_vs_sham.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR_RECR / f"volcano_recruited_{_slug(sub)}_timepoint_burn_vs_sham.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)
    print(f"Saved volcano: Inflammatory Monocytes · {sub}")

# ── 3. GO enrichment tile — one per (subtype, timepoint) ─────────────────────
for sub in SUBTYPES:
    for tp in ALL_TIMEPOINTS:
        df = de_recr_sub[sub].get(tp)
        if df is None:
            continue
        sig = df[df["padj"] < FDR_THRESH]
        gene_sets = {"Burn": sig[sig["lfc"] >  LFC_THRESH]["gene"].tolist(),
                     "Sham": sig[sig["lfc"] < -LFC_THRESH]["gene"].tolist()}
        rows = []
        for direction, glist in gene_sets.items():
            print(f"  [{sub} | {tp}] {direction}: {len(glist)} genes")
            if len(glist) < 5:
                continue
            try:
                enr = gp.enrichr(gene_list=glist, gene_sets=GO_LIB,
                                 organism="mouse", outdir=None, verbose=False)
                results = enr.res2d.sort_values("Adjusted P-value").copy()
                results["term_clean"] = results["Term"].apply(
                    lambda t: str(t).split("(")[0].strip().replace("_", " "))
                results["term_clean"] = results["term_clean"].apply(
                    lambda t: t[0].upper() + t[1:] if t else t)
                results = results[~results["term_clean"].apply(contains_excluded)]
                top = results.head(N_TERMS); N = len(glist)
                for _, row in top.iterrows():
                    k, n = (int(x) for x in str(row.get("Overlap", "1/1")).split("/"))
                    fe = (k / N) / (n / M_BG) if N > 0 and n > 0 else 1.0
                    rows.append({"pathway_clean": row["term_clean"],
                                 "padj": float(row["Adjusted P-value"]),
                                 "FoldEnrichment": round(fe, 1),
                                 "Count": k, "directionality": direction})
            except Exception as e:
                print(f"    ERROR: {e}")
        make_enrichment_tile(pd.DataFrame(rows),
                             output_name=FIGDIR_RECR / f"enrichment_tile_recruited_{_slug(sub)}_{_slug(tp)}.pdf",
                             title=f"Inflammatory Monocytes · {sub} — {tp}\nBurn vs Sham GO BP", tile_size=0.7)

