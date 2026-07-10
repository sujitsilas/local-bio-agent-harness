"""Per macrophage_subtypes × Timepoint:  DE + Volcano + GO  (Burn vs Sham)

Source: macrophages_resident_recruited.ipynb
Libraries: -
Key calls: gp.enrichr, plt.close, plt.show, plt.subplots, rank_genes_groups, sc.tl.rank_genes_groups, volcano
"""

# ══════════════════════════════════════════════════════════════════════════════
# Per macrophage_subtypes × Timepoint:  DE + Volcano + GO  (Burn vs Sham)
# Reuses helpers/constants from the identity cell (_slug, prepare_celltype_for_DE,
# pick_labels, make_enrichment_tile, contains_excluded, thresholds, colors, FIGDIR_MAC,
# ALL_TIMEPOINTS, N_LABEL, FS_ANNOT, GO_LIB, M_BG, N_TERMS, draw_volcano, get_contamination_genes)
# ══════════════════════════════════════════════════════════════════════════════
ID_COL = "macrophage_subtypes"
assert ID_COL in adata_mac.obs.columns, f"{ID_COL} not in adata_mac.obs"

# subtypes present, ordered by the mac_colors palette (lineage order), NaN dropped
_order  = list(mac_colors.keys()) if "mac_colors" in globals() else []
present = [c for c in adata_mac.obs[ID_COL].astype(str).unique()
          if c.lower() not in ("nan", "none", "")]
SUBTYPES = [s for s in _order if s in present] + [s for s in sorted(present) if s not in _order]
print("subtypes:", SUBTYPES)
print("timepoints:", ALL_TIMEPOINTS)

MUST_LABEL   = ["Arg1", "Nos2", "Ccr2"]
N_LABEL_DRAW = min(N_LABEL, 12)

# ── 1. DE per (subtype, timepoint) ────────────────────────────────────────────
de_sub_tp = {subt: {} for subt in SUBTYPES}
for subt in SUBTYPES:
    for tp in ALL_TIMEPOINTS:
        sel = ((adata_mac.obs['Timepoint'].astype(str) == tp).values &
               (adata_mac.obs[ID_COL].astype(str)       == subt).values)
        adata_sub = adata_mac[sel].copy()
        n_b = (adata_sub.obs['Type'] == 'Burn').sum()
        n_s = (adata_sub.obs['Type'] == 'Sham').sum()
        print(f"\n[{subt} | {tp}] {adata_sub.n_obs} cells  Burn={n_b}  Sham={n_s}")
        if n_b < 3 or n_s < 3:
            print("  Skipping — too few cells."); de_sub_tp[subt][tp] = None; continue

        adata_sub = prepare_celltype_for_DE(adata_sub, "Macrophages", normalize=False)
        sc.tl.rank_genes_groups(adata_sub, groupby='Type', groups=['Burn'], reference='Sham',
                                method='wilcoxon', use_raw=False, pts=True, key_added='rgg')
        df = sc.get.rank_genes_groups_df(adata_sub, group='Burn', key='rgg')
        df = df.dropna(subset=['logfoldchanges', 'pvals_adj'])
        df = df.rename(columns={'names': 'gene', 'logfoldchanges': 'lfc', 'pvals_adj': 'padj'})
        df['nlp'] = -np.log10(df['padj'].clip(lower=1e-300))
        de_sub_tp[subt][tp] = df

        nb = ((df.padj < FDR_THRESH) & (df.lfc >  LFC_THRESH)).sum()
        ns = ((df.padj < FDR_THRESH) & (df.lfc < -LFC_THRESH)).sum()
        print(f"  Burn↑: {nb}   Sham↑: {ns}")
        df.to_csv(FIGDIR_MAC / f"de_subtype_{_slug(subt)}_{_slug(tp)}_burn_vs_sham.csv", index=False)


# ── 2. Volcano grid — one figure per subtype (cols = timepoints) ──────────────
for subt in SUBTYPES:
    valid = [tp for tp in ALL_TIMEPOINTS if de_sub_tp[subt].get(tp) is not None]
    if not valid:
        print(f"No valid timepoints for {subt} — skipping volcano."); continue
    ncols = min(4, len(valid)); nrows = int(np.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    for i, tp in enumerate(valid):
        draw_volcano(axes_flat[i], de_sub_tp[subt][tp], f"{subt}\n{tp}")
    for ax in axes_flat[len(valid):]:
        ax.set_visible(False)
    fig.tight_layout(); fig.canvas.draw()
    fig.savefig(FIGDIR_MAC / f"volcano_subtype_{_slug(subt)}_timepoint_burn_vs_sham.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR_MAC / f"volcano_subtype_{_slug(subt)}_timepoint_burn_vs_sham.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(f"Saved volcano: {subt}")


# ── 3. GO enrichment tile — one per (subtype, timepoint) ──────────────────────
for subt in SUBTYPES:
    for tp in ALL_TIMEPOINTS:
        df = de_sub_tp[subt].get(tp)
        if df is None:
            continue
        sig = df[df['padj'] < FDR_THRESH]
        gene_sets = {'Burn': sig[sig['lfc'] >  LFC_THRESH]['gene'].tolist(),
                     'Sham': sig[sig['lfc'] < -LFC_THRESH]['gene'].tolist()}
        rows = []
        for direction, glist in gene_sets.items():
            print(f"  [{subt} | {tp}] {direction}: {len(glist)} genes")
            if len(glist) < 5:
                continue
            try:
                enr = gp.enrichr(gene_list=glist, gene_sets=GO_LIB,
                                 organism='mouse', outdir=None, verbose=False)
                results = enr.res2d.sort_values('Adjusted P-value').copy()
                results['term_clean'] = results['Term'].apply(
                    lambda t: str(t).split('(')[0].strip().replace('_', ' '))
                results['term_clean'] = results['term_clean'].apply(
                    lambda t: t[0].upper() + t[1:] if t else t)
                results = results[~results['term_clean'].apply(contains_excluded)]
                top = results.head(N_TERMS); N = len(glist)
                for _, row in top.iterrows():
                    k, n = (int(x) for x in str(row.get('Overlap', '1/1')).split('/'))
                    fe = (k / N) / (n / M_BG) if N > 0 and n > 0 else 1.0
                    rows.append({'pathway_clean': row['term_clean'],
                                 'padj': float(row['Adjusted P-value']),
                                 'FoldEnrichment': round(fe, 1),
                                 'Count': k, 'directionality': direction})
            except Exception as e:
                print(f"    ERROR: {e}")
        make_enrichment_tile(pd.DataFrame(rows),
                             output_name=FIGDIR_MAC / f"enrichment_tile_subtype_{_slug(subt)}_{_slug(tp)}.pdf",
                             title=f"{subt} — {tp}\nBurn vs Sham GO BP", tile_size=0.7)

