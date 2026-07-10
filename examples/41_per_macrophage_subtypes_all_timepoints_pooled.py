"""Per macrophage_subtypes (all timepoints pooled):  DE + Volcano + GO (Burn vs Sha

Source: macrophages_resident_recruited.ipynb
Libraries: -
Key calls: adjust_text, def draw_volcano, gp.enrichr, plt.close, plt.show, plt.subplots, rank_genes_groups, sc.tl.rank_genes_groups, scatter, volcano
"""

# ══════════════════════════════════════════════════════════════════════════════
# Per macrophage_subtypes (all timepoints pooled):  DE + Volcano + GO (Burn vs Sham)
# Reuses helpers/constants from the identity cell (_slug, prepare_celltype_for_DE,
# draw_volcano, make_enrichment_tile, contains_excluded, thresholds, colors, FIGDIR_MAC,
# GO_LIB, M_BG, N_TERMS). Assumes SUBTYPES is defined; rebuilds it if not.
# ══════════════════════════════════════════════════════════════════════════════
ID_COL = "macrophage_subtypes"
#assert ID_COL in adata_mac.obs.columns, f"{ID_COL} not in adata_mac.obs"
#if "SUBTYPES" not in globals():
#    _order  = list(mac_colors.keys()) if "mac_colors" in globals() else []
#    present = [c for c in adata_mac.obs[ID_COL].astype(str).unique()
#              if c.lower() not in ("nan", "none", "")]
#    SUBTYPES = [s for s in _order if s in present] + [s for s in sorted(present) if s not in _order]
#print("subtypes:", SUBTYPES)

# ['MΦ-Inf', 'MΦ-Act', 'MΦ-IFN/AS DCs', 'Early MDM', 
# 'MΦ-Res/Rep', 'LAM-I', 'LAM-II', 'Inf. Mono.']
SUBTYPES = ['Inf. Mono.', 'MΦ-Act', 'LAM-II', 'MΦ-Res/Rep']
LFC_THRESH = 0.1
HIGHLIGHT_GENES = {"Arg1", "Nos2"}
HIGHLIGHT_COL   = 'red'

N_LABEL_DRAW = min(N_LABEL, 10)
EXCLUDE_TERMS = [
    'Negative Regulation Of Neuroinflammatory Response',
    'Negative Regulation Of Cartilage Development',
    'Regulation Of Neuroinflammatory Response',
    'Myotube Cell Development',
    "Response To Amyloid-Beta",
    'Axon Guidance',
    'Sarcomere Organixation',
'Striated Muscle Contraction',
'Muscle Contraction',
"Establishment Of Apical/Basal Cell Polarity",
'Myofibril Contracion',
'Sarcomere Organization',
'Positive Regulation Of B Cell Receptor Signaling Pathway',
'Regulation Of B Cell Receptor Signaling Pathway',
"Myofibril Assembly",
    'Neuron Projection Guidance',
    'Extracellular Matrix Organization',
    'Negative Regulation Of Chemotaxis',
    'Positive Regulation Of Actin Filament Bundle Assembly',
    'Positive Regulation of Amyloid-Beta Clearance',
    'RNA Splicing, Via Transesterification Reactions With Bulged Adenosine As Nucleophile',
    "Proteasome-Mediated Ubiquitin-Dependent Protein Catabolic Process",
    'Epithelial To Mesenchymal Transition',
    "Positive Regulation Of Protein-Containing Complex Assembly",
    'Embryonic Skeletal System Morphogenesis',
    'Positive Regulation Of Protein Polyubiquitination',
    'Embryonic Organ Morphogenesis',
    'Microvillus Assembly',
    'Microvillus Organization',
    'Skeletal System Morphogenesis',
    'Mesenchymal Cell Differentiation',
    'Negative Regulation Of Chondrocyte Differentiation',
    'Embryonic Cranial Skeleton Differentiation',
    'Embryonic Cranial Morphogenesis',
    'Thymus Development',
    'Embryonic Cranial Skeleton Morphogenesis',
    'Primary Alcohol Metabolic Process',
    'Pharyngeal System Development',
    'Positive Regulation Of Muscle Hypertrophy',
    'Collagen Fibril Organization',
]


# ── one-axis volcano (factored out of the cell above) ─────────────────────────
def draw_volcano(ax, df, title):
    sig_burn = (df['padj'] < FDR_THRESH) & (df['lfc'] >  LFC_THRESH)
    sig_sham = (df['padj'] < FDR_THRESH) & (df['lfc'] < -LFC_THRESH)
    ax.scatter(df.loc[~sig_burn & ~sig_sham, 'lfc'], df.loc[~sig_burn & ~sig_sham, 'nlp'],
               c=NS_COL, s=5, alpha=0.4, linewidths=0, rasterized=True)
    ax.scatter(df.loc[sig_sham, 'lfc'], df.loc[sig_sham, 'nlp'],
               c=SHAM_COL, s=10, alpha=0.8, linewidths=0, rasterized=True)
    ax.scatter(df.loc[sig_burn, 'lfc'], df.loc[sig_burn, 'nlp'],
               c=BURN_COL, s=10, alpha=0.8, linewidths=0, rasterized=True)
    ax.axhline(-np.log10(FDR_THRESH), color='#7F8C8D', lw=0.9, ls='--', alpha=0.6)
    ax.axvline( LFC_THRESH, color='#7F8C8D', lw=0.9, ls='--', alpha=0.6)
    ax.axvline(-LFC_THRESH, color='#7F8C8D', lw=0.9, ls='--', alpha=0.6)

    df_sig   = df[(df['padj'] < FDR_THRESH) & (df['lfc'].abs() > LFC_THRESH)]
    burn_top = pick_labels(df_sig[df_sig.lfc > 0], N_LABEL_DRAW, ascending_lfc=False)
    sham_top = pick_labels(df_sig[df_sig.lfc < 0], N_LABEL_DRAW, ascending_lfc=True)
    forced   = df[df['gene'].isin(MUST_LABEL)]
    label_df = pd.concat([burn_top, sham_top, forced]).drop_duplicates('gene')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.5, max(df['nlp'].max() * 1.15, 1.0))
    cb = ax.text(0.03, 0.98, f"Burn↑ {sig_burn.sum()}", transform=ax.transAxes,
                 fontsize=18, va='top', ha='left', color=BURN_COL, fontweight='bold')
    cs = ax.text(0.03, 0.91, f"Sham↑ {sig_sham.sum()}", transform=ax.transAxes,
                 fontsize=18, va='top', ha='left', color=SHAM_COL, fontweight='bold')
    texts = [ax.text(r['lfc'], r['nlp'], r['gene'], fontsize=FS_ANNOT,
                     color=(HIGHLIGHT_COL if r['gene'] in HIGHLIGHT_GENES else 'black'),
                     fontweight='bold', ha='center') for _, r in label_df.iterrows()]

    if texts:
        adjust_text(texts, ax=ax, objects=[cb, cs],
                    arrowprops=dict(arrowstyle='-', color='#7F8C8D', lw=0.6),
                    expand=(1.3, 1.5), force_text=(0.6, 0.8), force_static=(0.2, 0.3),
                    force_pull=(0.05, 0.05), max_move=8, min_arrow_len=3,
                    only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy'},
                    ensure_inside_axes=True, time_lim=3.0, font_size=16)
    ax.set_xlabel('Log$_2$ Fold Change', fontsize=28, fontweight='bold')
    ax.set_ylabel('$-$Log$_{10}$(padj)', fontsize=28, fontweight='bold')
    ax.set_title(title, fontsize=26, fontweight='bold'); ax.grid(False)



# ── 1. DE per subtype (Burn vs Sham, pooled over timepoints) ──────────────────
de_sub = {}
for subt in SUBTYPES:
    sel = (adata_mac.obs[ID_COL].astype(str) == subt).values
    adata_sub = adata_mac[sel].copy()
    n_b = (adata_sub.obs['Type'] == 'Burn').sum()
    n_s = (adata_sub.obs['Type'] == 'Sham').sum()
    print(f"\n[{subt}] {adata_sub.n_obs} cells  Burn={n_b}  Sham={n_s}")
    if n_b < 3 or n_s < 3:
        print("  Skipping — too few cells."); de_sub[subt] = None; continue

    adata_sub = prepare_celltype_for_DE(adata_sub, "Macrophages", normalize=False)
    sc.tl.rank_genes_groups(adata_sub, groupby='Type', groups=['Burn'], reference='Sham',
                            method='wilcoxon', use_raw=False, pts=True, key_added='rgg')
    df = sc.get.rank_genes_groups_df(adata_sub, group='Burn', key='rgg')
    df = df.dropna(subset=['logfoldchanges', 'pvals_adj'])
    df = df.rename(columns={'names': 'gene', 'logfoldchanges': 'lfc', 'pvals_adj': 'padj'})
    df['nlp'] = -np.log10(df['padj'].clip(lower=1e-300))
    de_sub[subt] = df

    nb = ((df.padj < FDR_THRESH) & (df.lfc >  LFC_THRESH)).sum()
    ns = ((df.padj < FDR_THRESH) & (df.lfc < -LFC_THRESH)).sum()
    print(f"  Burn↑: {nb}   Sham↑: {ns}")
    df.to_csv(FIGDIR_MAC / f"de_subtype_pooled_{_slug(subt)}_burn_vs_sham.csv", index=False)


# ── 2. Volcano grid — one figure, panels = subtypes ───────────────────────────
valid = [s for s in SUBTYPES if de_sub.get(s) is not None]
if valid:
    ncols = min(4, len(valid)); nrows = int(np.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    for i, subt in enumerate(valid):
        draw_volcano(axes_flat[i], de_sub[subt], subt)
    for ax in axes_flat[len(valid):]:
        ax.set_visible(False)
    fig.tight_layout(); fig.canvas.draw()
    fig.savefig(FIGDIR_MAC / "volcano_subtype_pooled_burn_vs_sham.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR_MAC / "volcano_subtype_pooled_burn_vs_sham.png", dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print("Saved pooled subtype volcano grid.")


# ── 3. GO enrichment tile — one per subtype ───────────────────────────────────
for subt in SUBTYPES:
    df = de_sub.get(subt)
    if df is None:
        continue
    sig = df[df['padj'] < FDR_THRESH]
    gene_sets = {'Burn': sig[sig['lfc'] >  LFC_THRESH]['gene'].tolist(),
                 'Sham': sig[sig['lfc'] < -LFC_THRESH]['gene'].tolist()}
    rows = []
    for direction, glist in gene_sets.items():
        print(f"  [{subt}] {direction}: {len(glist)} genes")
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
                         output_name=FIGDIR_MAC / f"enrichment_tile_subtype_pooled_{_slug(subt)}.pdf",
                         title=f"{subt}\nBurn vs Sham GO BP", tile_size=0.7)

