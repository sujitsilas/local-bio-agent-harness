"""Per-UpSet-bin enrichment + highlight genes (Burn↑ / Sham↑)

Source: macrophages_resident_recruited.ipynb
Libraries: -
Key calls: def bin_label, def bins_enrichment, def deg_long, def make_enrichment_tile, gp.enrichr, plt.close, plt.cm, plt.show, plt.subplots
"""

# ══════════════════════════════════════════════════════════════════════════════
# Per-UpSet-bin enrichment + highlight genes (Burn↑ / Sham↑)
# Reuses de_id_tp, DISP, gp/GO_LIB/N_TERMS/M_BG/contains_excluded/make_enrichment_tile
# ══════════════════════════════════════════════════════════════════════════════
SHORT = {"Inflammatory Monocytes": "Inf. Mono.", "MΦ-Recruited": "MΦ-Recr", "MΦ-Resident/Repair": "MΦ-Res/Rep"}



def make_enrichment_tile(df, output_name, title='', tile_size=1, w_scale=5, h_scale=1.2):
    if df.empty:
        print(f'No data — skipping {output_name}'); return
    df = df.copy()
    df['logp'] = -np.log10(df['padj'].clip(lower=1e-300))
    preferred_order = ['Sham', 'Burn']
    available_dirs = df['directionality'].unique().tolist()
    dir_order = [d for d in preferred_order if d in available_dirs] + \
                [d for d in available_dirs if d not in preferred_order]
    df['directionality'] = pd.Categorical(df['directionality'], categories=dir_order, ordered=True)
    df = df.sort_values(['directionality', 'logp'], ascending=[True, False])

    pathways = df['pathway_clean'].unique().tolist()
    pathways_reversed = pathways[::-1]
    pathway_to_y = {p: i for i, p in enumerate(pathways_reversed)}
    num_rows = len(pathways); num_cols = len(dir_order)

    limit = 40
    trunc_pathways = [p[:limit] + '...' if len(p) > limit else p for p in pathways]
    trunc_pathways_reversed = trunc_pathways[::-1]
    max_len = max(len(p) for p in trunc_pathways)

    fig_w = (max_len * 0.12 + 1.0 + num_cols * tile_size + 2.5) * w_scale
    fig_h = (2.5 + num_rows * tile_size + 1.2) * h_scale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = mcolors.LinearSegmentedColormap.from_list('br', ['#377EB8', '#E41A1C'])
    norm_lp = mcolors.Normalize(vmin=df['logp'].min(), vmax=df['logp'].max())
    c_min, c_max = df['Count'].min(), df['Count'].max()
    get_alpha = lambda c: 0.4 + 0.6 * (c - c_min) / (c_max - c_min) if c_max > c_min else 1.0

    data_lookup = {(r['directionality'], r['pathway_clean']): r for _, r in df.iterrows()}
    for col_idx, grp in enumerate(dir_order):
        for pathway, row_idx in pathway_to_y.items():
            key = (grp, pathway)
            if key not in data_lookup: continue
            row = data_lookup[key]
            color = cmap(norm_lp(row['logp']))
            ax.add_patch(patches.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1,
                                           color=color, alpha=get_alpha(row['Count']), ec=None))
            ax.text(col_idx, row_idx, f"{row['FoldEnrichment']}",
                    ha='center', va='center', color='black', fontsize=35, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(dir_order, rotation=45, fontsize=53, color='black', fontweight='bold', ha='right')
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(trunc_pathways_reversed, fontsize=53, color='black')
    ax.set_xlim(-0.5, num_cols - 0.5); ax.set_ylim(-0.5, num_rows - 0.5)
    for spine in ax.spines.values(): spine.set_edgecolor('black'); spine.set_linewidth(1.2)
    ax.set_facecolor('white'); ax.grid(False)
    if title: ax.set_title(title, fontsize=19, fontweight='bold', pad=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_lp); sm.set_array([])
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0 + pos.height * 0.25, 0.018, pos.height * 0.5])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r'$-\log_{10}(p_{adj})$', fontsize=48, fontweight='bold', labelpad=12)
    cbar.ax.tick_params(labelsize=22); cbar.ax.yaxis.label.set_fontweight('bold')

    fig.canvas.draw()
    fig.savefig(output_name, dpi=600, bbox_inches='tight')
    fig.savefig(str(output_name).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(f'Saved: {output_name}')


def bin_label(sig, n_total):
    shorts = [SHORT.get(s, s) for s in sig]
    if len(sig) == 1:        return f"{shorts[0]} only"
    if len(sig) == n_total:  return "Shared (all)"
    return " & ".join(shorts)

def deg_long(direction):
    """All significant DEG rows (gene, identity, tp, lfc, padj, nlp) for a direction."""
    frames = []
    for ident in IDENTITIES:
        if re.search(r'ambig|low', ident, re.I):
            continue
        for tp in ALL_TIMEPOINTS:
            df = de_id_tp[ident].get(tp)
            if df is None:
                continue
            sub = (df[(df.padj < FDR_THRESH) & (df.lfc >  LFC_THRESH)] if direction == "Burn"
                   else df[(df.padj < FDR_THRESH) & (df.lfc < -LFC_THRESH)])
            if sub.empty:
                continue
            t = sub[['gene', 'lfc', 'padj', 'nlp']].copy()
            t['identity'] = DISP.get(ident, ident); t['tp'] = tp
            frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['gene', 'lfc', 'padj', 'nlp', 'identity', 'tp'])

def bins_enrichment(direction, n_highlight=4, min_genes=5):
    long = deg_long(direction)
    if long.empty:
        print(f"No DEGs for {direction}."); return
    n_total = long['identity'].nunique()

    # per-gene: strongest signal + which identities it's a DEG in (the bin signature)
    agg = (long.groupby('gene')
               .agg(nlp_max=('nlp', 'max'),
                    lfc_mean=('lfc', 'mean'),
                    idents=('identity', lambda s: tuple(sorted(set(s)))))
               .reset_index())
    agg['bin'] = agg['idents'].apply(lambda sig: bin_label(sig, n_total))
    agg.sort_values(['bin', 'nlp_max'], ascending=[True, False]).to_csv(
        FIGDIR_MAC / f"upset_bins_{direction.lower()}_genes.csv", index=False)

    rows, highlights = [], []
    for b, sub in agg.groupby('bin'):
        genes = sub['gene'].tolist()
        top   = sub.sort_values('nlp_max', ascending=False)['gene'].head(n_highlight).tolist()
        highlights += [(b, g) for g in top]
        print(f"\n[{direction} | {b}]  n={len(genes)}  highlight: {', '.join(top)}")
        if len(genes) < min_genes:
            print("  <5 genes — skipping enrichment."); continue
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=GO_LIB, organism='mouse',
                             outdir=None, verbose=False)
            res = enr.res2d.sort_values('Adjusted P-value').copy()
            res['term_clean'] = res['Term'].apply(lambda t: str(t).split('(')[0].strip().replace('_', ' '))
            res['term_clean'] = res['term_clean'].apply(lambda t: t[0].upper() + t[1:] if t else t)
            res = res[~res['term_clean'].apply(contains_excluded)]
            N = len(genes)
            for _, r in res.head(N_TERMS).iterrows():
                k, n = (int(x) for x in str(r.get('Overlap', '1/1')).split('/'))
                fe = (k / N) / (n / M_BG) if N > 0 and n > 0 else 1.0
                rows.append({'pathway_clean': r['term_clean'], 'padj': float(r['Adjusted P-value']),
                             'FoldEnrichment': round(fe, 1), 'Count': k, 'directionality': b})
        except Exception as e:
            print(f"  ERROR: {e}")

    pd.DataFrame(highlights, columns=['bin', 'highlight_gene']).to_csv(
        FIGDIR_MAC / f"upset_bins_{direction.lower()}_highlight_genes.csv", index=False)

    # tile: rows = GO terms, columns = bins
    make_enrichment_tile(pd.DataFrame(rows),
                         output_name=FIGDIR_MAC / f"upset_bins_enrichment_{direction.lower()}.pdf",
                         title=f"{direction}↑ DEG intersections — GO BP", tile_size=0.7)

for d in ["Burn", "Sham"]:
    bins_enrichment(d)

