"""CELL-TYPE DE + VOLCANO + GO — Burn vs Sham (timepoints pooled)

Source: macrophages_resident_recruited.ipynb
Libraries: -
Key calls: adjust_text, def _resolve_lineages, def get_contamination_genes_ct, def prepare_ct_for_DE, gp.enrichr, plt.close, plt.show, plt.subplots, rank_genes_groups, sc.tl.rank_genes_groups, scatter, volcano
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL-TYPE DE + VOLCANO + GO — Burn vs Sham (timepoints pooled)
# Loops over cell_types_full: Neutrophils / Macrophages / Keratinocytes / Fibroblasts
# Per-cell-type contamination filtering (remove OTHER lineages' markers before DE)
# Reuses from the notebook: pick_labels, make_enrichment_tile, contains_excluded,
#   _slug, and the style/threshold constants (FDR_THRESH, LFC_THRESH, BURN_COL, ...)
# ══════════════════════════════════════════════════════════════════════════════
FDR_THRESH = 0.05
LFC_THRESH = 0.5
CELL_TYPES = ["Neutrophils", "Macrophages", "Keratinocytes", "Fibroblasts"]
CT_COL     = "cell_types_full"

FIGDIR_CT = OUT / 'figures' / 'cell_types_full'
FIGDIR_CT.mkdir(parents=True, exist_ok=True)

EXCLUDE_TERMS = [
    'Negative Regulation Of Neuroinflammatory Response',
    'Negative Regulation Of Cartilage Development',
    'Regulation Of Neuroinflammatory Response',
    'Myotube Cell Development',
    "Response To Amyloid-Beta",
    'Axon Guidance',
    "Negative Regulation Of Axon Extension",
    'Neuron Projection Guidance',
    'Negative Regulation Of Chemotaxis',
    'Positive Regulation Of Actin Filament Bundle Assembly',
    'Positive Regulation of Amyloid-Beta Clearance',
    'Regulation of Angiogenesis',
    "Positive Regulation Of Vascualture Development",
    'RNA Splicing, Via Transesterification Reactions With Bulged Adenosine As Nucleophile',
    "Proteasome-Mediated Ubiquitin-Dependent Protein Catabolic Process",
    'Epithelial To Mesenchymal Transition',
    "Positive Regulation Of Protein-Containing Complex Assembly",
    'Embryonic Skeletal System Morphogenesis',
    'Embryonic Organ Morphogenesis',
    "Positive Regulation Of Vascualture Development",
    'Skeletal System Morphogenesis',
    'Mesenchymal Cell Differentiation',
    'Negative Regulation Of Chondrocyte Differentiation',
    'Embryonic Cranial Skeleton Differentiation',
    'Embryonic Cranial Morphogenesis',
    'Thymus Development',
    'Embryonic Cranial Skeleton Morphogenesis',
    'Positive Regulation Of Angiogenesis',
    'Primary Alcohol Metabolic Process',
    'Pharyngeal System Development',
    'Positive Regulation Of Muscle Hypertrophy',
]
# ── lineage marker panels (regex + literal) ───────────────────────────────────
# For a target cell type, contamination = union of markers from every OTHER lineage.
LINEAGE_MARKERS = {
    "Fibroblasts": {
        "regex": [r'^Col\d+[a-z]{1,2}\d+$',                                   # collagens
                  r'^(Acta2|Myh11|Tagln|Cnn\d+|Smtn|Des|Myl\d+|Tpm\d+|Cald\d+)'],  # (myo)fibro/muscle
        "genes": ['Dcn','Lum','Postn','Fn1','Sparc','Fbln1','Fbln2','Tnn','Tnc',
                  'Igfbp4','Igfbp6','Mgp','Gsn','Pdgfra','Pdgfrb','Sfrp2','Sfrp4',
                  'Mfap5','Mfap4','Ogn','Aspn','Loxl1','Thbs2','Pcolce','Serpinf1',
                  'Mmp2','Cygb','Twist2','Prrx1','Vim'],
    },
    "Keratinocytes": {
        "regex": [r'^Krt(ap)?\d+', r'^Sprr\d+', r'^Dsg\d', r'^Dsc\d'],
        "genes": ['Flg','Flg2','Tchh','Krtdap','Sbsn','Lor','Dsp','Dmkn','Lgals7',
                  'Perp','Cdh1','Cnfn','Calml3','Ivl','Grhl3','Epcam','Cldn1','Cstb'],
    },
    "Macrophages": {
        "regex": [r'^C1q[abc]$', r'^Ms4a6'],
        "genes": ['Lyz2','Lyz1','Cd68','Adgre1','Fcgr1','Csf1r','Mrc1','Apoe',
                  'Ms4a7','Ctss','Fcer1g','Aif1','Cd14','Pf4','Folr2', "Il1b", "Axl", "Gas6"],
    },
    "Neutrophils": {
        "regex": [r'^S100a[89]$'],
        "genes": ['Retnlg','Mmp9','Ngp','Camp','Ltf','Lcn2','G0s2','Ly6g','Wfdc21',
                  'Stfa2l1','Il1b','Cxcr2','Csf3r','Mpo','Elane','Prtn3'],
    },
}
GENERIC_AMBIENT = ['Fth1', 'Srgn']   # stripped from ALL cell types (ambient/stress; edit to taste)

def _resolve_lineages(all_genes):
    all_set = set(all_genes)
    resolved = {}
    for lin, spec in LINEAGE_MARKERS.items():
        hits = set(spec.get("genes", [])) & all_set
        for pat in spec.get("regex", []):
            hits |= {g for g in all_genes if re.match(pat, g)}
        resolved[lin] = hits
    return resolved

def get_contamination_genes_ct(all_genes, target_celltype):
    """Markers of every lineage EXCEPT the target, plus generic ambient — minus the
    target's own markers (so shared genes are never stripped from their home compartment)."""
    resolved = _resolve_lineages(all_genes)
    own = resolved.get(target_celltype, set())
    contam = set(GENERIC_AMBIENT) & set(all_genes)
    for lin, genes in resolved.items():
        if lin != target_celltype:
            contam |= genes
    return (contam - own) & set(all_genes)

def prepare_ct_for_DE(adata, target_celltype):
    contam = get_contamination_genes_ct(adata.var_names, target_celltype)
    clean  = [g for g in adata.var_names if g not in contam]
    print(f"  Removing {len(contam)} contaminating genes  ({adata.shape[1]} -> {len(clean)})")
    return adata[:, clean].copy()

# ── sanity: which requested types are actually present ────────────────────────
assert CT_COL in adata_full.obs, f"{CT_COL!r} not in adata_full.obs"
ct_series = adata_full.obs[CT_COL].astype(str)
present = [ct for ct in CELL_TYPES if (ct_series == ct).any()]
missing = [ct for ct in CELL_TYPES if ct not in present]
if missing:
    print(f"WARNING: not found in {CT_COL}: {missing}")
    print(f"  available values: {sorted(ct_series.unique())}")
print(f"{len(present)} cell types: {present}")

# ── 1. DE per cell type (Burn vs Sham; X already log-normalized) ──────────────
de_celltype = {}
for ct in present:
    sel = (ct_series == ct).values
    a = adata_full[sel].copy()
    n_b = (a.obs['Type'] == 'Burn').sum(); n_s = (a.obs['Type'] == 'Sham').sum()
    print(f"\n=== {ct} — {a.n_obs} cells  Burn={n_b}  Sham={n_s} ===")
    if n_b < 3 or n_s < 3:
        print('  Skipping — too few cells.'); de_celltype[ct] = None; continue

    a = prepare_ct_for_DE(a, ct)
    sc.tl.rank_genes_groups(a, groupby='Type', groups=['Burn'], reference='Sham',
                            method='wilcoxon', use_raw=False, pts=True, key_added='rgg')
    df = sc.get.rank_genes_groups_df(a, group='Burn', key='rgg')
    df = df.dropna(subset=['logfoldchanges', 'pvals_adj']).rename(
        columns={'names': 'gene', 'logfoldchanges': 'lfc', 'pvals_adj': 'padj'})
    df['nlp'] = -np.log10(df['padj'].clip(lower=1e-300))
    de_celltype[ct] = df
    nb = ((df.padj < FDR_THRESH) & (df.lfc >  LFC_THRESH)).sum()
    ns = ((df.padj < FDR_THRESH) & (df.lfc < -LFC_THRESH)).sum()
    print(f"  Burn↑: {nb}   Sham↑: {ns}")
    df.to_csv(FIGDIR_CT / f'de_{_slug(ct)}_burn_vs_sham.csv', index=False)

# ── 2. Volcano grid ───────────────────────────────────────────────────────────
MUST_LABEL     = ["Il1b","S100a8", "S100a9", "Mpo", "Tnfb", "Il1b", "Gas6", "Axl"]     # labeled where present (harmless otherwise)
HIGHLIGHT_GENES = {}
HIGHLIGHT_COL   = 'red'

valid_cts = [ct for ct in present if de_celltype.get(ct) is not None]
ncols = min(4, len(valid_cts)); nrows = int(np.ceil(len(valid_cts) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
axes_flat = np.atleast_1d(axes).flatten()
N_LABEL_DRAW = min(N_LABEL, 20)

for ax_idx, ct in enumerate(valid_cts):
    ax = axes_flat[ax_idx]; df = de_celltype[ct]
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

    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.5, df['nlp'].max() * 1.15)
    count_burn = ax.text(0.03, 0.97, f'Burn↑ {int(sig_burn.sum())}', transform=ax.transAxes,
                         fontsize=15, va='top', ha='left', color=BURN_COL, fontweight='bold')
    count_sham = ax.text(0.03, 0.91, f'Sham↑ {int(sig_sham.sum())}', transform=ax.transAxes,
                         fontsize=15, va='top', ha='left', color=SHAM_COL, fontweight='bold')

    texts = []
    for _, row in label_df.iterrows():
        lbl_col = HIGHLIGHT_COL if row['gene'] in HIGHLIGHT_GENES else 'black'
        texts.append(ax.text(row['lfc'], row['nlp'], row['gene'],
                             fontsize=FS_ANNOT, color=lbl_col, fontweight='bold', ha='center'))
    if texts:
        adjust_text(
            texts, ax=ax, objects=[count_burn, count_sham],
            arrowprops=dict(arrowstyle='-', color='#7F8C8D', lw=0.6),
            expand=(1.3, 1.5), force_text=(0.6, 0.8), force_static=(0.2, 0.3),
            force_pull=(0.05, 0.05), max_move=6, min_arrow_len=3,
            only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy'},
            ensure_inside_axes=True, time_lim=3.0,
        )

    ax.set_xlabel('Log$_2$ Fold Change', fontsize=28, fontweight='bold')
    ax.set_ylabel('$-$Log$_{10}$(padj)', fontsize=28, fontweight='bold')
    ax.set_title(f'{ct}', fontsize=30, fontweight='bold'); ax.grid(False)

for ax in axes_flat[len(valid_cts):]:
    ax.set_visible(False)
fig.tight_layout(); fig.canvas.draw()
fig.savefig(FIGDIR_CT / 'volcano_cell_types_full_burn_vs_sham.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIGDIR_CT / 'volcano_cell_types_full_burn_vs_sham.png', dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)
print('Saved: volcano_cell_types_full_burn_vs_sham.{pdf,png}')

# ── 3. GO enrichment tile — one per cell type ─────────────────────────────────
for ct in valid_cts:
    df  = de_celltype[ct]
    sig = df[df['padj'] < FDR_THRESH]
    gene_sets = {'Burn': sig[sig['lfc'] >  LFC_THRESH]['gene'].tolist(),
                 'Sham': sig[sig['lfc'] < -LFC_THRESH]['gene'].tolist()}
    rows = []
    for direction, glist in gene_sets.items():
        print(f'  {ct} | {direction}: {len(glist)} genes')
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
            N = len(glist)
            for _, row in results.head(N_TERMS).iterrows():
                k, n = (int(x) for x in str(row.get('Overlap', '1/1')).split('/'))
                fe = (k / N) / (n / M_BG) if N > 0 and n > 0 else 1.0
                rows.append({'pathway_clean': row['term_clean'],
                             'padj': float(row['Adjusted P-value']),
                             'FoldEnrichment': round(fe, 1),
                             'Count': k, 'directionality': direction})
        except Exception as e:
            print(f'    ERROR: {e}')
    make_enrichment_tile(pd.DataFrame(rows),
                         output_name=FIGDIR_CT / f'enrichment_tile_{_slug(ct)}.pdf',
                         title=f'GO Biological Process — {ct}\nBurn vs Sham',
                         tile_size=0.7)

