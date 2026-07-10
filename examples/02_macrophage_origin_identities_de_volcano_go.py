"""MACROPHAGE ORIGIN IDENTITIES: DE + VOLCANO + GO — Burn vs Sham (timepoints poole

Source: macrophages_resident_recruited.ipynb
Libraries: adjustText, gseapy, matplotlib, numpy, pandas, pathlib, re, scanpy, scipy
Key calls: adjust_text, def _slug, def contains_excluded, def get_contamination_genes, def make_enrichment_tile, def pick_labels, def prepare_celltype_for_DE, gp.enrichr, plt.close, plt.cm, plt.rcParams, plt.show
"""

# ══════════════════════════════════════════════════════════════════════════════
# MACROPHAGE ORIGIN IDENTITIES: DE + VOLCANO + GO — Burn vs Sham (timepoints pooled)
# Inflammatory MDMs / Recruited Macrophages / Resident Macrophages
# Self-contained: only requires `adata_full` (log-normalized X, obs: mac_identity/Type/cell_types_simple)
# ══════════════════════════════════════════════════════════════════════════════
import re
from pathlib import Path
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import gseapy as gp
from adjustText import adjust_text

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 30,
    'axes.linewidth': 1.4, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 1.4, 'ytick.major.width': 1.4,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.labelsize': 30, 'ytick.labelsize': 30,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# ── thresholds / colors ───────────────────────────────────────────────────────
FDR_THRESH = 0.05
LFC_THRESH = 0.5
N_LABEL    = 9
BURN_COL   = '#C0392B'
SHAM_COL   = '#2471A3'
NS_COL     = '#D5D8DC'
GO_LIB     = 'GO_Biological_Process_2023'
N_TERMS    = 5
M_BG       = 15000
FS_ANNOT   = 19

# genes whose volcano labels should be drawn in red (all others stay black)
HIGHLIGHT_GENES = {"Arg1", "Nos2"}
HIGHLIGHT_COL   = 'red'

OUT = Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608")
FIGDIR_MAC = OUT / 'figures' / 'mac_identity'
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def _slug(label: str) -> str:
    """Filesystem-safe slug (handles Mφ, DC/Mono, Endo., etc.)."""
    s = label.replace('φ', 'phi')
    s = re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_').lower()
    return s or 'cell'


def get_contamination_genes(all_genes, target_celltype):
    contamination = {}
    contamination["Macrophages"] = (
        [g for g in all_genes if re.match(r'^Col\d+[a-z]{1,2}\d+$', g)] +
        [g for g in all_genes if re.match(r'^(Acta2|Myh11|Tagln|Cnn\d+|Smtn|Des|Myl\d+|Tpm\d+|Cald\d+)', g)] +
        [g for g in all_genes if re.match(r'^Krt(ap)?\d+', g)] +
        ['Sparc','Fbln2','Tnn','Vim','Fth1','Srgn','Fn1','Flg','Tchh','Dcn','Postn','Igfbp4','Tnc']
    )
    contamination["Inflammatory Monocytes"]     = contamination["Macrophages"]
    contamination["MΦ-Recruited"] = contamination["Macrophages"]
    contamination["MΦ-Resident/Repair"]  = contamination["Macrophages"]
    return set(contamination.get(target_celltype, [])) & set(all_genes)


def prepare_celltype_for_DE(adata, celltype_name, normalize=True, target_sum=1e4):
    all_genes = adata.var_names
    contam = get_contamination_genes(all_genes, celltype_name)
    clean = [g for g in all_genes if g not in contam]
    print(f"\n=== {celltype_name} ===")
    print(f"Removing {len(contam)} contaminating genes")
    a = adata[:, clean].copy()
    print(f"Genes: {adata.shape[1]} -> {a.shape[1]}")
    if normalize:
        xmax = a.X.max()
        if xmax > 50:
            sc.pp.normalize_total(a, target_sum=target_sum); sc.pp.log1p(a)
            print("Normalized and log1p-transformed.")
        else:
            print("Data appears already normalized.")
    return a


def pick_labels(df_side, n, ascending_lfc=False):
    if df_side.empty:
        return df_side.iloc[:0]
    half = max(n // 2, 1)
    by_sig = df_side.nsmallest(half, 'padj')
    by_lfc = (df_side.nsmallest(half, 'lfc') if ascending_lfc
              else df_side.nlargest(half, 'lfc'))
    return pd.concat([by_sig, by_lfc]).drop_duplicates('gene').head(n)


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
    'RNA Splicing, Via Transesterification Reactions With Bulged Adenosine As Nucleophile',
    "Proteasome-Mediated Ubiquitin-Dependent Protein Catabolic Process",
    'Epithelial To Mesenchymal Transition',
    "Positive Regulation Of Protein-Containing Complex Assembly",
    'Embryonic Skeletal System Morphogenesis',
    'Embryonic Organ Morphogenesis',
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
]
def contains_excluded(term):
    tl = term.lower()
    return any(ex.lower() in tl for ex in EXCLUDE_TERMS)


def make_enrichment_tile(df, output_name, title='', tile_size=0.7):
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

    fig_w = max_len * 0.12 + 1.0 + num_cols * tile_size + 2.5
    fig_h = 2.5 + num_rows * tile_size + 1.2
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


# ── identity setup + compartment restriction ──────────────────────────────────
MAC_IDENTITIES = ["Inflammatory Monocytes", "MΦ-Recruited", "MΦ-Resident/Repair"]
ident = adata_full.obs["mac_identity"].astype(str)

target_groups = ["Mono.", "MΦ"]
comp_col = next((c for c in adata_full.obs.columns
                 if adata_full.obs[c].astype(str).isin(target_groups).any()), None)
if comp_col is not None:
    in_compartment = adata_full.obs[comp_col].astype(str).isin(target_groups).values
    print(f"Compartment column {comp_col!r}: {int(in_compartment.sum())} cells")
else:
    in_compartment = np.ones(adata_full.n_obs, dtype=bool)
    print("No Mono/MDM·Mφ compartment column found — using mac_identity as-is.")

ALL_IDENTITIES = [ct for ct in MAC_IDENTITIES if (ident == ct).any()]
print(f'{len(ALL_IDENTITIES)} identities: {ALL_IDENTITIES}')


# ── 1. DE per identity (contamination-cleaned; X already log-normalized) ───────
de_mac_identity = {}
for ct in ALL_IDENTITIES:
    sel = (ident == ct).values & in_compartment
    adata_ct = adata_full[sel].copy()
    n_b = (adata_ct.obs['Type'] == 'Burn').sum()
    n_s = (adata_ct.obs['Type'] == 'Sham').sum()
    print(f'\n{ct} — {adata_ct.n_obs} cells  Burn={n_b}  Sham={n_s}')
    if n_b < 3 or n_s < 3:
        print('  Skipping — too few cells.'); de_mac_identity[ct] = None; continue

    adata_ct = prepare_celltype_for_DE(adata_ct, "Macrophages", normalize=False)
    sc.tl.rank_genes_groups(adata_ct, groupby='Type', groups=['Burn'], reference='Sham',
                            method='wilcoxon', use_raw=False, pts=True, key_added='rgg')
    df = sc.get.rank_genes_groups_df(adata_ct, group='Burn', key='rgg')
    df = df.dropna(subset=['logfoldchanges', 'pvals_adj'])
    df = df.rename(columns={'names': 'gene', 'logfoldchanges': 'lfc', 'pvals_adj': 'padj'})
    df['nlp'] = -np.log10(df['padj'].clip(lower=1e-300))
    de_mac_identity[ct] = df
    nb = ((df.padj < FDR_THRESH) & (df.lfc > LFC_THRESH)).sum()
    ns = ((df.padj < FDR_THRESH) & (df.lfc < -LFC_THRESH)).sum()
    print(f'  Burn↑: {nb}   Sham↑: {ns}')
    df.to_csv(FIGDIR_MAC / f'de_{_slug(ct)}_burn_vs_sham.csv', index=False)


# ── 2. Volcano grid ───────────────────────────────────────────────────────────
# genes that must always be labeled (if present in the DE result)
MUST_LABEL = ["Arg1", "Nos2", "Ccr2"]

valid_cts = [ct for ct in ALL_IDENTITIES if de_mac_identity.get(ct) is not None]
ncols = min(4, len(valid_cts)); nrows = int(np.ceil(len(valid_cts) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
axes_flat = np.atleast_1d(axes).flatten()
N_LABEL_DRAW = min(N_LABEL, 12)

for ax_idx, ct in enumerate(valid_cts):
    ax = axes_flat[ax_idx]; df = de_mac_identity[ct]
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

    df_sig = df[(df['padj'] < FDR_THRESH) & (df['lfc'].abs() > LFC_THRESH)]
    burn_top = pick_labels(df_sig[df_sig.lfc > 0], N_LABEL_DRAW, ascending_lfc=False)
    sham_top = pick_labels(df_sig[df_sig.lfc < 0], N_LABEL_DRAW, ascending_lfc=True)

    # force-include the mandatory genes (whether or not they pass thresholds)
    forced = df[df['gene'].isin(MUST_LABEL)]
    label_df = pd.concat([burn_top, sham_top, forced]).drop_duplicates('gene')

    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.5, df['nlp'].max() * 1.15)

    nb_ = sig_burn.sum(); ns_ = sig_sham.sum()
    count_burn = ax.text(0.03, 0.97, f'Burn↑ {nb_}', transform=ax.transAxes,
                         fontsize=15, va='top', ha='left', color=BURN_COL, fontweight='bold')
    count_sham = ax.text(0.03, 0.91, f'Sham↑ {ns_}', transform=ax.transAxes,
                         fontsize=15, va='top', ha='left', color=SHAM_COL, fontweight='bold')

    texts = []
    for _, row in label_df.iterrows():                      # <- was pd.concat([burn_top, sham_top])
        lbl_col = HIGHLIGHT_COL if row['gene'] in HIGHLIGHT_GENES else 'black'
        texts.append(ax.text(row['lfc'], row['nlp'], row['gene'],
                             fontsize=FS_ANNOT, color=lbl_col, fontweight='bold', ha='center'))
    if texts:
        adjust_text(
            texts, ax=ax,
            objects=[count_burn, count_sham],
            arrowprops=dict(arrowstyle='-', color='#7F8C8D', lw=0.6),
            expand=(1.3, 1.5),          # more padding around each label  -> fewer overlaps
            force_text=(0.6, 0.8),      # stronger label–label repulsion  -> fewer overlaps
            force_static=(0.2, 0.3),    # push off the data points
            force_pull=(0.05, 0.05),    # tether back to its point        -> shorter leader lines
            max_move=6,                 # cap per-iteration drift          -> shorter leader lines
            min_arrow_len=3,
            only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy'},
            ensure_inside_axes=True,
            time_lim=3.0,               # give it more time to settle into a no-overlap layout
        )


    ax.set_xlabel('Log$_2$ Fold Change', fontsize=28, fontweight='bold')
    ax.set_ylabel('$-$Log$_{10}$(padj)', fontsize=28, fontweight='bold')
    ax.set_title(f'{ct}', fontsize=30, fontweight='bold'); ax.grid(False)

for ax in axes_flat[len(valid_cts):]:
    ax.set_visible(False)
fig.tight_layout(); fig.canvas.draw()
fig.savefig(FIGDIR_MAC / 'volcano_mac_identity_burn_vs_sham.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIGDIR_MAC / 'volcano_mac_identity_burn_vs_sham.png', dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)
print('Saved: volcano_mac_identity_burn_vs_sham.{pdf,png}')



# ── 3. GO enrichment tile — one per identity ──────────────────────────────────
for ct in valid_cts:
    df = de_mac_identity[ct]
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
                lambda t: (str(t).split('(')[0].strip().replace('_', ' ')))
            results['term_clean'] = results['term_clean'].apply(
                lambda t: t[0].upper() + t[1:] if t else t)
            results = results[~results['term_clean'].apply(contains_excluded)]
            top = results.head(N_TERMS)
            N = len(glist)
            for _, row in top.iterrows():
                k, n = (int(x) for x in str(row.get('Overlap', '1/1')).split('/'))
                fe = (k / N) / (n / M_BG) if N > 0 and n > 0 else 1.0
                rows.append({'pathway_clean': row['term_clean'],
                             'padj': float(row['Adjusted P-value']),
                             'FoldEnrichment': round(fe, 1),
                             'Count': k, 'directionality': direction})
        except Exception as e:
            print(f'    ERROR: {e}')
    make_enrichment_tile(pd.DataFrame(rows),
                         output_name=FIGDIR_MAC / f'enrichment_tile_{_slug(ct)}.pdf',
                         title=f'GO Biological Process — {ct}\nBurn vs Sham',
                         tile_size=0.7)

