"""Concise dot plot: inflammatory / M1 program across the three identities

Source: macrophages_resident_recruited.ipynb
Libraries: -
Key calls: dotplot, plt.rc_context, plt.setp, plt.show, sc.pl.dotplot
"""

# ── Concise dot plot: inflammatory / M1 program across the three identities ──
INFLAM_PANEL = ['Il1b', 'Il1a', 'Tnf', 'Il6', 'Ptgs2', 'Cxcl2', 'Cxcl3',
                'Ccl3', 'Ccl4', 'Nos2', 'Arg1', 'Saa3']
genes = [g for g in INFLAM_PANEL if g in adata_mac.var_names]
print("plotting:", genes, "| missing:", [g for g in INFLAM_PANEL if g not in genes])

order = MAC_IDENTITIES
adata_mac.obs['mac_identity'] = pd.Categorical(
    adata_mac.obs['mac_identity'].astype(str), categories=order, ordered=True)

FS_X, FS_Y = 18, 20        # gene-label / identity-label font sizes

with plt.rc_context({'font.size': 13, 'axes.linewidth': 1.2}):
    axd = sc.pl.dotplot(
        adata_mac, genes, groupby='mac_identity', categories_order=order,
        standard_scale='var', cmap='Reds', use_raw=False, show=False,
        figsize=(0.55 * len(genes) + 2.4, 2.3),
        colorbar_title='Scaled mean\nexpression', size_title='Fraction of\ncells (%)',
    )
    mainax = axd['mainplot_ax']
    plt.setp(mainax.get_xticklabels(), fontsize=FS_X, fontweight='bold')
    plt.setp(mainax.get_yticklabels(), fontsize=FS_Y, fontweight='bold')
    fig = mainax.get_figure()
    fig.savefig(f'{FIG}/dotplot_inflammatory_three_identities.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{FIG}/dotplot_inflammatory_three_identities.png', dpi=300, bbox_inches='tight')
    plt.show()

