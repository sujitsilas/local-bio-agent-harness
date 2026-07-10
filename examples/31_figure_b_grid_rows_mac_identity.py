"""Figure B-grid: rows = mac_identity, cols = timepoint

Source: macrophages_resident_recruited.ipynb
Libraries: -
Key calls: plt.show, plt.subplots, scatter, sns.kdeplot
"""

# ── Figure B-grid: rows = mac_identity, cols = timepoint ─────────────────────
panels_id = [c for c in MAC_IDENTITIES if (obs['mac_identity'] == c).any()]

# KS Burn vs Sham (Imbalance axis) per identity×timepoint, FDR across all cells
grid_cells = [(ct, tp) for ct in panels_id for tp in TP_ORDER]
raw_p = {k: ks_imbalance(obs[(obs['mac_identity'] == k[0]) & (obs['Timepoint'] == k[1])])[1]
         for k in grid_cells}
_keys = [k for k in grid_cells if np.isfinite(raw_p[k])]
padj_grid = {k: 1.0 for k in grid_cells}
if _keys:
    padj_grid.update(dict(zip(_keys, multipletests([raw_p[k] for k in _keys], method='fdr_bh')[1])))

nrow, ncol = len(panels_id), len(TP_ORDER)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.8 * nrow), squeeze=False,
                         sharex=True, sharey=True)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.09, top=0.93, wspace=0.07, hspace=0.10)

for ri, ct in enumerate(panels_id):
    for ci, tp in enumerate(TP_ORDER):
        ax  = axes[ri][ci]
        sub = obs[(obs['mac_identity'] == ct) & (obs['Timepoint'] == tp)]
        for cond in ['Sham', 'Burn']:
            c = sub[sub['Type'] == cond]
            ax.scatter(c[XK], c[YK], s=10, alpha=0.35, color=TYPE_PAL[cond],
                       edgecolors='none', rasterized=True, zorder=1)
            if len(c) > 5:
                sns.kdeplot(data=c, x=XK, y=YK, ax=ax, color=TYPE_PAL[cond],
                            levels=5, thresh=0.10, fill=True, alpha=0.20, zorder=2)
                sns.kdeplot(data=c, x=XK, y=YK, ax=ax, color=TYPE_PAL[cond],
                            levels=5, thresh=0.10, linewidths=1.6, alpha=0.9, zorder=3)
            if len(c):
                ctr = c[[XK, YK]].mean().values
                ax.scatter(*ctr, s=170, color=TYPE_PAL[cond], edgecolor='black', lw=1.8, zorder=6)

        ax.axhline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
        ax.axvline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

        d, _ = ks_imbalance(sub)
        nb = int((sub['Type'] == 'Burn').sum()); ns = int((sub['Type'] == 'Sham').sum())
        dtxt = f"D={d:.2f}" if np.isfinite(d) else "D=n/a"
        ax.text(0.04, 0.97, f"{dtxt}\n{stars(padj_grid[(ct, tp)])}", transform=ax.transAxes,
                ha='left', va='top', fontsize=FS_STAT - 2, fontweight='bold')
        ax.text(0.97, 0.97, f"B{nb}/S{ns}", transform=ax.transAxes, ha='right', va='top',
                fontsize=12, color='#7F8C8D')

        if ri == 0:
            ax.set_title(tp, fontsize=FS_TITLE, fontweight='bold', pad=8)
        if ri == nrow - 1:
            ax.set_xlabel(XLAB, fontsize=FS_LABEL, fontweight='bold')
        if ci == 0:
            ax.set_ylabel(YLAB, fontsize=FS_LABEL - 4, fontweight='bold', labelpad=8)
            ax.annotate(ct, xy=(-0.45, 0.5), xycoords='axes fraction', rotation=90,
                        ha='center', va='center', fontsize=FS_TITLE - 2, fontweight='bold')
        style_ax(ax, show_y=(ci == 0))

handles = [Line2D([0], [0], color=TYPE_PAL[c], lw=6, label=c) for c in ['Sham', 'Burn']]
fig.legend(handles=handles, fontsize=FS_LEGEND, loc='lower center',
           bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
fig.savefig(f'{FIG}/macident_inflam_vs_resolution_identity_x_timepoint.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG}/macident_inflam_vs_resolution_identity_x_timepoint.png', dpi=300, bbox_inches='tight')
plt.show()

