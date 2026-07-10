"""Figure A: one panel per mac_identity, Inflammation vs Resolution

Source: macrophages_resident_recruited.ipynb
Libraries: -
Key calls: plt.show, plt.subplots, scatter, sns.kdeplot
"""

# ── Figure A: one panel per mac_identity, Inflammation vs Resolution ──────────
panels = [c for c in MAC_IDENTITIES if (obs['mac_identity'] == c).any()]
pvals  = {c: ks_imbalance(obs[obs['mac_identity'] == c])[1] for c in panels}
padj   = dict(zip(panels, multipletests([pvals[c] if np.isfinite(pvals[c]) else 1.0
                                         for c in panels], method='fdr_bh')[1]))

ncol = len(panels)
fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 6.0), squeeze=False,
                         sharex=True, sharey=True)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.90, wspace=0.07)

for ci, ct in enumerate(panels):
    ax  = axes[0][ci]
    sub = obs[obs['mac_identity'] == ct]
    for cond in ['Sham', 'Burn']:
        c = sub[sub['Type'] == cond]
        ax.scatter(c[XK], c[YK], s=10, alpha=0.35, color=TYPE_PAL[cond],
                   edgecolors='none', rasterized=True, zorder=1)
        if len(c) > 5:
            sns.kdeplot(data=c, x=XK, y=YK, ax=ax, color=TYPE_PAL[cond],
                        levels=5, thresh=0.10, fill=True, alpha=0.20, zorder=2)
            sns.kdeplot(data=c, x=XK, y=YK, ax=ax, color=TYPE_PAL[cond],
                        levels=5, thresh=0.10, linewidths=1.6, alpha=0.9, zorder=3)
        ctr = c[[XK, YK]].mean().values
        ax.scatter(*ctr, s=220, color=TYPE_PAL[cond], edgecolor='black', lw=2, zorder=6)

    ax.axhline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
    ax.axvline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

    d, _ = ks_imbalance(sub)
    ax.text(0.04, 0.97, f"KS D={d:.2f}\n{stars(padj[ct])}", transform=ax.transAxes,
            ha='left', va='top', fontsize=FS_STAT, fontweight='bold')
    ax.set_title(ct, fontsize=FS_TITLE, fontweight='bold', pad=8)
    ax.set_xlabel(XLAB, fontsize=FS_LABEL, fontweight='bold')
    if ci == 0:
        ax.set_ylabel(YLAB, fontsize=FS_LABEL, fontweight='bold', labelpad=8)
    style_ax(ax, show_y=(ci == 0))

# annotate the "unresolved" quadrant on the last panel
axes[0][-1].text(0.97, 0.03, 'unresolved\n(inflamed, low repair)', transform=axes[0][-1].transAxes,
                 ha='right', va='bottom', fontsize=14, style='italic', color='#7F8C8D')

handles = [Line2D([0], [0], color=TYPE_PAL[c], lw=6, label=c) for c in ['Sham', 'Burn']]
fig.legend(handles=handles, fontsize=FS_LEGEND, loc='lower center',
           bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
fig.savefig(f'{FIG}/macident_inflammation_vs_resolution_byidentity.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG}/macident_inflammation_vs_resolution_byidentity.png', dpi=300, bbox_inches='tight')
plt.show()

