"""Inflammation trajectory summarized by SAMPLE (pseudobulk points + error bars)

Source: macrophages_resident_recruited.ipynb
Libraries: scipy, statsmodels
Key calls: def _overall_p, def _tt, plt.show, plt.subplots, scatter
"""

# ── Inflammation trajectory summarized by SAMPLE (pseudobulk points + error bars) ──
from scipy.stats import ttest_ind
try:
    import statsmodels.formula.api as smf
    _HAVE_SMF = True
except Exception:
    _HAVE_SMF = False

SCORE     = 'Inflammation_z'            # or 'Inflammation' for the raw module score
MIN_CELLS = 10                          # drop a sample×identity pseudobulk with fewer cells
DODGE     = {'Sham': -0.13, 'Burn': 0.13}
panels_id = [c for c in MAC_IDENTITIES if (obs['mac_identity'] == c).any()]
rng_jit   = np.random.default_rng(0)

# attach Sample (obs from Cell 1 preserves the adata index order)
_src = adata_mac if 'adata_mac' in globals() else adata_mi
obs['Sample'] = _src.obs.loc[obs.index, 'Sample'].astype(str).values

# ── per-sample pseudobulk means ───────────────────────────────────────────────
g    = obs.groupby(['Sample', 'mac_identity', 'Timepoint', 'Type'], observed=True)[SCORE]
samp = g.mean().reset_index().rename(columns={SCORE: 'score'})
samp['n'] = g.count().values
samp = samp[samp['n'] >= MIN_CELLS].copy()
print(samp.groupby(['mac_identity', 'Timepoint', 'Type'], observed=True)
          .size().rename('n_samples').reset_index().to_string(index=False))

# ── stats: per identity×timepoint Welch t-test on sample means (FDR) ──────────
gcells = [(ct, tp) for ct in panels_id for tp in TP_ORDER]
def _tt(ct, tp):
    d = samp[(samp['mac_identity'] == ct) & (samp['Timepoint'] == tp)]
    b = d.loc[d['Type'] == 'Burn', 'score']; h = d.loc[d['Type'] == 'Sham', 'score']
    return ttest_ind(b, h, equal_var=False).pvalue if len(b) > 1 and len(h) > 1 else np.nan
raw_p = {k: _tt(*k) for k in gcells}
_keys = [k for k in gcells if np.isfinite(raw_p[k])]
padj  = {k: np.nan for k in gcells}
if _keys:
    padj.update(dict(zip(_keys, multipletests([raw_p[k] for k in _keys], method='fdr_bh')[1])))

def _overall_p(ct):                     # burn-vs-sham effect controlling for timepoint (all samples)
    if not _HAVE_SMF: return np.nan
    d = samp[samp['mac_identity'] == ct]
    if d['Type'].nunique() < 2 or len(d) < 4: return np.nan
    try:
        fit = smf.ols('score ~ C(Type) + C(Timepoint)', data=d).fit()
        key = [k for k in fit.pvalues.index if k.startswith('C(Type)')]
        return fit.pvalues[key[0]] if key else np.nan
    except Exception:
        return np.nan

# ── shared y-range from the actual sample points ─────────────────────────────
ptmin, ptmax = samp['score'].min(), samp['score'].max()
rng    = ptmax - ptmin
star_y = ptmax + 0.06 * rng
ylim   = (ptmin - 0.12 * rng, ptmax + 0.30 * rng)
x      = np.arange(len(TP_ORDER)); tp_idx = {tp: i for i, tp in enumerate(TP_ORDER)}

fig, axes = plt.subplots(1, len(panels_id), figsize=(4.6 * len(panels_id), 3.9),
                         squeeze=False, sharey=True)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.82, wspace=0.08)

for ci, ct in enumerate(panels_id):
    ax = axes[0][ci]; d_ct = samp[samp['mac_identity'] == ct]
    for cond in ['Sham', 'Burn']:
        dc = d_ct[d_ct['Type'] == cond]
        xs = dc['Timepoint'].map(tp_idx).values + DODGE[cond] + rng_jit.uniform(-0.045, 0.045, len(dc))
        ax.scatter(xs, dc['score'], s=48, color=TYPE_PAL[cond], alpha=0.55,
                   edgecolors='black', linewidths=0.6, zorder=2)
        mx, my, me = [], [], []
        for tp in TP_ORDER:
            v = dc.loc[dc['Timepoint'] == tp, 'score'].values
            if len(v):
                mx.append(tp_idx[tp] + DODGE[cond]); my.append(v.mean())
                me.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.errorbar(mx, my, yerr=me, fmt='-o', color=TYPE_PAL[cond], lw=2.6, ms=11,
                    mec='black', mew=1.3, capsize=5, elinewidth=2.2, zorder=3, label=cond)
    for tp in TP_ORDER:
        s = stars(padj[(ct, tp)]) if np.isfinite(padj[(ct, tp)]) else 'ns'
        ax.text(tp_idx[tp], star_y, s, ha='center', va='bottom', fontsize=FS_STAT - 2, fontweight='bold')
    op = _overall_p(ct)
    if np.isfinite(op):
        ax.text(0.03, 0.97, f"Type p={op:.1e}", transform=ax.transAxes, ha='left', va='top',
                fontsize=13, fontweight='bold', color='#444')
    ax.axhline(0, ls=':', lw=1, c='grey', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(TP_ORDER)
    ax.set_xlim(-0.4, len(TP_ORDER) - 0.6); ax.set_ylim(ylim)
    ax.set_title(ct, fontsize=FS_TITLE - 4, fontweight='bold', pad=10)
    ax.set_xlabel('Timepoint', fontsize=FS_LABEL, fontweight='bold')
    if ci == 0: ax.set_ylabel('Inflammation (z)', fontsize=FS_LABEL, fontweight='bold')
    style_ax(ax, show_y=(ci == 0))

handles = [Line2D([0], [0], color=TYPE_PAL[c], lw=4, marker='o', ms=10, mec='black', label=c)
           for c in ['Sham', 'Burn']]
fig.legend(handles=handles, fontsize=FS_LEGEND, loc='lower center',
           bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
fig.savefig(f'{FIG}/macident_inflammation_trajectory_bysample.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG}/macident_inflammation_trajectory_bysample.png', dpi=300, bbox_inches='tight')
plt.show()

