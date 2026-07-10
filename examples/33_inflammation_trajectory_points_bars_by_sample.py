"""Inflammation trajectory — points/bars by SAMPLE, stats by mixed model

Source: macrophages_resident_recruited.ipynb
Libraries: scipy, statsmodels, warnings
Key calls: def _mixed, def _star, plt.show, plt.subplots, scatter
"""

# ── Inflammation trajectory — points/bars by SAMPLE, stats by mixed model ─────
import warnings
import statsmodels.formula.api as smf
from scipy.stats import norm, chi2

SCORE     = 'Inflammation_z'            # or 'Inflammation'
MIN_CELLS = 10
DODGE     = {'Sham': -0.13, 'Burn': 0.13}
panels_id = [c for c in MAC_IDENTITIES if (obs['mac_identity'] == c).any()]
rng_jit   = np.random.default_rng(0)
BURN_TERM = "C(Type, Treatment('Sham'))[T.Burn]"

# attach Sample (obs from Cell 1 preserves adata index order)
_src = adata_mac if 'adata_mac' in globals() else adata_mi
obs['Sample'] = _src.obs.loc[obs.index, 'Sample'].astype(str).values

# ── sample-level pseudobulk means (for the plotted points + SEM bars) ─────────
g    = obs.groupby(['Sample', 'mac_identity', 'Timepoint', 'Type'], observed=True)[SCORE]
samp = g.mean().reset_index().rename(columns={SCORE: 'score'})
samp['n'] = g.count().values
samp = samp[samp['n'] >= MIN_CELLS].copy()
print(samp.groupby(['mac_identity', 'Timepoint', 'Type'], observed=True)
          .size().rename('n_samples').reset_index().to_string(index=False))

# ── linear mixed model per identity: y ~ Type*Timepoint + (1|Sample) ──────────
def _mixed(ct):
    d = obs[(obs['mac_identity'] == ct) & obs['Type'].isin(['Sham', 'Burn'])].dropna(subset=[SCORE]).copy()
    if d['Type'].nunique() < 2 or d['Sample'].nunique() < 3:
        return {tp: (np.nan, np.nan) for tp in TP_ORDER}, np.nan
    ref = TP_ORDER[0]
    f = f"Q('{SCORE}') ~ C(Type, Treatment('Sham')) * C(Timepoint, Treatment('{ref}'))"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = smf.mixedlm(f, d, groups=d['Sample']).fit(method='lbfgs')
    except Exception as e:
        print(f"  LMM failed for {ct}: {e}")
        return {tp: (np.nan, np.nan) for tp in TP_ORDER}, np.nan

    fe = res.fe_params; names = fe.index.tolist()
    cov = res.cov_params().loc[names, names].values
    per_tp = {}
    for tp in TP_ORDER:                                   # Burn-Sham contrast at each timepoint
        c = np.zeros(len(names)); hit = False
        for i, nm in enumerate(names):
            if nm == BURN_TERM:
                c[i] = 1.0; hit = True
            elif nm.startswith(BURN_TERM + ':') and nm.endswith(f"[T.{tp}]"):
                c[i] = 1.0
        if not hit:
            per_tp[tp] = (np.nan, np.nan); continue
        eff = float(c @ fe.values); var = float(c @ cov @ c)
        per_tp[tp] = (eff, float(2 * norm.sf(abs(eff / np.sqrt(var)))) if var > 0 else np.nan)

    bidx = [i for i, nm in enumerate(names) if nm == BURN_TERM or nm.startswith(BURN_TERM + ':')]
    b = fe.values[bidx]; V = cov[np.ix_(bidx, bidx)]      # joint Wald: any Type effect
    try:
        overall = float(chi2.sf(float(b @ np.linalg.solve(V, b)), len(bidx)))
    except Exception:
        overall = np.nan
    return per_tp, overall

mixed   = {ct: _mixed(ct) for ct in panels_id}
overall = {ct: mixed[ct][1] for ct in panels_id}
gcells  = [(ct, tp) for ct in panels_id for tp in TP_ORDER]
raw_p   = {k: mixed[k[0]][0][k[1]][1] for k in gcells}
_keys   = [k for k in gcells if np.isfinite(raw_p[k])]
padj    = {k: np.nan for k in gcells}
if _keys:
    padj.update(dict(zip(_keys, multipletests([raw_p[k] for k in _keys], method='fdr_bh')[1])))
print("LMM Burn-Sham padj:")
for ct in panels_id:
    print(" ", ct, {tp: (f"{padj[(ct,tp)]:.1e}" if np.isfinite(padj[(ct,tp)]) else 'na') for tp in TP_ORDER})

# ── plot ──────────────────────────────────────────────────────────────────────
def _star(p):
    return 'na' if not np.isfinite(p) else stars(p)

ptmin, ptmax = samp['score'].min(), samp['score'].max()
r      = ptmax - ptmin
star_y = ptmax + 0.06 * r
ylim   = (ptmin - 0.12 * r, ptmax + 0.30 * r)
x      = np.arange(len(TP_ORDER)); tp_idx = {tp: i for i, tp in enumerate(TP_ORDER)}

fig, axes = plt.subplots(1, len(panels_id), figsize=(4.6 * len(panels_id), 5),
                         squeeze=False, sharey=True)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.82, wspace=0.08)

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
        ax.text(tp_idx[tp], star_y, _star(padj[(ct, tp)]), ha='center', va='bottom',
                fontsize=FS_STAT - 2, fontweight='bold')
    #if np.isfinite(overall[ct]):
        #ax.text(0.03, 0.97, f"Type p={overall[ct]:.1e}", transform=ax.transAxes,
         #       ha='left', va='top', fontsize=13, fontweight='bold', color='#444')
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
fig.savefig(f'{FIG}/macident_inflammation_trajectory_lmm.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG}/macident_inflammation_trajectory_lmm.png', dpi=300, bbox_inches='tight')
plt.show()

