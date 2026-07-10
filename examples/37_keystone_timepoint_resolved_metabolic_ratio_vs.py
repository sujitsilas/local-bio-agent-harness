"""KEYSTONE (timepoint-resolved): Metabolic_Ratio vs Recruited→Resident axis

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, re, scipy
Key calls: plt.show, plt.subplots, sc.tl.score_genes, scatter, sns.kdeplot
"""

# ══════════════════════════════════════════════════════════════════════════════
# KEYSTONE (timepoint-resolved): Metabolic_Ratio vs Recruited→Resident axis
#   rows = compartment, cols = timepoint, Burn vs Sham.
#   Shared axes; figure-level directional labels; Sham→Burn shift arrow.
# Safe to run standalone (recomputes scores if missing).
# ══════════════════════════════════════════════════════════════════════════════
import re, numpy as np, pandas as pd, scanpy as sc
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp

TYPE_PAL = {'Burn': '#C0392B', 'Sham': '#2980B9'}
FIG = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/figures"

# ── ensure the two axes exist (reuse if the keystone cell already ran) ────────
if 'Metabolic_Ratio' not in adata_mac.obs:
    adata_mac.obs['Metabolic_Ratio'] = adata_mac.obs['Glycolysis_Score'] - adata_mac.obs['OXPHOS_Score']
if 'Recr_to_Res' not in adata_mac.obs:
    try:
        _recr_genes, _res_genes = recruited_macs_genes, resident_macs_genes
    except NameError:
        _recr_genes = ["Ccr2","Ly6c2","Arg1","Fn1","Spp1","Trem2","Gpnmb","Cd9","Ms4a7","Itgax"]
        _res_genes  = ["Adgre1","Mertk","Timd4","Cd163","Mrc1","Folr2","Lyve1","Gas6",
                       "Selenop","C1qa","C1qb","C1qc","Pf4","Maf"]
    for name, gl in [('Recruited_Score', _recr_genes), ('Resident_Score', _res_genes)]:
        sc.tl.score_genes(adata_mac, [x for x in gl if x in adata_mac.var_names],
                          score_name=name, use_raw=False, random_state=0)
    d = adata_mac.obs['Resident_Score'] - adata_mac.obs['Recruited_Score']
    adata_mac.obs['Recr_to_Res'] = (d - d.mean()) / d.std()

obs    = adata_mac.obs.copy()
XK, YK = 'Recr_to_Res', 'Metabolic_Ratio'

# ── compartments (rows) and timepoints (cols) ────────────────────────────────
have = set(obs['mac_identity'].astype(str))
find = lambda k: next((h for h in sorted(have) if k in h.lower()), None)
COMPARTMENTS = [x for x in [find('inflamm'), find('recruit'), find('resident')] if x]
try:
    TPS = [t for t in ALL_TIMEPOINTS if t in set(obs['Timepoint'].astype(str))]
except NameError:
    TPS = sorted(obs['Timepoint'].astype(str).unique(), key=lambda t: int(re.search(r'\d+', t).group()))
print("rows:", COMPARTMENTS, "| cols:", TPS)

# shared limits across ALL panels → directly comparable
xlo, xhi = np.nanpercentile(obs[XK], [1, 99])
ylo, yhi = np.nanpercentile(obs[YK], [1, 99])
star = lambda p: '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'

nrow, ncol = len(COMPARTMENTS), len(TPS)
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4.2* nrow),
                         squeeze=False, sharex=True, sharey=True)
fig.subplots_adjust(left=0.11, right=0.93, bottom=0.11, top=0.92, wspace=0.06, hspace=0.10)

for ri, comp in enumerate(COMPARTMENTS):
    for ci, tp in enumerate(TPS):
        ax  = axes[ri][ci]
        sub = obs[(obs['mac_identity'].astype(str) == comp) &
                  (obs['Timepoint'].astype(str) == tp)]

        # faint quadrant guide: top-left = burn-locked, bottom-right = resolved
        ax.axhspan(0, yhi, xmin=0, xmax=0.5, color=TYPE_PAL['Burn'], alpha=0.03, zorder=0)
        ax.axhspan(ylo, 0, xmin=0.5, xmax=1, color=TYPE_PAL['Sham'], alpha=0.03, zorder=0)
        ax.axhline(0, ls=':', lw=0.9, c='grey', alpha=0.7, zorder=0)
        ax.axvline(0, ls=':', lw=0.9, c='grey', alpha=0.7, zorder=0)

        cents = {}
        for cond in ['Sham', 'Burn']:
            c = sub[sub['Type'] == cond]
            if len(c) == 0:
                continue
            ax.scatter(c[XK], c[YK], s=6, alpha=0.30, color=TYPE_PAL[cond],
                       edgecolors='none', rasterized=True, zorder=1)
            if len(c) >= 12:
                try:
                    sns.kdeplot(data=c, x=XK, y=YK, ax=ax, color=TYPE_PAL[cond],
                                levels=4, thresh=0.15, linewidths=1.4, alpha=0.9, zorder=3)
                except Exception:
                    pass
            cents[cond] = c[[XK, YK]].mean().values

        # Sham → Burn shift arrow + centroids
        #if {'Sham', 'Burn'} <= cents.keys():
        #    ax.annotate("", xy=cents['Burn'], xytext=cents['Sham'], zorder=7,
               #         arrowprops=dict(arrowstyle='-|>', lw=2.2, color='black', mutation_scale=18))
        for cond, ctr in cents.items():
            ax.scatter(*ctr, s=170, color=TYPE_PAL[cond], edgecolor='black', lw=1.8, zorder=8)

        # KS (Burn vs Sham on metabolic ratio) — pinned top-left with white bg
        b = sub.loc[sub['Type'] == 'Burn', YK].dropna(); h = sub.loc[sub['Type'] == 'Sham', YK].dropna()
        txt = f"KS D={ks_2samp(b, h)[0]:.2f} {star(ks_2samp(b, h)[1])}" if len(b) > 2 and len(h) > 2 else "n.s. (low n)"
        ax.text(0.50, 0.97, f"{txt}", transform=ax.transAxes, # \nB={len(b)} S={len(h)}
                ha='left', va='top', fontsize=18, fontweight='bold', zorder=10,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))

        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel(''); ax.set_ylabel('')           # <- kill seaborn auto-labels
        ax.tick_params(labelsize=25, width=1.2, length=5)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels(): lbl.set_fontweight('bold')
        if ri == 0:
            ax.set_title(tp, fontsize=30, fontweight='bold', pad=5)
    # compartment label on the RIGHT of each row (left side stays clean)
    axes[ri][ncol - 1].annotate(comp, xy=(1.05, 0.5), xycoords='axes fraction',
                                rotation=270, ha='left', va='center',
                                fontsize=19, fontweight='bold')

# single directional axis labels for the whole figure
fig.supxlabel("Recruited ↔ Resident (Identity Score)",
              fontsize=25, fontweight='bold', y=0.04)
fig.supylabel("OXPHOS ↔ Glycolytic (Metabolic Ratio)",
              fontsize=25, fontweight='bold', x=0.025)

handles = [Line2D([0], [0], color=TYPE_PAL[c], lw=6, label=c) for c in ['Sham', 'Burn']]
#handles += [Line2D([0], [0], color='black', lw=2, marker='>', markersize=8, label='Sham → Burn shift')]
fig.legend(handles=handles, fontsize=16, loc='upper center',
           bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)

fig.savefig(f'{FIG}/mac_metabolicratio_vs_recr2res_compartment_x_timepoint.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG}/mac_metabolicratio_vs_recr2res_compartment_x_timepoint.png', dpi=300, bbox_inches='tight')
plt.show()

