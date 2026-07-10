"""KEYSTONE: Metabolic_Ratio (Glyco−OXPHOS)  vs  Recruited→Resident axis,

Source: macrophages_resident_recruited.ipynb
Libraries: gseapy, matplotlib, re, scipy, statsmodels
Key calls: gp.get_library, plt.show, plt.subplots, sc.tl.score_genes, scatter, sns.kdeplot
"""

# ══════════════════════════════════════════════════════════════════════════════
# KEYSTONE: Metabolic_Ratio (Glyco−OXPHOS)  vs  Recruited→Resident axis,
# one panel per compartment, Burn vs Sham overlaid.
# Tests: burn is LOCKED in the recruited/glycolytic corner; sham resolves toward
# the resident/OXPHOS corner.  Self-contained from adata_mac.
# ══════════════════════════════════════════════════════════════════════════════
import re, numpy as np, pandas as pd, scanpy as sc
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp, spearmanr
from statsmodels.stats.multitest import multipletests

TYPE_PAL = {'Burn': '#C0392B', 'Sham': '#2980B9'}
FIG = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/figures"

# ── 1. metabolic ratio (glyco − oxphos): HIGH = glycolytic ───────────────────
if not {'Glycolysis_Score', 'OXPHOS_Score'}.issubset(adata_mac.obs.columns):
    import gseapy as gp
    h2m  = lambda g: g[0].upper() + g[1:].lower()
    hall = gp.get_library('MSigDB_Hallmark_2020', organism='Mouse')
    for key, name in [('Glycolysis', 'Glycolysis_Score'),
                      ('Oxidative Phosphorylation', 'OXPHOS_Score')]:
        genes = [h2m(g) for g in hall[key] if h2m(g) in adata_mac.var_names]
        sc.tl.score_genes(adata_mac, genes, score_name=name, use_raw=False, random_state=0)
adata_mac.obs['Metabolic_Ratio'] = adata_mac.obs['Glycolysis_Score'] - adata_mac.obs['OXPHOS_Score']

# ── 2. recruited→resident axis = z(Resident − Recruited identity score) ──────
try:
    _recr_genes, _res_genes = recruited_macs_genes, resident_macs_genes        # from cell 5
except NameError:                                                              # fallback
    _recr_genes = ["Ccr2","Ly6c2","Arg1","Fn1","Spp1","Trem2","Gpnmb","Cd9","Ms4a7","Itgax"]
    _res_genes  = ["Adgre1","Mertk","Timd4","Cd163","Mrc1","Folr2","Lyve1","Gas6",
                   "Selenop","C1qa","C1qb","C1qc","Pf4","Maf"]
for name, gl in [('Recruited_Score', _recr_genes), ('Resident_Score', _res_genes)]:
    g = [x for x in gl if x in adata_mac.var_names]
    sc.tl.score_genes(adata_mac, g, score_name=name, use_raw=False, random_state=0)
_z = lambda s: (s - s.mean()) / s.std()
adata_mac.obs['Recr_to_Res'] = _z(adata_mac.obs['Resident_Score'] - adata_mac.obs['Recruited_Score'])

obs    = adata_mac.obs.copy()
XK, YK = 'Recr_to_Res', 'Metabolic_Ratio'

# ── 3. three compartments (auto-match real labels) ───────────────────────────
have = set(obs['mac_identity'].astype(str))
find = lambda k: next((h for h in sorted(have) if k in h.lower()), None)
COMPARTMENTS = [x for x in [find('inflamm'), find('recruit'), find('resident')] if x]
print("compartments:", COMPARTMENTS)

xlo, xhi = np.nanpercentile(obs[XK], [1, 99])
ylo, yhi = np.nanpercentile(obs[YK], [1, 99])

# KS (Burn vs Sham on Metabolic_Ratio) + Spearman(axis vs metabolism) per compartment
rows = []
for comp in COMPARTMENTS:
    s = obs[obs['mac_identity'].astype(str) == comp]
    b = s.loc[s['Type'] == 'Burn', YK].dropna(); h = s.loc[s['Type'] == 'Sham', YK].dropna()
    D, p = ks_2samp(b, h) if len(b) > 2 and len(h) > 2 else (np.nan, np.nan)
    rho, _ = spearmanr(s[XK], s[YK], nan_policy='omit')
    rows.append(dict(comp=comp, D=D, p=p, rho=rho, n_burn=len(b), n_sham=len(h)))
ksdf = pd.DataFrame(rows); ok = ksdf['p'].notna()
ksdf.loc[ok, 'padj'] = multipletests(ksdf.loc[ok, 'p'], method='fdr_bh')[1]
print(ksdf.to_string(index=False))
ks_lookup = ksdf.set_index('comp')
star = lambda p: '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'

# ── 4. figure: one panel per compartment ─────────────────────────────────────
ncol = len(COMPARTMENTS)
fig, axes = plt.subplots(1, ncol, figsize=(5.6 * ncol, 6.2), squeeze=False,
                         sharex=True, sharey=True)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.88, wspace=0.07)
for ci, comp in enumerate(COMPARTMENTS):
    ax = axes[0][ci]; sub = obs[obs['mac_identity'].astype(str) == comp]
    for cond in ['Sham', 'Burn']:
        c = sub[sub['Type'] == cond]
        ax.scatter(c[XK], c[YK], s=8, alpha=0.30, color=TYPE_PAL[cond],
                   edgecolors='none', rasterized=True, zorder=1)
        sns.kdeplot(data=c, x=XK, y=YK, ax=ax, color=TYPE_PAL[cond],
                    levels=5, thresh=0.10, fill=True, alpha=0.18, zorder=2)
        sns.kdeplot(data=c, x=XK, y=YK, ax=ax, color=TYPE_PAL[cond],
                    levels=5, thresh=0.10, linewidths=1.6, alpha=0.9, zorder=3)
        ctr = c[[XK, YK]].mean().values
        ax.scatter(*ctr, s=240, color=TYPE_PAL[cond], edgecolor='black', lw=2, zorder=6)
    ax.axhline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
    ax.axvline(0, ls=':', lw=1, c='grey', alpha=0.8, zorder=0)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    r = ks_lookup.loc[comp]
    ax.text(0.04, 0.97, f"KS D={r.D:.2f} {star(r.padj)}\nρ={r.rho:+.2f}",
            transform=ax.transAxes, ha='left', va='top', fontsize=18, fontweight='bold')
    ax.set_title(comp, fontsize=23, fontweight='bold', pad=8)
    ax.set_xlabel("← Recruited        Resident →", fontsize=19, fontweight='bold')
    if ci == 0:
        ax.set_ylabel("← OXPHOS        Glycolytic →\n(Metabolic Ratio)",
                      fontsize=19, fontweight='bold', labelpad=8)
    ax.tick_params(labelsize=17, width=1.4, length=6)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels(): lbl.set_fontweight('bold')
handles = [Line2D([0], [0], color=TYPE_PAL[c], lw=6, label=c) for c in ['Sham', 'Burn']]
fig.legend(handles=handles, fontsize=20, loc='lower center',
           bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
fig.savefig(f'{FIG}/mac_metabolicratio_vs_recr2res_by_compartment.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG}/mac_metabolicratio_vs_recr2res_by_compartment.png', dpi=300, bbox_inches='tight')
plt.show()

