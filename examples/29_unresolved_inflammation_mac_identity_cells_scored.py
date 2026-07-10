"""UNRESOLVED INFLAMMATION — mac_identity cells scored for Inflammation vs Resoluti

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, numpy, re, scanpy, scipy, seaborn, statsmodels
Key calls: def ks_imbalance, def stars, def style_ax, gp.get_library, sc.tl.score_genes
"""

# ══════════════════════════════════════════════════════════════════════════════
# UNRESOLVED INFLAMMATION — mac_identity cells scored for Inflammation vs Resolution
# Burn vs Sham across the three origin identities and across timepoints
# Self-contained: requires only `adata_full` (obs: mac_identity / Type / Timepoint)
# ══════════════════════════════════════════════════════════════════════════════
import re
import numpy as np, pandas as pd
import scanpy as sc, gseapy as gp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

MAC_IDENTITIES = ["Inflammatory Monocytes", "MΦ-Recruited", "MΦ-Resident/Repair"]
TYPE_PAL = {'Burn': '#C0392B', 'Sham': '#2980B9'}
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_STAT = 26, 22, 22, 20, 18

# ── subset to the three macrophage/monocyte identities ────────────────────────
ident = adata_full.obs['mac_identity'].astype(str)
adata_mac = adata_full[ident.isin(MAC_IDENTITIES).values].copy()
adata_mac.obs['mac_identity'] = pd.Categorical(
    adata_mac.obs['mac_identity'].astype(str), categories=MAC_IDENTITIES, ordered=True)
print(f"{adata_mac.n_obs} cells:", adata_mac.obs['mac_identity'].value_counts().to_dict())

# ── gene modules ──────────────────────────────────────────────────────────────
h2m  = lambda g: g[0].upper() + g[1:].lower()                       # human -> mouse casing
hall = gp.get_library('MSigDB_Hallmark_2020', organism='Mouse')

# Inflammation = union of the three inflammatory Hallmark programs (robust key match)
_want = ['inflammatory response', 'tnf', 'interferon gamma']
inflam_keys  = [k for k in hall if any(w in k.lower() for w in _want)]
inflam_genes = sorted({h2m(g) for k in inflam_keys for g in hall[k]} & set(adata_mac.var_names))
print("Inflammation Hallmark sets:", inflam_keys, f"-> {len(inflam_genes)} genes")

# Resolution / repair = efferocytosis + pro-resolving + tissue-resident/M2
# (Arg1/Nos2 intentionally excluded so this axis is independent of the burn signal)
RESOLVE_GENES = [
    'Mertk','Gas6','Axl','Timd4','Anxa1',                                   # efferocytosis
    'Mrc1','Cd163','Stab1','Lyve1','Folr2','Cbr2','F13a1','Vsig4','Marco',  # tissue-resident / M2
    'Il10','Tgfb1','Tgfbi','Timp2','Igf1','Selenop','Nr4a1','Klf2','Klf4','Maf','Mafb',
    'Retnla','Chil3','Egr2','Cd36','C1qa','C1qb','C1qc','Apoe','Pf4',       # resident / homeostatic
]
resolve_genes = sorted(set(RESOLVE_GENES) & set(adata_mac.var_names))
missing = sorted(set(RESOLVE_GENES) - set(resolve_genes))
print(f"Resolution genes: {len(resolve_genes)} found | missing: {missing}")

sc.tl.score_genes(adata_mac, inflam_genes,  score_name='Inflammation', use_raw=False, random_state=0)
sc.tl.score_genes(adata_mac, resolve_genes, score_name='Resolution',   use_raw=False, random_state=0)

# ── tidy obs frame (z-score the two axes; Imbalance>0 = inflammation-dominant) ─
obs = adata_mac.obs[['Type','Timepoint','mac_identity','Inflammation','Resolution']].copy()
obs['Timepoint'] = obs['Timepoint'].astype(str)
z = lambda s: (s - s.mean()) / s.std()
obs['Inflammation_z'] = z(obs['Inflammation'])
obs['Resolution_z']   = z(obs['Resolution'])
obs['Imbalance']      = obs['Inflammation_z'] - obs['Resolution_z']

XK, YK = 'Inflammation_z', 'Resolution_z'
XLAB, YLAB = 'Inflammation (z)', 'Resolution / Repair (z)'
TP_ORDER = sorted(obs['Timepoint'].unique(), key=lambda t: int(re.search(r'\d+', t).group()))

FIG = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/figures"

def stars(p):
    if not np.isfinite(p): return 'n/a'
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'

def style_ax(ax, show_y=True):
    ax.tick_params(axis='both', labelsize=FS_TICK, width=1.4, length=6)
    ax.tick_params(axis='y', labelleft=show_y)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_linewidth(1.4); ax.spines['bottom'].set_linewidth(1.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

def ks_imbalance(sub):
    b = sub.loc[sub['Type'] == 'Burn', 'Imbalance'].dropna().values
    s = sub.loc[sub['Type'] == 'Sham', 'Imbalance'].dropna().values
    return ks_2samp(b, s) if len(b) > 2 and len(s) > 2 else (np.nan, np.nan)

xlo, xhi = np.nanpercentile(obs[XK], [1, 99])
ylo, yhi = np.nanpercentile(obs[YK], [1, 99])

