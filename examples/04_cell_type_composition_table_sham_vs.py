"""Cell-type composition table: Sham vs Burn proportions across timepoints

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, numpy, pandas, pathlib, re, scanpy, scipy, statsmodels
Key calls: .plot, def msem, def render_prop_table, def stars, def tp_per_cell, def transform, dendrogram, dotplot, plt.close, plt.show, plt.subplots, sc.pl.dotplot
"""

# ── Cell-type composition table: Sham vs Burn proportions across timepoints ──────
#   rows = cell types (cell_types_full); columns = D7 / D10 / D14 / D19
#   per-sample proportions -> logit transform -> within-timepoint Student's t-test.
import re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from pathlib import Path

CT_COL      = "cell_types_full"
SAMPLE_COL  = None          # None = auto-detect; else set explicitly e.g. "Sample"
SHAM, BURN  = "Sham", "Burn"
TRANSFORM   = "logit"       # "logit" | "asin" | "none"  (variance-stabilize proportions)
STAR_FROM   = "p"           # "p" = nominal p<0.05  |  "fdr" = BH-FDR within timepoint
obs = adata_full.obs

FIGDIR_MAC = globals().get("FIGDIR_MAC",
    Path("/Users/sujitsilas/Desktop/Philip Scumpia Lab/"
         "burn_sham_scrnaseq_macs_20260608/figures/mac_identity"))
FIGDIR_MAC.mkdir(parents=True, exist_ok=True)

# ── locate sample / condition / timepoint columns ───────────────────────────────
SAMP_CANDS = ["Sample","sample","orig.ident","orig_ident","SampleID","sample_id",
              "library","Library","mouse","Mouse","replicate","Replicate","batch"]
samp_col = SAMPLE_COL or next((c for c in SAMP_CANDS if c in obs.columns), None)
assert samp_col, f"set SAMPLE_COL manually from: {list(obs.columns)}"
type_col = next((c for c in ["Type","type","condition","Condition","group","Group"]
                 if c in obs.columns), None)
assert type_col, f"set condition col manually from: {list(obs.columns)}"
tp_src = next((c for c in obs.columns if "time" in c.lower()), None)
assert tp_src, f"no timepoint-like column in: {list(obs.columns)}"
tp = obs[tp_src].astype(str).str.extract(r"(\d+)")[0].radd("D")

df = pd.DataFrame({"sample": obs[samp_col].astype(str).values,
                   "Type":   obs[type_col].astype(str).values,
                   "Timepoint": tp.values,
                   "ct":     obs[CT_COL].astype(str).values})
df = df[~df["ct"].isin(["nan", "None"])]

ct_order = (list(obs[CT_COL].cat.categories) if hasattr(obs[CT_COL], "cat")
            else sorted(df["ct"].unique()))
ct_order = [c for c in ct_order if c in set(df["ct"])]
tp_order = sorted(df["Timepoint"].unique(), key=lambda t: int(re.search(r"\d+", t).group()))

# ── DIAGNOSTIC: how many samples (replicates) per condition × timepoint? ─────────
nmat = (df.drop_duplicates(["sample", "Type", "Timepoint"])
          .groupby(["Type", "Timepoint"]).size().unstack("Timepoint").reindex(columns=tp_order))
print(f"sample column = {samp_col!r}\nReplicates per condition × timepoint:")
print(nmat.fillna(0).astype(int), "\n")
if (nmat.fillna(0) < 2).any().any():
    print("⚠ some condition×timepoint groups have <2 replicates → no valid test there.\n")

# ── per-sample composition (% of cells within each sample-timepoint) ─────────────
wide = (df.groupby(["sample", "Type", "Timepoint", "ct"]).size()
          .unstack("ct", fill_value=0).reindex(columns=ct_order, fill_value=0))
prop = wide.div(wide.sum(1), axis=0).mul(100).reset_index()   # percent per sample

def transform(frac):
    frac = np.clip(np.asarray(frac, float) / 100.0, 1e-3, 1 - 1e-3)
    if TRANSFORM == "logit": return np.log(frac / (1 - frac))
    if TRANSFORM == "asin":  return np.arcsin(np.sqrt(frac))
    return frac * 100
def msem(x):
    x = x[np.isfinite(x)]
    if len(x) == 0: return (np.nan, np.nan)
    return (float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 1 else 0.0)
def stars(p):
    if not np.isfinite(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

# ── within-timepoint Student's t-test on transformed per-sample proportions ──────
rows, rawp = [], {}
for ct in ct_order:
    for tp_ in tp_order:
        s = prop[(prop.Type == SHAM) & (prop.Timepoint == tp_)][ct].values
        b = prop[(prop.Type == BURN) & (prop.Timepoint == tp_)][ct].values
        sm, ss = msem(s); bm, bs = msem(b)
        p = (ttest_ind(transform(b), transform(s), equal_var=True).pvalue
             if (len(b) >= 2 and len(s) >= 2) else np.nan)
        rawp[(ct, tp_)] = p
        rows.append(dict(cell_type=ct, Timepoint=tp_, n_sham=len(s), n_burn=len(b),
                         Sham_mean=sm, Sham_sem=ss, Burn_mean=bm, Burn_sem=bs,
                         delta_pp=bm - sm, p=p))
tidy = pd.DataFrame(rows)

# BH-FDR *within each timepoint* (not across all 40 tests)
fdr = {k: np.nan for k in rawp}
for tp_ in tp_order:
    keys = [(ct, tp_) for ct in ct_order if np.isfinite(rawp[(ct, tp_)])]
    if keys:
        q = multipletests([rawp[k] for k in keys], method="fdr_bh")[1]
        fdr.update(dict(zip(keys, q)))
tidy["fdr"] = [fdr[(r.cell_type, r.Timepoint)] for r in tidy.itertuples()]
sig_val = tidy["p"] if STAR_FROM == "p" else tidy["fdr"]
tidy["sig"] = [stars(v) or "n.s." for v in sig_val]
tidy.to_csv(FIGDIR_MAC / "table_celltype_proportions_by_timepoint.csv", index=False)
print(tidy.to_string(index=False))

# ── render: rows = cell types, columns = timepoints ─────────────────────────────
disp = pd.DataFrame(index=ct_order, columns=tp_order, dtype=object)
sig  = pd.DataFrame(False, index=ct_order, columns=tp_order)
for ct in ct_order:
    for tp_ in tp_order:
        r = tidy[(tidy.cell_type == ct) & (tidy.Timepoint == tp_)].iloc[0]
        v = r.p if STAR_FROM == "p" else r.fdr
        disp.loc[ct, tp_] = (f"Sham {r.Sham_mean:4.1f}±{r.Sham_sem:.1f}\n"
                             f"Burn {r.Burn_mean:4.1f}±{r.Burn_sem:.1f} {stars(v)}").rstrip()
        sig.loc[ct, tp_] = np.isfinite(v) and v < 0.05

def render_prop_table(disp, sig, title, stem):
    cts, tps = list(disp.index), list(disp.columns)
    cols = ["Cell type"] + tps
    cellText = [[ct] + [disp.loc[ct, tp_] for tp_ in tps] for ct in cts]
    nrow, ncol = len(cts), len(cols)
    fig, ax = plt.subplots(figsize=(2.0*ncol + 1.0, 0.75*nrow + 1.4)); ax.axis("off")
    tbl = ax.table(cellText=cellText, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.3)
    tbl.auto_set_column_width(col=list(range(ncol)))
    for j in range(ncol):
        c = tbl[0, j]; c.set_facecolor("#34495E"); c.set_text_props(color="white", fontweight="bold")
    for i, ct in enumerate(cts):
        tbl[i+1, 0].set_text_props(fontweight="bold")
        for j, tp_ in enumerate(tps):
            if sig.loc[ct, tp_]:
                tbl[i+1, j+1].set_facecolor("#FDEDEC")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIGDIR_MAC / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGDIR_MAC / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

lab = "nominal p" if STAR_FROM == "p" else "BH-FDR (within timepoint)"
render_prop_table(
    disp, sig,
    "Cell-type composition (% of cells per sample) — Burn vs Sham across timepoints\n"
    f"mean ± SEM;  Student's t on {TRANSFORM}-proportions;  * {lab}<0.05, ** <0.01, *** <0.001",
    "table_celltype_proportions_by_timepoint")


adata_full.obs["cell_types_simple"].unique()

import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

CT_SIMPLE = 'cell_types_simple'
assert CT_SIMPLE in adata_full.obs, f"{CT_SIMPLE!r} not in adata_full.obs"

# ── curated markers, grouped by lineage (order ≈ structural → immune) ──────────
MARKERS = {
    'Krt':     ['Krt5','Krt14','Krt1','Krt10','Krt15','Dsp','Pkp1','Perp','Lgals7'],
    'Seb':     ['Scd1','Elovl4','Elovl6','Dhcr24','Sdr16c6','Mgst1','Far2'],
    'Endo.':   ['Pecam1','Egfl7','Cldn5','Cdh5','Flt1','Emcn'],
    'Fibs':    ['Col1a1','Col1a2','Col3a1','Col5a1','Pdgfra','Dpt','Sparc','Dcn','Lum','Crabp1'],
    'Smcs':    ['Acta1','Acta2','Myh11','Tagln','Myl9','Ttn','Tnnt3','Tnnc2','Atp2a1','Des'],
    'T Cells': ['Cd3e','Cd3d','Cd3g','Cd8a','Nkg7','Ccl5'],
    'cDCs':    ['Cd74','H2-Aa','H2-Ab1','H2-Eb1','Flt3','Xcr1','Itgax'],
    'Mono.':   ['Ly6c2','Plac8','Ccr2','Vcan','F13a1','Chil3','Gngt2'],
    'MΦ':      ['Lyz2','Ctss','Mrc1','Trem2','C1qa','C1qb','Adgre1','Csf1r','Tgfbi','Mmp12','Arg1','Nos2'],
    'Neu':     ['S100a8','S100a9','Retnlg','Mpo','Mmp9','Ly6g','Csf3r','Cxcr2','Cxcl2','Il1b','Srgn'],
}

# ── keep only genes present in this object (CellRanger naming varies) ──────────
present_markers, missing = {}, []
for grp, genes in MARKERS.items():
    keep = [g for g in genes if g in adata_full.var_names]
    missing += [g for g in genes if g not in adata_full.var_names]
    if keep:
        present_markers[grp] = keep
if missing:
    print('Not in adata_full.var_names (skipped):', sorted(set(missing)))

n_genes  = sum(len(v) for v in present_markers.values())
n_groups = adata_full.obs[CT_SIMPLE].astype(str).nunique()

FIGDIR_DP = OUT / 'figures' / 'dotplot'
FIGDIR_DP.mkdir(parents=True, exist_ok=True)

# ── build dot plot, keep the handle so we can style each axis ──────────────────
dp = sc.pl.dotplot(
    adata_full,
    var_names=present_markers,
    groupby=CT_SIMPLE,
    standard_scale='var',            # scale each gene 0–1 across groups
    cmap='plasma_r',                 # yellow (low) → magenta → blue (high)
    dendrogram=True,                 # cluster the cell types
    figsize=(24, 3.5),
    return_fig=True,
)

# fixed-width legend column so the colorbar + size dots aren't compressed
dp.legend(width=2.8, colorbar_title='Scaled\nexpression', size_title='% expressing')
dp.make_figure()
axd = dp.get_axes()

# ── publication styling ───────────────────────────────────────────────────────
main_ax = axd['mainplot_ax']
for lab in main_ax.get_xticklabels():                 # gene names (bottom)
    lab.set_fontsize(15); lab.set_fontweight('bold')
for lab in main_ax.get_yticklabels():                 # cell_types_simple (rows)
    lab.set_fontsize(18); lab.set_fontweight('bold')

# top lineage group labels — bold + slight rotation
gg_ax = axd.get('gene_group_ax')
if gg_ax is not None:
    for txt in gg_ax.texts:
        txt.set_fontsize(17); txt.set_fontweight('bold')
        txt.set_rotation(30); txt.set_ha('left'); txt.set_va('bottom')
        txt.set_rotation_mode('anchor')

# legend fonts (best-effort; keys may differ by scanpy version)
for key in ('color_legend_ax', 'size_legend_ax'):
    ax = axd.get(key)
    if ax is None:
        continue
    if ax.title:
        ax.title.set_fontsize(14); ax.title.set_fontweight('bold')
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(12)

fig = dp.fig
fig.savefig(FIGDIR_DP / 'dotplot_cell_types_simple.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIGDIR_DP / 'dotplot_cell_types_simple.png', dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)
print(f'Saved dotplot: {n_genes} genes × {n_groups} groups → {FIGDIR_DP}')


adata_full.write_h5ad("/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/filtered_final_06252026.h5ad")

adata_mac.write_h5ad("/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/macrophages_final_06252026.h5ad")

adata_mac

from matplotlib.ticker import PercentFormatter

SPLIT_COL = "Type_Timepoint_C"
labels = ["Inflammatory Monocytes", "MΦ-Recruited", "MΦ-Resident/Repair"]
id_palette = dict(zip(labels, ["#D62728", "#F39C12", "#2CA02C"]))  # red, orange, green

# desired Sham/Burn × timepoint order
timepoint_order = ["Sham D7", "Burn D7", "Sham D10", "Burn D10",
                   "Sham D14", "Burn D14", "Sham D19", "Burn D19"]

assert SPLIT_COL in adata_full.obs.columns, (
    f"{SPLIT_COL!r} not found. Candidates: "
    f"{[c for c in adata_full.obs.columns if 'imepoint' in c or 'Type' in c]}"
)

# ── macrophage compartment (Mono/MDM + Mφ) ────────────────────────────────────
target_groups = ["Mono/MDM", "Mφ"]
comp_col = next((c for c in adata_full.obs.columns
                 if adata_full.obs[c].astype(str).isin(target_groups).any()), None)
mask = (adata_full.obs[comp_col].astype(str).isin(target_groups).values
        if comp_col is not None else np.ones(adata_full.n_obs, dtype=bool))

# ── proportions within each Type_Timepoint_C (unassigned/Others excluded) ─────
comp = adata_full.obs.loc[mask, [SPLIT_COL, "mac_identity"]].copy()
comp = comp[comp["mac_identity"].isin(labels)]
prop = pd.crosstab(comp[SPLIT_COL], comp["mac_identity"], normalize="index")[labels]

# apply the requested order; keep only groups that exist, append any unexpected ones
present = [g for g in timepoint_order if g in prop.index]
leftover = [g for g in prop.index if g not in timepoint_order]
if leftover:
    print("WARNING: groups not in timepoint_order (check label format), appended:", leftover)
prop = prop.reindex(present + leftover)
print(prop.round(3))

# ── figure (compact, bars close together) ─────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
})

n = len(prop)
fig, ax = plt.subplots(figsize=(10, 6.5))

x = np.arange(n)
bottom = np.zeros(n)
for lab in labels:
    ax.bar(x, prop[lab].values, bottom=bottom, width=0.8,   # near-touching bars
           color=id_palette[lab], edgecolor="white", linewidth=1.0, label=lab)
    bottom += prop[lab].values

ax.set_xticks(x)
ax.set_xticklabels(prop.index, fontsize=20, fontweight="bold",
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel(SPLIT_COL, fontsize=26, fontweight="bold", labelpad=8)
ax.set_ylabel("Proportion of cells", fontsize=26, fontweight="bold", labelpad=8)
ax.set_ylim(0, 1)
ax.set_xlim(-0.6, n - 0.4)        # trims side padding -> compact
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.tick_params(axis="y", labelsize=20, width=1.5, length=6)
ax.margins(x=0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(fontsize=16, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "proportions_by_type_timepoint.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "proportions_by_type_timepoint.pdf", bbox_inches="tight")
plt.show()


import re
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

SPLIT_COL  = "Type_Timepoint_C"
labels     = ["Inflammatory Monocytes", "MΦ-Recruited", "MΦ-Resident/Repair"]
id_palette = dict(zip(labels, ["#D62728", "#F39C12", "#2CA02C"]))   # red, orange, green

assert SPLIT_COL in adata_full.obs.columns, f"{SPLIT_COL!r} not in adata_full.obs"

# ── macrophage compartment (Mono/MDM + Mφ) ────────────────────────────────────
target_groups = ["Mono/MDM", "Mφ"]
comp_col = next((c for c in adata_full.obs.columns
                 if adata_full.obs[c].astype(str).isin(target_groups).any()), None)
mask = (adata_full.obs[comp_col].astype(str).isin(target_groups).to_numpy()
        if comp_col is not None else np.ones(adata_full.n_obs, dtype=bool))

# ── proportions of the 3 identities within each Type_Timepoint_C group ─────────
comp = adata_full.obs.loc[mask, [SPLIT_COL, "mac_identity"]].copy()
comp = comp[comp["mac_identity"].astype(str).isin(labels)]
prop = pd.crosstab(comp[SPLIT_COL].astype(str), comp["mac_identity"], normalize="index")[labels]

# ── split each Type_Timepoint_C value -> Type + Timepoint ─────────────────────
recs = []
for grp, row in prop.iterrows():
    s  = str(grp)
    ty = "Burn" if "burn" in s.lower() else ("Sham" if "sham" in s.lower() else None)
    dm = re.search(r"\d+", s)
    if ty is None or dm is None:
        print("skipped unparsed group:", s); continue
    tp = f"D{dm.group()}"
    for lab in labels:
        recs.append({"Type": ty, "Timepoint": tp, "identity": lab, "prop": row[lab]})
long = pd.DataFrame(recs)
tp_order = sorted(long["Timepoint"].unique(), key=lambda t: int(re.search(r"\d+", t).group()))
xpos = {t: i for i, t in enumerate(tp_order)}
ls = {"Burn": "-", "Sham": "--"}

# ── line plot ─────────────────────────────────────────────────────────────────
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

fig, ax = plt.subplots(figsize=(9, 6.5))
for lab in labels:
    for ty in ["Sham", "Burn"]:
        sub = (long[(long.identity == lab) & (long.Type == ty)]
               .set_index("Timepoint").reindex(tp_order))
        ax.plot([xpos[t] for t in tp_order], sub["prop"].values,
                color=id_palette[lab], linestyle=ls[ty], lw=3,
                marker="o", ms=9, markeredgecolor="white", markeredgewidth=1.0)

ax.set_xticks(range(len(tp_order)))
ax.set_xticklabels(tp_order, fontsize=20, fontweight="bold")
ax.set_xlabel("Timepoint", fontsize=26, fontweight="bold", labelpad=8)
ax.set_ylabel("Proportion of cells", fontsize=26, fontweight="bold", labelpad=8)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.tick_params(axis="y", labelsize=20, width=1.5, length=6)
ax.grid(axis="y", alpha=0.3)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

id_handles = [Line2D([0], [0], color=id_palette[l], lw=3, label=l) for l in labels]
ty_handles = [Line2D([0], [0], color="black", lw=3, ls="-",  label="Burn"),
              Line2D([0], [0], color="black", lw=3, ls="--", label="Sham")]
leg1 = ax.legend(handles=id_handles, title="Identity", loc="upper left",
                 bbox_to_anchor=(1.02, 1.0), fontsize=14, title_fontsize=15, frameon=False)
ax.add_artist(leg1)
ax.legend(handles=ty_handles, title="Condition", loc="upper left",
          bbox_to_anchor=(1.02, 0.45), fontsize=14, title_fontsize=15, frameon=False)

fig.tight_layout()
fig.savefig(FIGDIR_MAC / "proportions_by_type_timepoint_lines.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "proportions_by_type_timepoint_lines.pdf", bbox_inches="tight")
plt.show()


from scipy.stats import ttest_ind
import re
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter

ID_COL = "mac_identity"

# alias -> canonical display label (covers BOTH the old and current labelings;
# canonical labels pass through unchanged). dict order = plot order.
ALIASES = {
    "Inflammatory Monocytes": "Inflammatory Monocytes",
    "Inflammatory Monocyte":  "Inflammatory Monocytes",
    "Recruited Macrophages":  "MΦ-Recruited",
    "MΦ-Recruited":           "MΦ-Recruited",
    "Resident Macrophages":   "MΦ-Resident/Repair",
    "MΦ-Resident/Repair":     "MΦ-Resident/Repair",
    "MΦ-Resident":            "MΦ-Resident/Repair",
}
labels = ["Inflammatory Monocytes", "MΦ-Recruited", "MΦ-Resident/Repair"]

raw_vals = sorted(adata_mac.obs[ID_COL].astype(str).unique())
print("raw mac_identity values:", raw_vals)
unmapped = [v for v in raw_vals if ALIASES.get(v, v) not in labels]
if unmapped:
    print(f"  NOTE: these values are not among the 3 identities and will be excluded: {unmapped}")

# ── sample column ─────────────────────────────────────────────────────────────
SAMP_CANDS = ["Sample", "sample", "orig.ident", "orig_ident", "SampleID", "sample_id",
              "library", "Library", "mouse", "Mouse", "replicate", "Replicate", "batch"]
sample_col = next((c for c in SAMP_CANDS if c in adata_mac.obs.columns), None)
assert sample_col is not None, f"Set sample col manually from: {list(adata_mac.obs.columns)}"

# ── timepoint per cell ('D#') ─────────────────────────────────────────────────
def tp_per_cell(obs):
    for c in obs.columns:
        if "time" in c.lower(): return obs[c].astype(str)
    for c in obs.columns:
        if obs[c].astype(str).str.fullmatch(r"D?\d+").mean() > 0.8: return obs[c].astype(str)
    for c in obs.columns:
        ext = obs[c].astype(str).str.extract(r"(D\d+)")[0]
        if ext.notna().mean() > 0.8: return ext
    raise ValueError("no timepoint column")
tp_norm = tp_per_cell(adata_mac.obs).astype(str).str.extract(r"(\d+)")[0].radd("D")

# ── per-sample proportions, using canonical display labels ────────────────────
ident = adata_mac.obs[ID_COL].astype(str).map(lambda v: ALIASES.get(v, v))   # pass-through
work = pd.DataFrame({"sample": adata_mac.obs[sample_col].astype(str).values,
                     "Type":   adata_mac.obs["Type"].astype(str).values,
                     "Timepoint": tp_norm.values,
                     "identity": ident.values})
work = work[work["identity"].isin(labels)]
assert len(work), "No cells matched the 3 identities — check ALIASES against the printed raw values."
print("cells per identity:", work["identity"].value_counts().to_dict())

counts = work.groupby(["sample", "Type", "Timepoint", "identity"]).size().unstack("identity", fill_value=0)
for lab in labels:                                            # guarantee all columns exist
    if lab not in counts.columns: counts[lab] = 0
counts = counts[labels]                                       # fixed column order
props = counts.div(counts.sum(1), axis=0).reset_index()

tp_order = sorted(props["Timepoint"].unique(), key=lambda t: int(re.search(r"\d+", t).group()))
xpos = {t: i for i, t in enumerate(tp_order)}

# ── palette: reuse existing (remapped via ALIASES) or fall back to defaults ────
_defaults = {"Inflammatory Monocytes": "#C0392B",
             "MΦ-Recruited": "#E69138", "MΦ-Resident/Repair": "#2E9E4F"}
try:
    id_palette = {ALIASES.get(k, k): v for k, v in id_palette.items()}
except (NameError, AttributeError):
    id_palette = {}
id_palette = {lab: id_palette.get(lab, _defaults[lab]) for lab in labels}

# ── stats (Burn vs Sham, per identity/timepoint) ──────────────────────────────
def stars(p):
    if not np.isfinite(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
sig = {}
for lab in labels:
    for tp in tp_order:
        b = props[(props.Type == "Burn") & (props.Timepoint == tp)][lab].dropna().values
        s = props[(props.Type == "Sham") & (props.Timepoint == tp)][lab].dropna().values
        sig[(lab, tp)] = ttest_ind(b, s, equal_var=False).pvalue if len(b) >= 2 and len(s) >= 2 else np.nan

# ── per-sample mean ± SEM ─────────────────────────────────────────────────────
psm  = props.melt(id_vars=["sample", "Type", "Timepoint"], value_vars=labels,
                  var_name="identity", value_name="prop")
summ = (psm.groupby(["identity", "Type", "Timepoint"])["prop"]
            .agg(mean="mean", sem="sem").reset_index())

# ── plot ──────────────────────────────────────────────────────────────────────
conds       = ["Sham", "Burn"]
title_color = {"Sham": "#2471A3", "Burn": "#C0392B"}
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

fig, axes = plt.subplots(2, 1, figsize=(6, 9), sharey=True)
for ax, ty in zip(axes, conds):
    for lab in labels:
        s_ = (summ[(summ.identity == lab) & (summ.Type == ty)]
              .set_index("Timepoint").reindex(tp_order))
        means = s_["mean"].values
        sems  = np.nan_to_num(s_["sem"].values)
        x = [xpos[t] for t in tp_order]
        ax.errorbar(x, means, yerr=sems, fmt="-o", color=id_palette[lab], lw=3,
                    ms=10, markeredgecolor="white", markeredgewidth=1.0,
                    capsize=5, capthick=2, elinewidth=2, label=lab, zorder=3)
        for k, tp in enumerate(tp_order):
            txt = stars(sig.get((lab, tp), np.nan))
            if txt:
                ax.annotate(txt, (x[k], means[k] + sems[k]), xytext=(0, 8),
                            textcoords="offset points", ha="center", va="bottom",
                            color=id_palette[lab], fontsize=18, fontweight="bold")
    ax.set_title(ty, fontsize=28, fontweight="bold", color=title_color[ty], pad=10)
    ax.set_xticks(range(len(tp_order)))
    ax.set_xticklabels(tp_order, fontsize=25, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="y", labelsize=20, width=1.5, length=6)
    ax.grid(axis="y", alpha=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

for ax in axes:
    ax.set_ylabel("Proportion of cells", fontsize=24, fontweight="bold", labelpad=8)
axes[1].legend(title="Identity", fontsize=14, title_fontsize=15, frameon=False,
               loc="center left", bbox_to_anchor=(1.02, 0.5))

fig.subplots_adjust(left=0.20, right=0.78, top=0.94, bottom=0.07, hspace=0.42)
fig.savefig(FIGDIR_MAC / "proportions_lines_split_by_condition_stats.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "proportions_lines_split_by_condition_stats.pdf", bbox_inches="tight")
plt.show()


from matplotlib.lines import Line2D

# dot-only handles: marker with no connecting line
handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=20,
                  markerfacecolor=id_palette[lab], markeredgecolor='black',
                  markeredgewidth=1.3, label=lab) for lab in labels]

fig_leg, ax_leg = plt.subplots(figsize=(4.2, 0.7 * len(labels) + 0.4))
ax_leg.axis('off')
leg = ax_leg.legend(handles=handles, loc='center', frameon=False,
                    ncol=1, labelspacing=0.6, handletextpad=0.6,
                    fontsize=22, borderpad=0)
for txt in leg.get_texts():
    txt.set_fontweight('bold')

fig_leg.savefig(FIGDIR_MAC / "proportions_legend_vertical.pdf", bbox_inches='tight')
fig_leg.savefig(FIGDIR_MAC / "proportions_legend_vertical.png", dpi=300, bbox_inches='tight')
plt.show()


adata_mac = sc.read_h5ad("/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/macrophages_burn_sham.h5ad")

import pandas as pd
import numpy as np

LABELS = ["Inflammatory Monocytes", "MΦ-Recruited", "MΦ-Resident/Repair"]
ANNOT_CSV = "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/mac_identity_annotations.csv"

# ── 1. pull the 3-identity annotation off adata_full, keyed by cell id ─────────
ann = adata_full.obs["mac_identity"].astype(str)
ann = ann[ann.isin(LABELS)]            # keep only the 3 macrophage identities
ann.name = "mac_identity"
ann.index.name = "cell_id"
ann.to_csv(ANNOT_CSV)
print(f"Saved {len(ann)} annotations -> {ANNOT_CSV}")
print(ann.value_counts(), "\n")

# ── 2. merge onto adata_mac by cell id (barcode) ──────────────────────────────
mapped = ann.reindex(adata_mac.obs_names)
n_match = int(mapped.notna().sum())
print(f"Matched {n_match}/{adata_mac.n_obs} adata_mac cells by barcode "
      f"({100*n_match/adata_mac.n_obs:.1f}%)")

if n_match == 0:
    # barcodes are formatted differently between the two objects — inspect and fix
    print("\n⚠ No matches. Example cell ids:")
    print("  adata_mac :", list(adata_mac.obs_names[:3]))
    print("  annotation:", list(ann.index[:3]))
else:
    adata_mac.obs["mac_identity"] = pd.Categorical(mapped.values, categories=LABELS)
    print("\nadata_mac.obs['mac_identity'] value counts:")
    print(adata_mac.obs["mac_identity"].value_counts(dropna=False))

# ── 3. (optional) write the annotated object back out ─────────────────────────
# adata_mac.write_h5ad(
#     "/Users/sujitsilas/Desktop/Philip Scumpia Lab/burn_sham_scrnaseq_macs_20260608/"
#     "macrophages_burn_sham_annotated.h5ad"
# )


mac_colors = {
    'MΦ-Inf'  : '#FA8072',
    'MΦ-Act' : '#E31A1C',
    'MΦ-IFN/AS DCs' : '#1F78B4',
    'Early MDM'     : '#6A3D9A',
    'MΦ-Res/Rep' : '#33A02C',
    'LAM'     : '#FB9A99',
    'LAM-I'   : '#FDBF6F',
    'LAM-II'  : '#B15928',
    'Inf. Mono.' : '#FF7F00',
}

# 2. Define the desired order for plotting/categories
desired_order = {
    'MΦ-Inf',
    'MΦ-Act',
        'Inf. Mono.',
    'MΦ-IFN/AS DCs',
        'LAM-I',
    'LAM-II',
    'MΦ-Res/Rep',
    'Early MDM',
}

# 3. Map your clusters based on the DEG analysis
mac_subset_ids = {
    "0": "MΦ-Act",   # Nos2-hi, Arg1-hi, Ptges, Ptgs2 — inflammatory M1
    "1": "MΦ-Res/Rep",   # C1qa/b/c, Mrc1-hi, Mertk, Adgre1, Gas6; Slc40a1/Spic = iron-recycling resident
    "2": "MΦ-Inf",    # Arg1, Vegfa, Hilpda, Pdpn — hypoxic/angiogenic, not inflammatory
    "3": "Early MDM",       # Ciita, H2-Ab1/Aa/Eb1, Cd74; lipid-program-low (Msr1/Abca1 deep-neg)
    "4": "MΦ-Res/Rep",   # Stab1, Maf, Hpgds, C4b, Apoe, Mrc1-hi — reparative
    "5": "LAM-I",       # Spp1, Lgals3, Ctsb/l/s, Fth1-hi, Hmox1-hi, Grn, Cd68-peak
    "6": "Inf. Mono.",       # Ccr2, Vcan, Cd80, Tlr2, Osm, Olr1+ — recruited monocyte-derived
    "7": "LAM-II",    # PPP/redox (H6pd, Tkt, Prdx1, Txn1), Smad7/Igf2r-hi
    "8": "MΦ-Act",     # Cd36, Msr1, Abca1, Plin2, Spp1, Trem2, Gpnmb, Hmox1-hi, Fth1-hi
    "9": "MΦ-IFN/AS DCs",   # Irf7, Rsad2, Isg15, Stat1/2, Ifit3 — interferon
}

adata_mac
