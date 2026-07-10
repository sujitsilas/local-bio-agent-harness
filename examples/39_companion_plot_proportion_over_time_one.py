"""Companion plot: proportion over time, ONE PANEL PER COMPARTMENT, Burn vs Sham

Source: macrophages_resident_recruited.ipynb
Libraries: matplotlib, re, scipy
Key calls: def stars, def tp_per_cell, plt.show, plt.subplots
"""

# ══════════════════════════════════════════════════════════════════════════════
# Companion plot: proportion over time, ONE PANEL PER COMPARTMENT, Burn vs Sham
# (per-sample mean ± SEM, Welch t-test Burn vs Sham per timepobint)
# Same styling as proportions_lines_split_by_condition_stats.
# ══════════════════════════════════════════════════════════════════════════════
from scipy.stats import ttest_ind
from matplotlib.ticker import PercentFormatter
import re, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

ID_COL = "mac_identity"

# Build raw -> display map from the labels that ACTUALLY exist
# (robust to both 'Recruited Macrophages' and 'MΦ-Recruited' naming).
_have = sorted(adata_mac.obs[ID_COL].astype(str).unique())
_find = lambda *keys: next((h for h in _have
                            if any(k in h.lower() for k in keys)), None)
_raw = {                                   # display label -> matched raw value
    "Inflammatory Monocytes": _find("inflamm", "mono"),
    "MΦ-Recruited":           _find("recruit"),
    "MΦ-Resident/Repair":     _find("resident", "repair", "res/rep"),
}
RENAME = {raw: disp for disp, raw in _raw.items() if raw is not None}  # raw -> display
labels = [disp for disp, raw in _raw.items() if raw is not None]       # display order
print("raw mac_identity values:", _have)
print("mapping (raw -> display):", RENAME)
assert RENAME, "No mac_identity values matched — check the printed raw values above."

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

# ── per-sample proportions ────────────────────────────────────────────────────
ident = adata_mac.obs[ID_COL].astype(str).map(RENAME)
work  = pd.DataFrame({"sample": adata_mac.obs[sample_col].astype(str).values,
                      "Type":   adata_mac.obs["Type"].astype(str).values,
                      "Timepoint": tp_norm.values,
                      "identity": ident.values})
work   = work[work["identity"].isin(labels)]
counts = work.groupby(["sample", "Type", "Timepoint", "identity"]).size().unstack("identity", fill_value=0)
for lab in labels:
    if lab not in counts.columns: counts[lab] = 0
props  = counts.div(counts.sum(1), axis=0).reset_index()

tp_order = sorted(props["Timepoint"].unique(), key=lambda t: int(re.search(r"\d+", t).group()))
xpos     = {t: i for i, t in enumerate(tp_order)}

# ── stats (Burn vs Sham per identity/timepoint) ───────────────────────────────
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

# ── plot: rows = compartments, two lines (Burn vs Sham) ───────────────────────
TYPE_PAL = {"Sham": "#2471A3", "Burn": "#C0392B"}
mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

fig, axes = plt.subplots(len(labels), 1, figsize=(6, 11), sharex=True)
for ax, lab in zip(axes, labels):
    ymax = 0.0
    for ty in ["Sham", "Burn"]:
        s_    = (summ[(summ.identity == lab) & (summ.Type == ty)]
                 .set_index("Timepoint").reindex(tp_order))
        means = s_["mean"].values
        sems  = np.nan_to_num(s_["sem"].values)
        x     = [xpos[t] for t in tp_order]
        ax.errorbar(x, means, yerr=sems, fmt="-o", color=TYPE_PAL[ty], lw=3,
                    ms=10, markeredgecolor="white", markeredgewidth=1.0,
                    capsize=5, capthick=2, elinewidth=2, label=ty, zorder=3)
        ymax = np.nanmax([ymax, np.nanmax(np.nan_to_num(means) + sems)])

    # significance stars above the higher of the two lines at each timepoint
    for k, tp in enumerate(tp_order):
        txt = stars(sig.get((lab, tp), np.nan))
        if not txt: continue
        tops = []
        for ty in ["Sham", "Burn"]:
            r = summ[(summ.identity == lab) & (summ.Type == ty) & (summ.Timepoint == tp)]
            if len(r):
                tops.append(np.nan_to_num(r["mean"].values[0]) + np.nan_to_num(r["sem"].values[0]))
        if tops:
            ax.annotate(txt, (xpos[tp], max(tops)), xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom",
                        color="black", fontsize=18, fontweight="bold")

    ax.set_title(lab, fontsize=22, fontweight="bold", pad=10)
    ax.set_ylim(0, ymax * 1.30 if ymax > 0 else 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Proportion\nof cells", fontsize=20, fontweight="bold", labelpad=8)
    ax.tick_params(axis="y", labelsize=18, width=1.5, length=6)
    ax.grid(axis="y", alpha=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

axes[-1].set_xticks(range(len(tp_order)))
axes[-1].set_xticklabels(tp_order, fontsize=20, fontweight="bold")
axes[-1].set_xlabel("Timepoint", fontsize=22, fontweight="bold", labelpad=8)
axes[0].legend(title="Condition", fontsize=14, title_fontsize=15, frameon=False,
               loc="center left", bbox_to_anchor=(1.02, 0.5))

fig.subplots_adjust(left=0.20, right=0.78, top=0.95, bottom=0.07, hspace=0.32)
fig.savefig(FIGDIR_MAC / "proportions_lines_split_by_compartment_burn_vs_sham.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR_MAC / "proportions_lines_split_by_compartment_burn_vs_sham.pdf", bbox_inches="tight")
plt.show()

