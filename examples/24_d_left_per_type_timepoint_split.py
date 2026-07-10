"""D-left: per-Type_Timepoint split UMAPs (publication)

Source: macrophages_resident_recruited.ipynb
Libraries: pandas
Key calls: plt.show, plt.subplots, plt.tight_layout, sc.pl.umap
"""

# ---------- D-left: per-Type_Timepoint split UMAPs (publication) ----------
import pandas as pd

# Type_Timepoint ordering
adata_mac.obs["Type_Timepoint_C"] = adata_mac.obs["Type_Timepoint_C"].astype("category")
tt_levels = ["Sham D7","Burn D7","Sham D10","Burn D10","Sham D14","Burn D14","Sham D19","Burn D19"]
tt_levels = [t for t in tt_levels if t in adata_mac.obs["Type_Timepoint_C"].cat.categories]
adata_mac.obs["Type_Timepoint_C"] = adata_mac.obs["Type_Timepoint_C"].cat.set_categories(tt_levels)

# ── make mac_colors apply reliably: per-category color LIST in category order ──
if not pd.api.types.is_categorical_dtype(adata_mac.obs["macrophage_subtypes"]):
    adata_mac.obs["macrophage_subtypes"] = adata_mac.obs["macrophage_subtypes"].astype("category")
cats = list(adata_mac.obs["macrophage_subtypes"].cat.categories)
miss = [c for c in cats if c not in mac_colors]
if miss:
    print("WARNING: no mac_colors entry for:", miss)
palette_list = [mac_colors.get(c, "#999999") for c in cats]
adata_mac.uns["macrophage_subtypes_colors"] = palette_list      # belt-and-suspenders

type_order = ["Sham", "Burn"]
tp_order   = sorted({tt.split()[1] for tt in tt_levels}, key=lambda t: int(re.sub(r"\D", "", t)))
grid = [[f"{ty} {tp}" if f"{ty} {tp}" in tt_levels else None for tp in tp_order] for ty in type_order]

um = adata_mac.obsm["X_umap"]
xpad = 0.04 * (um[:, 0].max() - um[:, 0].min()); ypad = 0.04 * (um[:, 1].max() - um[:, 1].min())
xlim = (um[:, 0].min() - xpad, um[:, 0].max() + xpad)
ylim = (um[:, 1].min() - ypad, um[:, 1].max() + ypad)

ncols = len(tp_order)
fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 9), squeeze=False)
for r, row in enumerate(grid):
    for c, tt in enumerate(row):
        ax = axes[r][c]
        if tt is None:
            ax.set_visible(False); continue
        sub = adata_mac[adata_mac.obs["Type_Timepoint_C"] == tt]
        sc.pl.umap(sub, color="macrophage_subtypes", palette=palette_list,   # <- list, not dict
                   ax=ax, show=False, frameon=True, size=60, legend_loc=None, title="")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        style_umap_axes(ax,
                        xlabel="UMAP 1" if r == 1 else "",
                        ylabel="UMAP 2" if c == 0 else "",
                        title=f"{tt}  (n = {len(sub):,})")
        ax.title.set_fontsize(19)
        ax.title.set_fontweight("bold")        # <- shrink just this panel title
        ax.grid(False)


# shared legend (use desired_order if defined, else category order)
try:
    legend_cats = [c for c in desired_order if c in cats]
except NameError:
    legend_cats = cats
handles = [mpl.patches.Patch(facecolor=mac_colors.get(c, "#999999"), label=c) for c in legend_cats]
leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                 frameon=False, fontsize=(PUB["legend_fs"] if "PUB" in globals() else 14))
for t in leg.get_texts(): t.set_fontweight("bold")

plt.tight_layout()
plt.show()

