"""D-left COMPANION: same Type_Timepoint grid, colored by ORIGIN IDENTITY

Source: macrophages_resident_recruited.ipynb
Libraries: pandas
Key calls: plt.show, plt.subplots, plt.tight_layout, sc.pl.umap
"""

# ---------- D-left COMPANION: same Type_Timepoint grid, colored by ORIGIN IDENTITY ----------
import pandas as pd, numpy as np

ID_COL = "mac_identity"
have = list(pd.unique(adata_mac.obs[ID_COL].astype(str)))
find = lambda k: next((h for h in have if k in h.lower()), None)
id_order = [x for x in [find("inflamm"), find("recruit"), find("resident")] if x]   # Inf, Recruited, Resident
id_palette = dict(zip(id_order, ["#D62728", "#F39C12", "#2CA02C"]))                  # red, orange, green
print("identities:", id_order)

# categorical (origin first, any others appended) + color list scanpy can use
adata_mac.obs[ID_COL] = pd.Categorical(
    adata_mac.obs[ID_COL].astype(str),
    categories=id_order + [c for c in have if c not in id_order])
id_cats = list(adata_mac.obs[ID_COL].cat.categories)
id_palette_list = [id_palette.get(c, "#DDDDDD") for c in id_cats]
adata_mac.uns[f"{ID_COL}_colors"] = id_palette_list

# layout (recomputed so this cell stands alone)
tt_levels = [t for t in ["Sham D7","Burn D7","Sham D10","Burn D10","Sham D14","Burn D14","Sham D19","Burn D19"]
             if t in adata_mac.obs["Type_Timepoint_C"].cat.categories]
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
        sc.pl.umap(sub, color=ID_COL, palette=id_palette_list, ax=ax, show=False,
                   frameon=True, size=60, legend_loc=None, title="")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        style_umap_axes(ax,
                        xlabel="UMAP 1" if r == 1 else "",
                        ylabel="UMAP 2" if c == 0 else "",
                        title=f"{tt}  (n = {len(sub):,})")
        ax.title.set_fontsize(19); ax.title.set_fontweight("bold")
        ax.grid(False)

handles = [mpl.patches.Patch(facecolor=id_palette[c], label=c) for c in id_order]
leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
                 fontsize=(PUB["legend_fs"] if "PUB" in globals() else 14))
for t in leg.get_texts(): t.set_fontweight("bold")

plt.tight_layout()
plt.show()

