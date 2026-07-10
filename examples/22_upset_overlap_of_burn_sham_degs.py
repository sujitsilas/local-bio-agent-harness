"""UpSet: overlap of Burn↑ / Sham↑ DEGs across mac identities (all timepoints poole

Source: macrophages_resident_recruited.ipynb
Libraries: upsetplot
Key calls: def collect_degs, def make_upset, plt.close, plt.figure, plt.show
"""

# ══════════════════════════════════════════════════════════════════════════════
# UpSet: overlap of Burn↑ / Sham↑ DEGs across mac identities (all timepoints pooled)
# Requires de_id_tp from the per-identity DE cell.
# ══════════════════════════════════════════════════════════════════════════════
from upsetplot import from_contents, UpSet   # pip install upsetplot

# raw identity -> display label
DISP = {
    "Inflammatory Monocytes": "Inflammatory Monocytes",
    "Recruited Macrophages":  "MΦ-Recruited",
    "Resident Macrophages":   "MΦ-Resident/Repair",
}

def collect_degs(direction):
    """Union of significant DEGs per identity, across all timepoints (excl. Ambiguous/Low)."""
    pools = {}
    for ident in IDENTITIES:
        if re.search(r'ambig|low', ident, re.I):      # <- skip Ambiguous/Low
            continue
        genes = set()
        for tp in ALL_TIMEPOINTS:
            df = de_id_tp[ident].get(tp)
            if df is None:
                continue
            sub = (df[(df.padj < FDR_THRESH) & (df.lfc >  LFC_THRESH)] if direction == "Burn"
                   else df[(df.padj < FDR_THRESH) & (df.lfc < -LFC_THRESH)])
            genes |= set(sub["gene"])
        pools[DISP.get(ident, ident)] = genes
    return pools


def make_upset(direction, color):
    pools = {k: v for k, v in collect_degs(direction).items() if len(v) > 0}
    print(f"{direction}↑ DEG pool sizes:", {k: len(v) for k, v in pools.items()})
    if len(pools) < 2:
        print(f"  <2 non-empty pools for {direction} — skipping UpSet."); return

    # membership table (which genes fall in which intersection) -> CSV
    allg = sorted(set().union(*pools.values()))
    mt = pd.DataFrame({"gene": allg})
    for k, v in pools.items():
        mt[k] = mt["gene"].isin(v)
    mt.to_csv(FIGDIR_MAC / f"upset_{direction.lower()}_deg_membership.csv", index=False)

    data = from_contents(pools)
    fig = plt.figure(figsize=(15, 10))
    UpSet(data, subset_size="count", show_counts=True, sort_by="cardinality",
          facecolor=color, totals_plot_elements=0, element_size=None).plot(fig=fig)

    fig.suptitle(f"{direction}↑ DEGs shared across macrophage identities\n(all timepoints pooled)",
                 fontsize=16, fontweight="bold")

    fig.savefig(FIGDIR_MAC / f"upset_{direction.lower()}_degs_mac_identity.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR_MAC / f"upset_{direction.lower()}_degs_mac_identity.png", dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)
    print(f"Saved UpSet: {direction}")

make_upset("Burn", BURN_COL)
make_upset("Sham", SHAM_COL)

