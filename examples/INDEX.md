# Examples corpus — the user's code style

Grep this folder for a task (e.g. `grep -rl volcano examples/`), then read the
matching snippet and mirror its style. Extracted code only; no outputs.

| # | task | key calls | libraries |
| - | - | - | - |
| 01_dendrogram.py | dendrogram | adjust_text, def style_umap, dendrogram, plt.figure, plt.show, plt.subplots | adjustText, cellrank, matplotlib, numpy, os, pandas |
| 02_macrophage_origin_identities_de_volcano_go.py | MACROPHAGE ORIGIN IDENTITIES: DE + VOLCANO + GO — Burn vs Sham (timepoints poole | adjust_text, def _slug, def contains_excluded, def get_contamination_genes, def make_enrichment_tile, def pick_labels | adjustText, gseapy, matplotlib, numpy, pandas, pathlib |
| 03_cell_type_de_volcano_go_burn.py | CELL-TYPE DE + VOLCANO + GO — Burn vs Sham (timepoints pooled) | adjust_text, def _resolve_lineages, def get_contamination_genes_ct, def prepare_ct_for_DE, gp.enrichr, plt.close |  |
| 04_cell_type_composition_table_sham_vs.py | Cell-type composition table: Sham vs Burn proportions across timepoints | .plot, def msem, def render_prop_table, def stars, def tp_per_cell, def transform | matplotlib, numpy, pandas, pathlib, re, scanpy |
| 05_4_apply_mapping_order_categories_and.py | 4. Apply mapping, order categories, and assign colors to your AnnData object | .plot, adjust_text, def _lm, def _lm_contrasts, def _overall_p, def _per_tp_ttest | adjustText, math, matplotlib, numpy, pathlib, scanpy |
| 06_milo_style_differential_abundance_pure_python.py | Milo-style differential abundance (pure Python) — Burn vs Sham macrophages | def _style, plt.cm, plt.get_cmap, plt.show, plt.subplots, sc.pp.neighbors | matplotlib, numpy, pathlib, scanpy, scipy, statsmodels |
| 07_milo_neighborhood_da_graph_on_umap.py | Milo neighborhood DA graph on UMAP — node size = neighborhood size | .plot, def _ov, def cliffs, def expr_of, def shorten, plt.show | matplotlib, numpy, pandas, pathlib, scanpy, scipy |
| 08_pip_install_moscot.py | pip install moscot | adjust_text, cellrank.estimators, cellrank.kernels, plt.show, plt.subplots, umap | adjustText, cellrank, matplotlib, moscot, numpy, pathlib |
| 09_macrophage_origin_identities_de_volcano_go.py | MACROPHAGE ORIGIN IDENTITIES: DE + VOLCANO + GO — Burn vs Sham (timepoints poole | adjust_text, def _slug, def contains_excluded, def get_contamination_genes, def make_enrichment_tile, def pick_labels | adjustText, gseapy, matplotlib, numpy, pandas, pathlib |
| 10_per_mac_identity_timepoint_de_volcano.py | Per mac_identity × Timepoint:  DE + Volcano + GO  (Burn vs Sham) | adjust_text, def draw_volcano, def get_contamination_genes, gp.enrichr, plt.close, plt.show |  |
| 11_m_recruited_only_per_macrophage_subtypes.py | MΦ-Recruited only:  per macrophage_subtypes × Timepoint  DE + Volcano + GO | gp.enrichr, plt.close, plt.show, plt.subplots, rank_genes_groups, sc.pp.log1p |  |
| 12_m_recruited_only_per_macrophage_subtypes.py | MΦ-Recruited only:  per macrophage_subtypes × Timepoint  DE + Volcano + GO | gp.enrichr, plt.close, plt.show, plt.subplots, rank_genes_groups, sc.pp.log1p |  |
| 13_m_recruited_only_per_macrophage_subtypes.py | MΦ-Recruited only:  per macrophage_subtypes × Timepoint  DE + Volcano + GO | gp.enrichr, plt.close, plt.show, plt.subplots, rank_genes_groups, sc.pp.log1p |  |
| 14_consecutive_timepoint_de_later_vs_earlier.py | Consecutive-timepoint DE (LATER vs EARLIER), per identity × condition | adjust_text, def _run_tp, def _volcano_tp, gp.enrichr, plt.close, plt.show | adjustText, numpy |
| 15_consecutive_timepoint_de_later_vs_earlier.py | Consecutive-timepoint DE (LATER vs EARLIER), per identity × condition | adjust_text, def _run_tp, def _volcano_tp, gp.enrichr, plt.close, plt.show | adjustText, numpy |
| 16_consecutive_timepoint_de_later_vs_earlier.py | Consecutive-timepoint DE (LATER vs EARLIER), per identity × condition | adjust_text, def _run_tp, def _volcano_tp, gp.enrichr, plt.close, plt.show | adjustText, numpy |
| 17_four_way_plots_of_burn_vs.py | FOUR-WAY plots of Burn-vs-Sham DE across consecutive timepoints, per identity | .plot, adjust_text, def _de_bvs, def _fourway_tp, def _top, plt.close | adjustText, matplotlib, numpy, re, scipy |
| 18_curated_panel_from_the_four_way.py | Curated panel from the four-way temporal analysis → dotplot by Type_Timepoint_C | def _classify, dendrogram, dotplot, plt.savefig, plt.show, sc.pl.dotplot | numpy |
| 19_burn_vs_sham_deg_overlap_venn.py | Burn vs Sham DEG overlap (Venn) + GO enrichment of the condition-UNIQUE genes | def _barh, def _enrich, def _sig, def _venn2, gp.enrichr, plt.close | matplotlib, numpy |
| 20_burn_vs_sham_deg_overlap_venn.py | Burn vs Sham DEG overlap (Venn) + top-10 unique genes per side | def _sig, def _top_unique, def _venn2, plt.close, plt.show, plt.subplots | matplotlib, numpy |
| 21_burn_vs_sham_deg_overlap_venn.py | Burn vs Sham DEG overlap (Venn) + GO enrichment of the condition-UNIQUE genes | def _barh, def _enrich, def _sig, def _venn2, gp.enrichr, plt.close | matplotlib, numpy, re |
| 22_upset_overlap_of_burn_sham_degs.py | UpSet: overlap of Burn↑ / Sham↑ DEGs across mac identities (all timepoints poole | def collect_degs, def make_upset, plt.close, plt.figure, plt.show | upsetplot |
| 23_per_upset_bin_enrichment_highlight_genes.py | Per-UpSet-bin enrichment + highlight genes (Burn↑ / Sham↑) | def bin_label, def bins_enrichment, def deg_long, def make_enrichment_tile, gp.enrichr, plt.close |  |
| 24_d_left_per_type_timepoint_split.py | D-left: per-Type_Timepoint split UMAPs (publication) | plt.show, plt.subplots, plt.tight_layout, sc.pl.umap | pandas |
| 25_d_left_companion_same_type_timepoint.py | D-left COMPANION: same Type_Timepoint grid, colored by ORIGIN IDENTITY | plt.show, plt.subplots, plt.tight_layout, sc.pl.umap | pandas |
| 26_per_mac_identity_2_type_timepoint.py | per mac_identity: 2 (Type) × timepoint split UMAPs, colored by subtype | .plot, barplot, def conf_ellipse, def style_ax, def tp_per_cell, gp.get_library | matplotlib, numpy, pandas, re, scipy |
| 27_1_define_your_original_color_palette.py | 1. Define your original color palette | .plot, def add_ct_labels, def conf_ellipse, def score_set, def stars, def style_ax | matplotlib, numpy, re, scanpy, scipy, statsmodels |
| 28_figure.py | figure | def add_ct_labels, def conf_ellipsoid, plt.figure, plt.savefig, plt.show, plt.subplots | gseapy, matplotlib, mpl_toolkits, numpy, re, scipy |
| 29_unresolved_inflammation_mac_identity_cells_scored.py | UNRESOLVED INFLAMMATION — mac_identity cells scored for Inflammation vs Resoluti | def ks_imbalance, def stars, def style_ax, gp.get_library, sc.tl.score_genes | matplotlib, numpy, re, scanpy, scipy, seaborn |
| 30_figure_a_one_panel_per_mac.py | Figure A: one panel per mac_identity, Inflammation vs Resolution | plt.show, plt.subplots, scatter, sns.kdeplot |  |
| 31_figure_b_grid_rows_mac_identity.py | Figure B-grid: rows = mac_identity, cols = timepoint | plt.show, plt.subplots, scatter, sns.kdeplot |  |
| 32_inflammation_trajectory_summarized_by_sample_pseudobulk.py | Inflammation trajectory summarized by SAMPLE (pseudobulk points + error bars) | def _overall_p, def _tt, plt.show, plt.subplots, scatter | scipy, statsmodels |
| 33_inflammation_trajectory_points_bars_by_sample.py | Inflammation trajectory — points/bars by SAMPLE, stats by mixed model | def _mixed, def _star, plt.show, plt.subplots, scatter | scipy, statsmodels, warnings |
| 34_concise_dot_plot_inflammatory_m1_program.py | Concise dot plot: inflammatory / M1 program across the three identities | dotplot, plt.rc_context, plt.setp, plt.show, sc.pl.dotplot |  |
| 35_1_define_your_original_color_palette.py | 1. Define your original color palette | adjust_text, def add_ct_labels, def tp_per_cell, dotplot, plt.gcf, plt.show | adjustText, matplotlib, numpy, os, pandas, re |
| 36_keystone_metabolic_ratio_glyco_oxphos_vs.py | KEYSTONE: Metabolic_Ratio (Glyco−OXPHOS)  vs  Recruited→Resident axis, | gp.get_library, plt.show, plt.subplots, sc.tl.score_genes, scatter, sns.kdeplot | gseapy, matplotlib, re, scipy, statsmodels |
| 37_keystone_timepoint_resolved_metabolic_ratio_vs.py | KEYSTONE (timepoint-resolved): Metabolic_Ratio vs Recruited→Resident axis | plt.show, plt.subplots, sc.tl.score_genes, scatter, sns.kdeplot | matplotlib, re, scipy |
| 38_keystone_timepoint_resolved_metabolic_ratio_vs.py | KEYSTONE (timepoint-resolved): Metabolic_Ratio vs Recruited→Resident axis | plt.show, plt.subplots, sc.tl.score_genes, scatter, sns.kdeplot | matplotlib, re, scipy |
| 39_companion_plot_proportion_over_time_one.py | Companion plot: proportion over time, ONE PANEL PER COMPARTMENT, Burn vs Sham | def stars, def tp_per_cell, plt.show, plt.subplots | matplotlib, re, scipy |
| 40_macrophage_subtype_trajectory_py_monocle_sham.py | Macrophage subtype trajectory (py-monocle) — Sham vs Burn, rooted on Inf. Mono | .plot, def _chaikin, def _chains_from_edges, def _fit_cond, def _p, def _pick_inf_root | collections, matplotlib, numpy, pathlib, py_monocle, scanpy |
| 41_per_macrophage_subtypes_all_timepoints_pooled.py | Per macrophage_subtypes (all timepoints pooled):  DE + Volcano + GO (Burn vs Sha | adjust_text, def draw_volcano, gp.enrichr, plt.close, plt.show, plt.subplots |  |
| 42_per_macrophage_subtypes_timepoint_de_volcano.py | Per macrophage_subtypes × Timepoint:  DE + Volcano + GO  (Burn vs Sham) | gp.enrichr, plt.close, plt.show, plt.subplots, rank_genes_groups, sc.tl.rank_genes_groups |  |
| 43_lightweight_cellchat_style_crosstalk_inf_mono.py | Lightweight CellChat-style crosstalk: Inf.Mono / Recruited / Resident macrophage | def _edge, def _selfloop, def comm_matrix, def draw_net, dendrogram, plt.rcParams | matplotlib, numpy, pathlib |
| 44_more_informative_crosstalk_views_reuse_scoring.py | More informative crosstalk views (reuse scoring from the CellChat-style cell) | def _st, def _vec, def coexp_stats, def render_table, heatmap, plt.Normalize | matplotlib, numpy, pathlib, scipy, statsmodels |
