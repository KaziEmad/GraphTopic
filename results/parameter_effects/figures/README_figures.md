# Figure data index

Each figure has matching `.pdf` (vector, for the paper), `.png` (600 dpi), and `.csv` (exact plotted values for analysis).

Start from `figure_data_index.csv`.

| figure | section | description |
|---|---|---|
| `fig1_sensitivity` | main | Normalised score vs TopM / TopN / threshold (aggregate mean ± std) |
| `fig2_effect_stability` | main | (a) Spearman importance; (b) CV of best-per-group hyperparameter |
| `fig3_pairwise_landscape` | main | Aggregate contour of normalised score for each hyperparameter pair |
| `fig4_cost_by_param` | main | Wall-time distributions by TopM / TopN / threshold (Optuna trials) |
| `fig5_cost_comparison` | main | (a) Mean wall time per model; (b) Optuna score vs runtime with Pareto front |
| `supp_stability_violin` | supplementary | Distribution of best-per-group hyperparameter values by dataset |
| `supp_score_by_bin` | supplementary | Binned mean±std normalised score vs each hyperparameter |
| `supp_heatmap` | supplementary | Best hyperparameter value heatmap (dataset × chunk) |
| `supp_by_chunk` | supplementary | Sensitivity curves stratified by chunk index |
| `supp_by_dataset` | supplementary | Sensitivity curves stratified by dataset |
| `supp_top20_pairs` | supplementary | Top-20% trials in each hyperparameter pair plane |
| `supp_per_dataset_contour` | supplementary | Per-dataset contour for pair TopM×TopN (first 3 datasets) |
| `supp_3d_surface` | supplementary | 3D surface of normalised score for TopM×TopN |
