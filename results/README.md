# Results

All outputs behind the revised manuscript and the response to the reviewers. The main [README](../README.md) summarizes them; this file lists what each file contains.

## `full_comparison_v1/` — GraphTopic vs. nine baselines on seven datasets

Produced by `experiments/full_dataset_comparison_v1.py`.

| File | Content |
|---|---|
| `global_comparison_summary.csv` | **Main table.** One row per model × dataset (70 rows): mean of every metric over the chunks, `*_std` (standard deviation over chunks), `*_weighted` (weighted by documents), `num_topics`, `runtime_sec`, `peak_memory_mb`, `n_good_chunks`, and, for GraphTopic, the selected `TopM`/`TopN`/`overlapping_threshold` per chunk. |
| `<dataset>_full_comparison.csv` | Per-chunk rows (`chunk_idx` = 0, 1, …) and the `AVERAGE` row for one dataset. |
| `<dataset>_raw_results.pkl` | Raw per-chunk results, including the topics; this is the cache the script reads on restart. |
| `<dataset>_comparison_state.json` | Exact configuration of the run: models, chunk size, parameter grid, Optuna settings, and the hash used for caching. |
| `spider_<dataset>.png`, `spider_global.png` | Polar charts (min–max normalized per dataset); `global` shows all datasets plus their mean. |
| `bars_<dataset>.png`, `bars_global.png` | Bar charts of the raw scores (± standard deviation over chunks). |
| `hyperparameter_optimization/<dataset>/chunk_<k>_<hash>/trials.csv` | Every Optuna trial of GraphTopic on that chunk: `TopM`, `TopN`, `overlapping_threshold`, all seven metrics, `num_topics`, `status`, `runtime_sec`, and the tuning objective (`multi_objective`). |
| `hyperparameter_optimization/.../optuna.db` | The full Optuna study (SQLite); open it with `optuna.load_study(storage="sqlite:///optuna.db", study_name=...)`. |

Dataset names in the file names: `SetFit_20_newsgroups`, `SetFit_bbc-news`, `TrumpTweet`, `SetFit_imdb`, `ronitHF_pubmed-10k`, `TopicNet_RTL-Wiki`, `TopicNet_NIPS`.

The figures produced by the script label the proposed model "GraphTopic (Ours)".

## `parameter_effects/` — what the three hyperparameters do

This is an analysis of the 592 completed Optuna trials stored above; no model was re-run for it.

| File | Content |
|---|---|
| `cost_discussion_v1.md` | Short write-up: what was measured, caveats, the effects on score and on cost, and runtime vs. baselines. |
| `robust_defaults_table.csv` / `.tex` | Per hyperparameter: robust default (weighted mean of the top-20% trials), median best value, IQR, Spearman ρ and p, and the CV of the best value over the 30 chunks. |
| `robust_defaults_summary.csv` | Same values as `robust_defaults_table.csv`. |
| `tuning_cost_per_chunk.csv` | Number of trials and summed trial runtime per dataset × chunk. |
| `runtime_comparison_v1.csv` | Mean runtime per model × dataset and the ratio to GraphTopic. |
| `figures/fig1_sensitivity` | Normalized score vs. TopM / TopN / θ (mean ± s.d.). |
| `figures/fig2_effect_stability` | (a) Spearman importance; (b) CV of the best value per chunk. |
| `figures/fig3_pairwise_landscape` | Score landscape for each pair of hyperparameters (interpolated). |
| `figures/fig4_cost_by_param` | Trial runtime by value of each hyperparameter. |
| `figures/fig5_cost_comparison` | (a) Mean wall time per model; (b) score vs. runtime with the Pareto front. |
| `figures/figure_data_index.csv`, `README_figures.md` | Index of the figures. |
| `supplementary/supp_*` | Violin plots, binned scores, heatmap (dataset × chunk), effects by chunk and by dataset, top-20% trials, per-dataset contours, 3-D surface. |

Each figure comes as `.pdf` (vector), `.png` (600 dpi), and `.csv` (the plotted values).

## `scalability/` — runtime vs. corpus size

There are 10 models × 3 datasets (BBC, 20NG, Trump) × 10 sizes (10%–100%) = 300 runs.

| File | Content |
|---|---|
| `scalability_runs.csv` | One row per run: `fraction`, `n_docs`, `n_words`, `prep_sec`, `fit_sec`, `status`, the metrics, and the model `settings`. |
| `growth_report_by_dataset.csv` | Per model × dataset: start/end documents and runtime, runtime/data/normalized index at 100%, scaling exponent `scaling_ratio` (= s), s for the first and second halves, `runtime_growth_per_doubling_pct` (= g), and the verdict. |
| `growth_report_by_model.csv` | Per model, averaged over the three datasets. |
| `growth_index_series.csv` | Runtime index, data index, and normalized index at every size (the data behind the plots). |
| `runtime_vs_docs_all_models.png` | Runtime vs. number of documents. |
| `normalized_index_all_models.png` | Normalized index (100 = linear growth). |
| `growth_index_all_models.png` | Runtime index vs. data index, with linear and quadratic references. |
| `growth_per_doubling_all_models.png` | Growth per doubling of the data. |

Definitions: $s = \ln(T_{end}/T_{start}) / \ln(N_{end}/N_{start})$; $g = (2^s - 1)\times100\%$; normalized index $= 100\times$ runtime index / data index.

## `projection_check/` — is the 1-D overlap ω faithful?

Produced by `experiments/projection_check.py` on 20NG, BBC, and Trump.

| File | Content |
|---|---|
| `summary.json` | Per dataset and overall: number of pairs, Spearman/Pearson correlations of ω with `ovl_3d`, `cos_centroid_768d`, and `knn_mix_768d`, the agreement of the merge decisions, Cohen's κ, and the number of merged pairs. |
| `pairs.csv` | Every intercepted subtopic pair with ω and the three reference measures. |
| `omega_vs_full_space.(pdf|png)` | Scatter plots of ω against each reference measure. |
| `projection_example.(pdf|png)` | Worked example of a merged and a non-merged pair. |
