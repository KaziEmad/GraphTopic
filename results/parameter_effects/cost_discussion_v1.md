# Computational cost and hyperparameter effects (v1)
Source: existing runs under `full_comparison_v1` (no models were re-run for this note).
## What is measured
- **Wall time** (`runtime_sec`): stopwatch around each model call **after** `prepare_v1_docs`, including topic extraction **and** v1 coherence / semantic scoring.
- **Peak memory** (`peak_memory_mb`): RSS delta of the process; reported for completeness but **not** used in conclusions (shared machine, GC noise).
- Optuna trial times come from `trials.csv`; baseline times from `*_full_comparison.csv`.
## Caveats
- Single shared machine; GPU used by TopMost neural baselines (ProdLDA, HyperMiner, …) where available.
- One timed run per chunk (no multi-seed timing study).
- GraphTopic Optuna search cost is **extra** to the final reported GraphTopic runtime (final row uses the best params found).
## Hyperparameter effects on performance
- **overlapping_threshold**: Spearman ρ = -0.201 (p = 8.49e-07) vs normalised multi-objective score.
- **TopN**: Spearman ρ = -0.108 (p = 0.00853) vs normalised multi-objective score.
- **TopM**: Spearman ρ = -0.083 (p = 0.0432) vs normalised multi-objective score.
- Recommended default (weighted top-20% mean) **TopM ≈ 982**.
- Recommended default (weighted top-20% mean) **TopN ≈ 14.58**.
- Recommended default (weighted top-20% mean) **overlapping_threshold ≈ 0.4952**.
## Hyperparameter effects on cost
See `fig4_cost_by_param.pdf / .png` (runtime by TopM / TopN / threshold) and `fig5_cost_comparison.pdf / .png (panel b)` (quality–cost trade-off with Pareto front).
- Mean Optuna compute per chunk (sum of trial `runtime_sec`): **11390.0 s** (median 12044.3 s) across 30 groups.
## Wall-time comparison vs baselines
- GraphTopic mean wall time per chunk (across datasets): **545.8 s**.
- Fastest model (mean): **NMF** at 10.3 s (dataset TrumpTweet).
- Slowest model (mean): **ECRTM** at 7960.8 s (dataset SetFit/imdb).

Overall mean runtime per model (averaged across datasets):

| Model | Mean runtime_sec |
|---|---:|
| ECRTM | 1856.6 |
| GraphTopic | 545.8 |
| HyperMiner | 146.4 |
| BERTopic | 78.9 |
| FASTopic | 73.4 |
| LDA | 69.3 |
| ProdLDA | 61.8 |
| HDP | 57.5 |
| NMF | 57.3 |
| LSI | 57.1 |

Ratios vs GraphTopic are in `runtime_comparison_v1.csv` (`runtime_ratio_vs_graphtopic`).

## Practicality takeaway
GraphTopic’s wall time is dominated by KeyBERT keyphrase extraction and embedding-based topic merging, plus v1 coherence scoring of bigram/trigram topics. Relative to classical baselines (LDA/NMF/LSI) it is typically slower per chunk; relative to neural baselines the gap depends on GPU availability. The Optuna search over {TopM, TopN, θ} adds a one-time tuning cost (see `tuning_cost_per_chunk.csv`); after defaults are fixed, only a single GraphTopic fit is needed.
