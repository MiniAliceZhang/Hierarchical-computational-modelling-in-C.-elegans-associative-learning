# Derived result tables

These CSV files are the compact summary tables used to check the main numbers and redraw the figures. They do not contain raw trajectories or per-frame data.

## `v13/`

| File | Content |
|---|---|
| `ALL_CONDITIONS_results.csv` | Final-fit parameters and losses for RW, RW-switch, KRW, KRW-switch |
| `CV_ALL_FOLDS_results.csv` | Per-fold held-out losses |
| `CV_FITTED_PARAMS.csv` | Parameters fitted in each CV fold |
| `CV_MODEL_SUMMARY.csv` | CV loss summaries |
| `CV_GROUP_PATTERN_SCORES.csv` | Group-pattern scores from CV |
| `MODEL_GROUP_CAPTURE_SUMMARY.csv` | Overall group-difference capture score per model |
| `GROUP_DIFFERENCE_CAPTURE_BY_FEATURE.csv` | Capture score per feature |
| `PER_WORM_FEATURE_TABLE.csv` | Per-worm behavioural features |
| `PAIRWISE_GROUP_STATS.csv` | Pairwise condition statistics |

## `validation/`

| File | Content |
|---|---|
| `value_state_summary.csv` | Value-term ablation: full / no-interaction / no-value |
| `fold_ablation_summary.csv` | Fold-wise value-term ablation |
| `hmm_pr_summary.csv` | Real HMM PR fraction, event rate, duration, transition probabilities |
| `hmm_pr_sensitivity.csv` | q95 sensitivity analysis |
| `temporal_validation_summary.csv` | Real HMM vs KRW-switch vs KRW-AR1 |
| `ar1_best_params.csv` | Best `rho`/`sigma` for KRW-AR1 |
| `state_conditional_summary.csv` | PR/non-PR `|omega|` and speed |
| `ctx_transition_entry.csv` | CTX/speed coefficients for PR entry |
| `transition_regression.csv` | Full transition regression output |
| `calibration_comparison.csv` | Original vs calibrated PR statistics and behaviour |
| `calibration_best_per_condition.csv` | Best calibration parameters per condition |
| `high_salt_ctx_summary.csv` | High-salt CTX descriptive statistics |
