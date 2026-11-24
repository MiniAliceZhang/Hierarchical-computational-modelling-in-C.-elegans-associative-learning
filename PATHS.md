# Paths to edit before running

The research scripts use absolute Windows paths from the lab storage. They are kept unchanged for provenance. Before running on another machine, edit the constants below.

| File | Constant(s) | What it should point to |
|---|---|---|
| `code/core/DistributionOverlapDecision.py` | `BASE_OUTPUT_DIR` | Output folder for V13 fits/CV, e.g. `results/v13` |
| `code/core/DistributionOverlapDecision.py` | input pickle | `condition_dict.pkl` in the working directory (or edit the `open(...)` call) |
| `code/core/AnalyzeFitResults_V13.py` | `--results-dir`, `--output-dir` | V13 result folder and analysis output folder |
| `code/core/SaveFeatures.py` | `save_main_path`, raw-data paths | Raw tracking folder and feature-output folder |
| `code/validation/hmm_pr_analysis.py` | `ROOT`, `OUT` | Code root and output folder for HMM summaries |
| `code/validation/temporal_control_analysis.py` | `ROOT`, `REVIEW` | Code root and review-analysis output folder |
| `code/validation/state_conditional_kinematics.py` | `ROOT`, `OUT` | Code root and state-kinematics output folder |
| `code/validation/transition_regression.py` | `ROOT`, `OUT` | Code root and transition-regression output folder |
| `code/validation/state_calibration.py` | `ROOT`, `OUT` | Code root and calibration output folder |
| `code/validation/high_salt_summary.py` | `ROOT`, `OUT` | Code root and high-salt output folder |
| `code/validation/value_state_link_analysis.py` | `ROOT`, `OUT` | Code root and value-ablation output folder |
| `code/validation/fold_ablation_analysis.py` | `ROOT`, `OUT` | Code root and fold-ablation output folder |
| `code/validation/pr_real_proxy_analysis.py` | `ROOT`, `OUT` | Code root and real-proxy output folder |
| `code/validation/ctx_transition_test.py` | `ROOT` | Code root; reads V13 params and caches |
| `code/figures/plot_value_state_figures.py` | `OUT` | Folder containing `value_state_summary.csv` and `real_pr_proxy_summary.csv` |
| `code/figures/replot_state_conditional.py` | `OUT` | Folder containing `state_conditional_summary.csv` |
| `code/figures/make_ctx_transition_materials.py` | `ROOT`, `OUT` | Code root and transition-regression folder |
| `code/figures/make_results_figs_v13_v2.py` | `BASE`, `ANALYSIS`, `OUT` | V13 result folder, V13 analysis folder, figure output |
| `code/figures/make_ch3_expansion_figs.py` | `V13`, `V13A`, `V15`, `OUT` | V13/V15 result folders and figure output; the V15-dependent functions are not needed for the final thesis |

`results/` contains the derived summary tables used in the thesis, so most figure scripts can be tested by pointing their input constants at `results/v13` and `results/validation`.

## Minimal data layout

```
data/
  condition_dict.pkl          # not included; see data/README.md
results/
  v13/
    ALL_CONDITIONS_results.csv
    CV_ALL_FOLDS_results.csv
    CV_FITTED_PARAMS.csv
    CV_MODEL_SUMMARY.csv
    CV_GROUP_PATTERN_SCORES.csv
    MODEL_GROUP_CAPTURE_SUMMARY.csv
    GROUP_DIFFERENCE_CAPTURE_BY_FEATURE.csv
    PER_WORM_FEATURE_TABLE.csv
    PAIRWISE_GROUP_STATS.csv
  validation/
    value_state_summary.csv
    hmm_pr_summary.csv
    hmm_pr_sensitivity.csv
    fold_ablation_summary.csv
    temporal_validation_summary.csv
    ar1_best_params.csv
    state_conditional_summary.csv
    ctx_transition_entry.csv
    transition_regression.csv
    calibration_comparison.csv
    calibration_best_per_condition.csv
    high_salt_ctx_summary.csv
```
