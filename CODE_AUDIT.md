# Code audit: what was used in the final thesis

This audit covers the code under `E:\ZHY\code\AssociativeModel\Fitting` and the result-figure scripts under the archived LaTeX project. Status meanings:

- **Core** — needed to reproduce a main result, table or figure in the final thesis.
- **Core / intermediate** — produces an input table used by a core analysis or figure.
- **Figure** — generates a final thesis figure; data may come from the core analyses.
- **Optional** — useful for checking or formatting, but not required for the main results.
- **Legacy / exploratory** — earlier version, test, or abandoned branch; not used in the final thesis.

## 1. Main model, data and post-hoc analysis

| File | Role | Status | Notes |
|---|---|---|---|
| `DistributionOverlapDecision.py` | V13 model: RW/KRW × non-switch/switch fitting, simulation, repeated held-out evaluation and 5-fold CV | **Core** | Main computational model; outputs `SingleConc_resultsV13_CV_features/` |
| `SaveFeatures.py` | Extracts per-worm features from raw tracking tables | **Core** | Produces `PER_WORM_FEATURE_TABLE.csv`-type inputs |
| `AnalyzeFitResults_V13.py` | Post-hoc feature distributions, group-difference capture, loss/feature summaries | **Core** | Produces the capture tables used in Figs. 3.9–3.10 |
| `CompareGroup.ipynb` | Group-level descriptive statistics and feature figures | **Core** | Notebook; outputs stripped in this repo |
| `SingleConc.py` | Earlier fitting script (pre-V13) | Legacy | Superseded by `DistributionOverlapDecision.py` |
| `Fittry8merge.py` | Earlier merged-condition fitting script | Legacy | Not used in final thesis |

## 2. HMM and direct validation

| File | Role | Status | Notes |
|---|---|---|---|
| `hmm_pr_analysis.py` | Two-state HMM on q90 high-angular-velocity frames; Baum–Welch and Viterbi; PR fraction/rate/duration; Fig. 3.13 | **Core** | Emissions fixed at 0.05/0.70; 20 training trajectories per condition |
| `hmm_cache_sequences.py` | Caches HMM observation sequences | **Core / intermediate** | Needed by downstream HMM analyses |
| `xy_cache_sequences.py` | Caches x–y trajectories at 0.25 s | **Core / intermediate** | Needed by AR1/kinematics analyses |
| `regression_cache.py` | Caches transition-regression tables | **Core / intermediate** | Needed by `transition_regression.py` |
| `kinematic_utils.py` | x–y kinematics, ACF and speed-profile helpers | **Core** | Shared by temporal/state analyses |
| `ar1_fit_grid.py` | Fits KRW-AR1 `rho`/`sigma` on a small grid | **Core** | Grid `rho∈{0.90,0.95,0.98}`, `sigma∈{1.5,2.5}` |
| `temporal_control_analysis.py` | Compares real HMM, KRW-switch and KRW-AR1 on PR statistics, transition probabilities and ACF | **Core** | Produces Table 3.5 and direct-validation figures |
| `state_conditional_kinematics.py` | PR/non-PR `|omega|` and speed comparison | **Core** | Produces Fig. 3.14 source data |
| `transition_regression.py` | Entry/exit probability regression on CTX and speed | **Core** | Produces Fig. 3.15 source data |
| `state_calibration.py` | Post-hoc calibration of `min_pir_steps` and `a0` | **Core** | Produces the calibration numbers quoted in §4.3 |
| `high_salt_summary.py` | High-salt CTX descriptive statistics | **Core** | Produces Table 3.6 |
| `pr_real_proxy_analysis.py` | Real high-angle-turn proxy statistics | **Core / intermediate** | Produces `real_pr_proxy_summary.csv` used by value-state plotting |
| `hmm_postprocess.py` | Earlier HMM summary/plot script | Legacy | Superseded by `hmm_pr_analysis.py` |
| `hmm_explore.py`, `hmm_binary_test.py`, `hmm_diag.py` | HMM exploration/diagnostics | Exploratory | Not needed for final results |
| `ar1_smoke_test.py`, `ar1_pr_test.py` | AR1 sanity checks | Exploratory | Not needed for final results |
| `real_xy_kinematics_test.py`, `xy_kinematics_test.py` | Kinematics sanity checks | Exploratory | Not needed for final results |
| `nonswitch_check.py`, `nonswitch_direct_stats.py`, `switch_direct_stats.py` | Early non-switch/switch direct-stat checks | Exploratory | Final numbers come from `temporal_control_analysis.py` |
| `duration_calibration_test.py` | Early duration-calibration test | Legacy | Superseded by `state_calibration.py` |

## 3. Value-term ablation and state calibration

| File | Role | Status | Notes |
|---|---|---|---|
| `value_state_link_analysis.py` | Final-parameter counterfactual ablation: full / no-interaction / no-value | **Core** | 60 trajectories per condition/model; seed 20261116 |
| `fold_ablation_analysis.py` | Same ablation on 5-fold KRW-switch parameters | **Core** | 20 trajectories per fold; produces Fig. 3.11 |
| `plot_value_state_figures.py` | Plots value-state ablation and HMM-versus-model summaries | **Figure** | Produces Fig. 3.12 |
| `replot_state_conditional.py` | Replots state-conditioned `|omega|` with Chinese labels | **Figure** | Produces Fig. 3.14 |
| `make_ctx_transition_materials.py` | Formats CTX/speed transition regression and plots Fig. 3.15 | **Figure** | Uses `ctx_transition_test.csv` |
| `ctx_transition_test.py` | Fits observable CTX/speed transition regressions | **Core / intermediate** | Produces `ctx_transition_test.csv` |
| `make_review_materials.py` | Formats direct-validation tables | Optional | Table drafts |
| `make_calibration_materials.py` | Formats calibration tables | Optional | Calibration is quoted in text, not a final table |
| `replot_calibration.py` | Plots calibration comparison | Optional | Not a final thesis figure |

## 4. Result and schematic figure scripts (archived)

| File | Role | Status | Notes |
|---|---|---|---|
| `make_results_figs_v13_v2.py` | Figs. 3.1–3.6, 3.9, 3.10: loss comparison, trajectory comparison, feature capture, CTX distribution, loss–capture map | **Figure** | Uses V13 result CSVs |
| `make_ch3_expansion_figs.py` | Fold loss, loss decomposition, Cliffs delta, p_high scan, net displacement | **Figure** | Some functions also produce unused V15 bootstrap/overfit figures |
| `make_control_figs_v15.py` | V15 control-model figures | Not used | Control-model section was removed from the final thesis |
| `make_conclusion_fig.py`, `build_conclusion_revised.py`, `make_roadmap_cn.py` | Conclusion framework and roadmap figures | Not used | Final Chapter 4 uses the circuit hypothesis figure instead |
| `make_circuit_fig.py` | Earlier two-panel circuit schematic | Not used | Final circuit figure was redrawn separately |
| `fix_8h_labels.py`, `restructure_ch2_v2.py` | Figure label/panel assembly for Chapter 2 | Optional | Used for the 1g/5g and trajectory panel labels |

## 5. Bottom line

The minimal set needed for the final thesis is:

- **Main model:** `DistributionOverlapDecision.py`, `SaveFeatures.py`, `AnalyzeFitResults_V13.py`, `CompareGroup.ipynb`.
- **Direct validation:** `hmm_pr_analysis.py`, `hmm_cache_sequences.py`, `xy_cache_sequences.py`, `regression_cache.py`, `kinematic_utils.py`, `ar1_fit_grid.py`, `temporal_control_analysis.py`, `state_conditional_kinematics.py`, `transition_regression.py`, `state_calibration.py`, `high_salt_summary.py`.
- **Value ablation:** `value_state_link_analysis.py`, `fold_ablation_analysis.py`, `pr_real_proxy_analysis.py`, `plot_value_state_figures.py`.
- **Figure/table formatting:** `replot_state_conditional.py`, `ctx_transition_test.py`, `make_ctx_transition_materials.py`, `make_results_figs_v13_v2.py`, `make_ch3_expansion_figs.py`.

Everything else is legacy, exploratory, or was used only in the removed control-model branch.
