# Hierarchical computational modelling of value representation and behavioural state transitions in *C. elegans* associative learning

This repository contains the core analysis code for the master's thesis:

> **线虫联合学习下的价值表征与行为状态转换的分层计算建模**
> Hierarchical Computational Modeling of Value Representation and Behavioral State Transitions in *Caenorhabditis elegans* during Associative Learning

The code compares four behavioural-generating models (RW / KRW × non-switch / switch) on salt-chemotaxis associative-learning trajectories, and adds direct validation of the switch state with an HMM, an AR(1) continuous control, and a state-statistics calibration analysis.

## Repository layout

```
code/
  core/
    DistributionOverlapDecision.py   # V13 model: fitting, simulation, 5-fold CV
    SaveFeatures.py                  # tracking-table feature extraction
    AnalyzeFitResults_V13.py         # post-hoc feature/capture analysis
  validation/
    hmm_pr_analysis.py               # two-state HMM for real PR-like states
    hmm_cache_sequences.py           # cache HMM observation sequences
    xy_cache_sequences.py            # cache x-y trajectories
    regression_cache.py              # cache transition-regression tables
    kinematic_utils.py               # kinematics, ACF and speed-profile helpers
    ar1_fit_grid.py                  # KRW-AR1 rho/sigma grid fit
    temporal_control_analysis.py     # real HMM vs KRW-switch vs KRW-AR1
    state_conditional_kinematics.py  # PR/non-PR |omega| and speed
    transition_regression.py         # entry/exit regression on CTX and speed
    state_calibration.py             # post-hoc PR-statistics calibration
    high_salt_summary.py             # high-salt CTX descriptive statistics
    value_state_link_analysis.py     # value-term ablation (final parameters)
    fold_ablation_analysis.py        # value-term ablation (5-fold parameters)
    pr_real_proxy_analysis.py        # real high-angle-turn proxy statistics
    ctx_transition_test.py           # observable CTX/speed transition regression
  figures/
    make_results_figs_v13_v2.py      # Figs. 3.1-3.6, 3.9, 3.10
    make_ch3_expansion_figs.py       # fold loss, loss decomposition, Cliffs delta, p_high scan
    plot_value_state_figures.py      # Fig. 3.12 (value-state ablation)
    replot_state_conditional.py      # Fig. 3.14 (state-conditioned |omega|)
    make_ctx_transition_materials.py # Fig. 3.15 and table drafts
  notebooks/
    CompareGroup.ipynb               # group-level statistics and feature figures
results/
  v13/                               # main V13 fit/CV/capture tables
  validation/                        # HMM, AR1, ablation, calibration summaries
data/
  README.md                          # expected data schema (raw data not included)
```

## What is not included

- Raw tracking tables and `condition_dict.pkl` (about 15 GB).
- The V15 control-model branch (SoftRW / Recent / PMatch), which was removed from the final thesis.
- Earlier fitting scripts (`SingleConc.py`, `Fittry8merge.py`) and exploratory HMM/AR1 tests.
- Schematic figures made in PowerPoint/Illustrator (task design, circuit hypothesis, pipeline and loss-computation diagrams).
- The full LaTeX source and thesis PDF.

`CODE_AUDIT.md` gives a file-by-file audit of what was used in the final thesis and what was legacy/exploratory.

## Data expected by the code

The fitting scripts expect a pickle file:

```
condition_dict.pkl
```

It is a Python dictionary:

```python
{
  "3g-1gNa-1h":  [df_0, df_1, ...],
  "3g-1gNa-4h":  [df_0, df_1, ...],
  "1gNa-end(8x)": [df_0, df_1, ...],
}
```

Each DataFrame contains at least:

```
X, Y, ID, Timestamp, speed, agl_velocity, CTX
```

`X`, `Y` are pixel coordinates; `ID` is the worm/trajectory identifier; `Timestamp` is in frames at 20 Hz; `speed` is linear speed; `agl_velocity` is angular velocity; `CTX` is the chemotaxis index. The raw data are available from the author on reasonable request.

## Path configuration

The scripts were written for the lab storage and contain absolute Windows paths at the top of each file. Before running, edit the constants listed in `PATHS.md`, typically:

- `ROOT` / `BASE` / `BASE_OUTPUT_DIR`
- `OUT` / `FIG` / `DATA`
- `condition_dict.pkl` path

The derived summary tables used to draw the thesis figures are included under `results/`, so the figure scripts can be checked without re-running the full fitting.

## Suggested reproduction order

1. `code/core/SaveFeatures.py` — build per-worm feature tables from raw tracking files.
2. `code/core/DistributionOverlapDecision.py` — fit the four main models and run within-condition 5-fold CV.
3. `code/core/AnalyzeFitResults_V13.py` — compute group-difference capture scores and post-hoc summaries.
4. `code/validation/hmm_cache_sequences.py`, `xy_cache_sequences.py`, `regression_cache.py` — build caches.
5. `code/validation/hmm_pr_analysis.py` — fit/decode the real HMM.
6. `code/validation/ar1_fit_grid.py` — fit KRW-AR1 `rho/sigma`.
7. `code/validation/temporal_control_analysis.py` — compare real HMM, KRW-switch and KRW-AR1.
8. `code/validation/state_conditional_kinematics.py`, `transition_regression.py` — state-conditioned kinematics and transition regression.
9. `code/validation/state_calibration.py` — post-hoc calibration sensitivity.
10. `code/validation/high_salt_summary.py` — high-salt descriptive statistics.
11. `code/validation/pr_real_proxy_analysis.py`, `value_state_link_analysis.py`, `fold_ablation_analysis.py` — value-term ablation.
12. `code/validation/ctx_transition_test.py` and the figure scripts under `code/figures/`.

The full fitting is expensive: the main model uses Bayesian optimisation with repeated trajectory simulations. For a quick check, use the included summary CSVs and the figure scripts.

## Environment

Python 3.10 is recommended. See `requirements.txt`.

## Contact

张含玥 (ZHANG Hanyue), Peking University / Chinese Institute for Brain Research, Beijing.

## License

No open-source license has been applied yet. Please contact the author before reusing the code or data.
