# -*- coding: utf-8 -*-
"""
AnalyzeFitResults_V13.py

Post-hoc analysis script for SingleConc_V13_CV_features.py outputs.

Purpose
-------
This script compares fitted trajectories and behavioral feature distributions
against the original real trajectories. It is designed for the output structure
created by SingleConc_V13_CV_features.py:

    SingleConc_resultsV13_CV_features/
        final_fit/
            3g-1gNa-1h/
                3g-1gNa-1h_RW_trajectories.csv
                ...
            3g-1gNa-4h/
            1gNa-end(8x)/
        ALL_CONDITIONS_results.csv
        CV_MODEL_SUMMARY.csv                 # optional, if CV was run
        CV_GROUP_PATTERN_SCORES.csv          # optional, if CV was run

Main outputs
------------
The script creates an analysis folder, by default:

    SingleConc_resultsV13_analysis/

and saves:

1. Trajectory comparison figures:
    trajectories_5panel_<condition>.png

2. Across-condition group-difference violin plots:
    onefig_mean_CTX_5panels.png
    onefig_mean_abs_agl_velocity_5panels.png
    onefig_net_displacement_5panels.png
    onefig_mean_dist_to_midline_5panels.png

3. Per-condition real-vs-model violin plots:
    model_vs_real_<feature>.png

4. Distribution overlay plots:
    distribution_overlay_CTX.png
    distribution_overlay_mean_CTX.png
    distribution_overlay_net_displacement.png
    distribution_overlay_mean_dist_to_midline.png

5. Loss bar plots:
    final_loss_by_condition_bar.png
    final_loss_mean_bar.png
    cv_test_loss_by_condition_bar.png        # optional
    cv_test_loss_mean_bar.png                # optional

6. Statistics tables:
    PER_WORM_FEATURE_TABLE.csv
    FRAME_FEATURE_TABLE.csv
    PAIRWISE_GROUP_STATS.csv
    GROUP_DIFFERENCE_CAPTURE_BY_FEATURE.csv
    MODEL_GROUP_CAPTURE_SUMMARY.csv
    LOSS_RANKING_FINAL.csv
    LOSS_RANKING_CV.csv                      # optional
    analysis_summary.txt

Run
---
Put this script in the same directory as:
    condition_dict.pkl
    SingleConc_resultsV13_CV_features/

Then run:
    python AnalyzeFitResults_V13.py

Or specify paths:
    python AnalyzeFitResults_V13.py \
        --condition-pkl condition_dict.pkl \
        --results-dir SingleConc_resultsV13_CV_features \
        --output-dir SingleConc_resultsV13_analysis

Notes
-----
- Group-difference significance is assessed by Mann-Whitney U tests with Holm
  correction across the three condition pairs within each model and feature.
- The group-capture score is a post-hoc diagnostic, not a replacement for
  held-out loss. It combines direction matching, significance matching,
  z-scored condition-mean pattern similarity, and pairwise effect-size correlation.
- Cross-validation files are used if present. If not present, the script still
  analyzes final-fit trajectories and final-fit loss.
"""

import argparse
import glob
import math
import os
import pickle
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================
# CONFIG
# =========================
CONDITIONS = [
    "3g-1gNa-1h",
    "3g-1gNa-4h",
    "1gNa-end(8x)",
]

CONDITION_COLORS = {
    "3g-1gNa-1h": np.array([0.075353, 0.027206, 0.538007, 1.      ]), # 深蓝紫
    "3g-1gNa-4h": np.array([0.374897, 0.002245, 0.651876, 1.      ]), # 紫色
    "1gNa-end(8x)": np.array([0.615812, 0.094564, 0.61714 , 1.      ]), # 紫红
}

CONDITION_LABELS = {
    "3g-1gNa-1h": "3g-1gNa-1h",
    "3g-1gNa-4h": "3g-1gNa-4h",
    "1gNa-end(8x)": "1gNa-end(8x)",
}

MODELS = ["Real Data", "RW", "RW_switch", "KRW", "KRW_switch"]
SIM_MODELS = ["RW", "RW_switch", "KRW", "KRW_switch"]

def map_color_dict(key_items, color_map='Pastel1'):
    '''
    为给定的项目列表生成一套颜色
    '''
    # 从 0.1 到 0.9 取色，避开色盘两端过暗或过亮的颜色，视觉效果更佳
    color_vec_norm = np.linspace(0.1, 0.9, len(key_items))
    colormap = mpl.colormaps[color_map]
    colors = colormap(color_vec_norm)
    color_dict = {}
    for i, item in enumerate(key_items):
        color_dict[item] = colors[i]
    return color_dict

# 为4个模拟模型生成一套独立的 Plasma 渐变色
SIM_MODEL_COLORS = map_color_dict(SIM_MODELS, color_map='tab20c')

# 合并最终的颜色字典
# Real Data 使用深灰色，模拟模型使用新生成的 Plasma 色系
MODEL_COLORS = {
    "Real Data": "#333333", # 深灰色，作为基准
    "RW": SIM_MODEL_COLORS["RW"],
    "RW_switch": SIM_MODEL_COLORS["RW_switch"],
    "KRW": SIM_MODEL_COLORS["KRW"],
    "KRW_switch": SIM_MODEL_COLORS["KRW_switch"],
}


# Same geometry as V13 fitting code.
WIDTH_REAL = 4100
HEIGHT_REAL = 3500
BIN_SIZE = 10
MIDLINE_X = (WIDTH_REAL + 2 * BIN_SIZE) / 2.0
FIT_MAX_TIMESTAMP = int(12 * 60 * 20)
DEFAULT_DF_SELECTION = list(range(10, 20))

# Plot controls.
MAX_TRAJECTORY_WORMS_REAL = 90
MAX_TRAJECTORY_WORMS_SIM = 90
TRAJ_ALPHA_REAL = 0.25
TRAJ_ALPHA_SIM = 0.45
SCATTER_POINTS_MAX = 350
FIG_DPI = 180
ALPHA_SIGNIFICANCE = 0.05

PAIR_ORDER = [
    ("3g-1gNa-1h", "3g-1gNa-4h"),
    ("3g-1gNa-1h", "1gNa-end(8x)"),
    ("3g-1gNa-4h", "1gNa-end(8x)"),
]

FEATURES = [
    "mean_CTX",
    "mean_abs_agl_velocity",
    "net_displacement",
    "mean_dist_to_midline",
]

FEATURE_LABELS = {
    "mean_CTX": "Mean CTX",
    "mean_abs_agl_velocity": "Mean |Angular Velocity|",
    "net_displacement": "Net Displacement (pixels)",
    "mean_dist_to_midline": "Mean Distance to Midline (pixels)",
}

# Lower-is-better features are uncommon here; all features use mean-difference
# direction relative to real data. No sign flip is applied.


# =========================
# GENERAL HELPERS
# =========================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_condition(condition: str) -> str:
    return condition.replace(" ", "_")


def finite_array(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    return arr[np.isfinite(arr)]


def sem(x: Iterable[float]) -> float:
    arr = finite_array(x)
    if len(arr) <= 1:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / math.sqrt(len(arr)))


def zscore_vec(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(arr)):
        return np.full_like(arr, np.nan, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def sign0(x: float, eps: float = 1e-12) -> int:
    if not np.isfinite(x) or abs(x) <= eps:
        return 0
    return 1 if x > 0 else -1


def p_to_label(p: float) -> str:
    if not np.isfinite(p):
        return "na"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni correction for a short list of p-values."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return adjusted.tolist()

    order = valid[np.argsort(p[valid])]
    m = len(order)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted.tolist()


def mannwhitney_p(a: Sequence[float], b: Sequence[float]) -> float:
    x = finite_array(a)
    y = finite_array(b)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def cliffs_delta(a: Sequence[float], b: Sequence[float], max_pairs: int = 250_000) -> float:
    """Cliff's delta, positive if b tends to be larger than a.

    Uses sub-sampling for very large pair counts to avoid quadratic memory/time.
    """
    x = finite_array(a)
    y = finite_array(b)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    n_pairs = len(x) * len(y)
    rng = np.random.default_rng(202606)
    if n_pairs > max_pairs:
        xi = rng.choice(x, size=int(math.sqrt(max_pairs)), replace=True)
        yi = rng.choice(y, size=int(math.sqrt(max_pairs)), replace=True)
        x, y = xi, yi
    diff = y[:, None] - x[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def sample_for_scatter(arr: Sequence[float], max_points: int = SCATTER_POINTS_MAX, seed: int = 123) -> np.ndarray:
    x = finite_array(arr)
    if len(x) <= max_points:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_points, replace=False)
    return x[idx]


def apply_common_axes(ax, title: str, ylabel: str, xlabels: Sequence[str]) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=25, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)


def add_sig_bracket(ax, x1: float, x2: float, y: float, h: float, text: str, fontsize: int = 7) -> None:
    if text == "na":
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color="black")
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=fontsize)


# =========================
# DATA LOADING
# =========================
def load_condition_dict(path: str) -> Dict[str, List[pd.DataFrame]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find condition_dict.pkl at {path}. This file is required for real-data comparison."
        )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError("condition_dict.pkl must contain a dict mapping condition names to dataframe lists")
    return obj


def select_condition_dfs(condition_dict: Dict[str, List[pd.DataFrame]], condition: str,
                         df_selection: Optional[Sequence[int]]) -> Tuple[List[pd.DataFrame], List[int]]:
    if condition not in condition_dict:
        return [], []
    dfs = condition_dict[condition]
    if df_selection is None:
        return dfs, list(range(len(dfs)))
    idxs = [int(i) for i in df_selection if 0 <= int(i) < len(dfs)]
    return [dfs[i] for i in idxs], idxs


def standardize_real_df(df: pd.DataFrame, df_index: int, condition: str) -> pd.DataFrame:
    d = df.copy()
    if "ID" not in d.columns:
        d["ID"] = np.arange(len(d))
    d["orig_ID"] = d["ID"].astype(str)
    d["ID"] = d["ID"].astype(str) + f"_df{df_index}"
    d["df_index"] = df_index
    d["condition"] = condition
    d["model"] = "Real Data"

    if "Timestamp" in d.columns:
        d = d[d["Timestamp"] <= FIT_MAX_TIMESTAMP].copy()

    x_col = "X" if "X" in d.columns else "x" if "x" in d.columns else None
    y_col = "Y" if "Y" in d.columns else "y" if "y" in d.columns else None
    if x_col is None or y_col is None:
        raise ValueError(f"Real dataframe for {condition} is missing X/Y or x/y columns")

    d = d.rename(columns={x_col: "x", y_col: "y"})

    # Required feature columns may exist from SaveFeatures.py. If absent, leave as NaN;
    # trajectory features can still be computed from x/y.
    for col in ["CTX", "agl_velocity", "speed"]:
        if col not in d.columns:
            d[col] = np.nan
    d["abs_agl_velocity"] = np.abs(pd.to_numeric(d["agl_velocity"], errors="coerce"))
    return d


def load_real_data(condition_dict: Dict[str, List[pd.DataFrame]], df_selection: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame_rows = []
    traj_rows = []
    for condition in CONDITIONS:
        dfs, idxs = select_condition_dfs(condition_dict, condition, df_selection)
        if not dfs:
            print(f"[WARNING] No real data found for condition: {condition}")
            continue
        for df_obj, idx in zip(dfs, idxs):
            d = standardize_real_df(df_obj, idx, condition)
            frame_cols = ["condition", "model", "df_index", "ID", "CTX", "agl_velocity", "abs_agl_velocity", "speed"]
            frame_rows.append(d[frame_cols].copy())
            traj_cols = ["condition", "model", "df_index", "ID", "x", "y"]
            if "Timestamp" in d.columns:
                traj_cols.append("Timestamp")
            traj_rows.append(d[traj_cols].copy())
    if not frame_rows:
        raise RuntimeError("No real frame data could be loaded")
    return pd.concat(frame_rows, ignore_index=True), pd.concat(traj_rows, ignore_index=True)


def find_final_fit_dir(results_dir: str) -> str:
    candidates = [
        os.path.join(results_dir, "final_fit"),
        results_dir,
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Could not find final_fit directory inside {results_dir}")


def find_condition_dir(final_fit_dir: str, condition: str) -> Optional[str]:
    candidates = [
        os.path.join(final_fit_dir, condition),
        os.path.join(final_fit_dir, sanitize_condition(condition)),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # Fallback: fuzzy match.
    for d in glob.glob(os.path.join(final_fit_dir, "*")):
        if os.path.isdir(d) and condition in os.path.basename(d):
            return d
    return None


def find_trajectory_csv(condition_dir: str, condition: str, model: str) -> Optional[str]:
    exact = os.path.join(condition_dir, f"{condition}_{model}_trajectories.csv")
    if os.path.exists(exact):
        return exact
    patterns = [
        os.path.join(condition_dir, f"*{model}*trajectories*.csv"),
        os.path.join(condition_dir, f"*{model}*traj*.csv"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    # Avoid matching RW inside RW_switch when asking for RW.
    clean_matches = []
    for m in matches:
        base = os.path.basename(m)
        if model == "RW" and "RW_switch" in base:
            continue
        if model == "KRW" and "KRW_switch" in base:
            continue
        clean_matches.append(m)
    return sorted(clean_matches)[0] if clean_matches else None


def load_simulation_trajectories(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    final_fit_dir = find_final_fit_dir(results_dir)
    frame_rows = []
    traj_rows = []
    for condition in CONDITIONS:
        cdir = find_condition_dir(final_fit_dir, condition)
        if cdir is None:
            print(f"[WARNING] Missing final-fit condition directory for {condition}")
            continue
        for model in SIM_MODELS:
            path = find_trajectory_csv(cdir, condition, model)
            if path is None:
                print(f"[WARNING] Missing trajectory CSV for {condition} / {model} in {cdir}")
                continue
            d = pd.read_csv(path)
            if "ID" not in d.columns:
                d["ID"] = 0
            d["ID"] = d["ID"].astype(str)
            d["condition"] = condition
            d["model"] = model
            if "x" not in d.columns or "y" not in d.columns:
                raise ValueError(f"Simulation trajectory file missing x/y columns: {path}")
            for col in ["CTX", "agl_velocity", "speed"]:
                if col not in d.columns:
                    d[col] = np.nan
            if "abs_agl_velocity" not in d.columns:
                d["abs_agl_velocity"] = np.abs(pd.to_numeric(d["agl_velocity"], errors="coerce"))
            if "mean_dist_to_midline" not in d.columns:
                d["mean_dist_to_midline"] = np.nan
            if "net_displacement" not in d.columns:
                d["net_displacement"] = np.nan
            frame_cols = ["condition", "model", "ID", "CTX", "agl_velocity", "abs_agl_velocity", "speed"]
            frame_rows.append(d[frame_cols].copy())
            traj_cols = ["condition", "model", "ID", "x", "y", "net_displacement", "mean_dist_to_midline"]
            optional_cols = ["timestep", "select_dist", "selected_U", "selected_reward", "p_high", "state"]
            for col in optional_cols:
                if col in d.columns:
                    traj_cols.append(col)
            traj_rows.append(d[traj_cols].copy())
    if not frame_rows:
        raise RuntimeError(f"No simulated trajectory CSVs found under {results_dir}")
    return pd.concat(frame_rows, ignore_index=True), pd.concat(traj_rows, ignore_index=True)


def compute_worm_features_from_frame_traj(frame_df: pd.DataFrame, traj_df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per worm with group-level behavioral features."""
    feature_rows = []
    key_cols = ["condition", "model", "ID"]

    # Frame-derived means: CTX and absolute angular velocity.
    frame_summary = frame_df.groupby(key_cols).agg(
        mean_CTX=("CTX", "mean"),
        mean_abs_agl_velocity=("abs_agl_velocity", "mean"),
    ).reset_index()

    for (condition, model, worm_id), g in traj_df.groupby(key_cols):
        gg = g.dropna(subset=["x", "y"])
        if len(gg) > 1:
            start = gg.iloc[0][["x", "y"]].values.astype(float)
            end = gg.iloc[-1][["x", "y"]].values.astype(float)
            net_disp_from_xy = float(np.linalg.norm(end - start))
            mean_mid_from_xy = float(np.nanmean(np.abs(gg["x"].values.astype(float) - MIDLINE_X)))
        else:
            net_disp_from_xy = np.nan
            mean_mid_from_xy = np.nan

        net_col = finite_array(gg["net_displacement"].values) if "net_displacement" in gg.columns else np.array([])
        mid_col = finite_array(gg["mean_dist_to_midline"].values) if "mean_dist_to_midline" in gg.columns else np.array([])
        net_disp = float(net_col[0]) if len(net_col) else net_disp_from_xy
        mean_mid = float(mid_col[0]) if len(mid_col) else mean_mid_from_xy

        feature_rows.append({
            "condition": condition,
            "model": model,
            "ID": worm_id,
            "net_displacement": net_disp,
            "mean_dist_to_midline": mean_mid,
        })

    traj_summary = pd.DataFrame(feature_rows)
    merged = pd.merge(frame_summary, traj_summary, on=key_cols, how="outer")
    return merged


def load_all_analysis_data(condition_pkl: str, results_dir: str, df_selection: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_dict = load_condition_dict(condition_pkl)
    real_frame, real_traj = load_real_data(condition_dict, df_selection)
    sim_frame, sim_traj = load_simulation_trajectories(results_dir)

    frame_df = pd.concat([real_frame, sim_frame], ignore_index=True)
    traj_df = pd.concat([real_traj, sim_traj], ignore_index=True)
    worm_df = compute_worm_features_from_frame_traj(frame_df, traj_df)

    # Keep consistent model order.
    worm_df["model"] = pd.Categorical(worm_df["model"], categories=MODELS, ordered=True)
    frame_df["model"] = pd.Categorical(frame_df["model"], categories=MODELS, ordered=True)
    traj_df["model"] = pd.Categorical(traj_df["model"], categories=MODELS, ordered=True)
    return frame_df, traj_df, worm_df


# =========================
# STATISTICS
# =========================
def pairwise_group_stats(worm_df: pd.DataFrame, feature: str, model: str) -> pd.DataFrame:
    rows = []
    p_raw = []
    for c1, c2 in PAIR_ORDER:
        a = worm_df[(worm_df["model"] == model) & (worm_df["condition"] == c1)][feature].values
        b = worm_df[(worm_df["model"] == model) & (worm_df["condition"] == c2)][feature].values
        p = mannwhitney_p(a, b)
        p_raw.append(p)
        diff = float(np.nanmean(b) - np.nanmean(a)) if len(finite_array(a)) and len(finite_array(b)) else np.nan
        rows.append({
            "feature": feature,
            "model": model,
            "condition_a": c1,
            "condition_b": c2,
            "n_a": int(len(finite_array(a))),
            "n_b": int(len(finite_array(b))),
            "mean_a": float(np.nanmean(finite_array(a))) if len(finite_array(a)) else np.nan,
            "mean_b": float(np.nanmean(finite_array(b))) if len(finite_array(b)) else np.nan,
            "mean_diff_b_minus_a": diff,
            "direction_sign": sign0(diff),
            "cliffs_delta_b_minus_a": cliffs_delta(a, b),
            "p_raw": p,
        })
    p_adj = holm_adjust(p_raw)
    for r, p in zip(rows, p_adj):
        r["p_holm"] = p
        r["significant_holm_0p05"] = bool(np.isfinite(p) and p < ALPHA_SIGNIFICANCE)
        r["sig_label"] = p_to_label(p)
    return pd.DataFrame(rows)


def compute_all_pairwise_stats(worm_df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    for feature in FEATURES:
        if feature not in worm_df.columns:
            continue
        for model in MODELS:
            if len(worm_df[worm_df["model"] == model]) == 0:
                continue
            all_rows.append(pairwise_group_stats(worm_df, feature, model))
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def compute_group_capture_scores(worm_df: pd.DataFrame, pairwise_stats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for feature in FEATURES:
        real_means = []
        for c in CONDITIONS:
            real_means.append(
                float(np.nanmean(worm_df[(worm_df["model"] == "Real Data") & (worm_df["condition"] == c)][feature].values))
            )
        real_z = zscore_vec(real_means)
        real_pair = pairwise_stats[(pairwise_stats["model"] == "Real Data") & (pairwise_stats["feature"] == feature)].copy()
        if real_pair.empty:
            continue
        real_pair = real_pair.set_index(["condition_a", "condition_b"])
        real_dirs = []
        real_sigs = []
        real_effects = []
        for pair in PAIR_ORDER:
            if pair in real_pair.index:
                real_dirs.append(int(real_pair.loc[pair, "direction_sign"]))
                real_sigs.append(bool(real_pair.loc[pair, "significant_holm_0p05"]))
                real_effects.append(float(real_pair.loc[pair, "cliffs_delta_b_minus_a"]))
            else:
                real_dirs.append(0)
                real_sigs.append(False)
                real_effects.append(np.nan)

        for model in SIM_MODELS:
            model_means = []
            for c in CONDITIONS:
                model_means.append(
                    float(np.nanmean(worm_df[(worm_df["model"] == model) & (worm_df["condition"] == c)][feature].values))
                )
            model_z = zscore_vec(model_means)
            model_pair = pairwise_stats[(pairwise_stats["model"] == model) & (pairwise_stats["feature"] == feature)].copy()
            if model_pair.empty:
                continue
            model_pair = model_pair.set_index(["condition_a", "condition_b"])
            model_dirs = []
            model_sigs = []
            model_effects = []
            for pair in PAIR_ORDER:
                if pair in model_pair.index:
                    model_dirs.append(int(model_pair.loc[pair, "direction_sign"]))
                    model_sigs.append(bool(model_pair.loc[pair, "significant_holm_0p05"]))
                    model_effects.append(float(model_pair.loc[pair, "cliffs_delta_b_minus_a"]))
                else:
                    model_dirs.append(0)
                    model_sigs.append(False)
                    model_effects.append(np.nan)

            direction_match_rate = float(np.mean([a == b for a, b in zip(real_dirs, model_dirs)]))
            significance_match_rate = float(np.mean([a == b for a, b in zip(real_sigs, model_sigs)]))
            pattern_mse_z = float(np.nanmean((real_z - model_z) ** 2))
            pattern_similarity = float(np.exp(-pattern_mse_z)) if np.isfinite(pattern_mse_z) else 0.0

            real_eff = np.asarray(real_effects, dtype=float)
            model_eff = np.asarray(model_effects, dtype=float)
            valid = np.isfinite(real_eff) & np.isfinite(model_eff)
            if valid.sum() >= 2:
                rho = spearmanr(real_eff[valid], model_eff[valid]).correlation
                if not np.isfinite(rho):
                    rho = 0.0
            else:
                rho = 0.0
            rho01 = float((rho + 1.0) / 2.0)

            # Transparent heuristic score in [approximately 0, 1].
            # It rewards matching direction/significance and close z-scored condition means.
            capture_score = (
                0.35 * direction_match_rate
                + 0.30 * significance_match_rate
                + 0.20 * pattern_similarity
                + 0.15 * rho01
            )

            rows.append({
                "feature": feature,
                "model": model,
                "real_condition_means": ";".join(f"{c}:{v:.6g}" for c, v in zip(CONDITIONS, real_means)),
                "model_condition_means": ";".join(f"{c}:{v:.6g}" for c, v in zip(CONDITIONS, model_means)),
                "direction_match_rate": direction_match_rate,
                "significance_match_rate": significance_match_rate,
                "pattern_mse_z": pattern_mse_z,
                "pattern_similarity_exp_neg_mse": pattern_similarity,
                "effect_spearman_rho": float(rho),
                "capture_score": float(capture_score),
            })

    feature_scores = pd.DataFrame(rows)
    if feature_scores.empty:
        return feature_scores, pd.DataFrame()

    summary = feature_scores.groupby("model").agg(
        mean_capture_score=("capture_score", "mean"),
        sem_capture_score=("capture_score", sem),
        mean_direction_match_rate=("direction_match_rate", "mean"),
        mean_significance_match_rate=("significance_match_rate", "mean"),
        mean_pattern_mse_z=("pattern_mse_z", "mean"),
        mean_effect_spearman_rho=("effect_spearman_rho", "mean"),
    ).reset_index()
    summary = summary.sort_values("mean_capture_score", ascending=False)
    summary["rank_by_group_capture"] = np.arange(1, len(summary) + 1)
    return feature_scores, summary


# =========================
# PLOTTING
# =========================
def plot_trajectories_5panel(traj_df: pd.DataFrame, output_dir: str) -> None:
    out_dir = ensure_dir(os.path.join(output_dir, "trajectories"))
    rng = np.random.default_rng(202606)
    for condition in CONDITIONS:
        fig, axes = plt.subplots(1, len(MODELS), figsize=(20, 4), sharex=True, sharey=True)
        for ax, model in zip(axes, MODELS):
            d = traj_df[(traj_df["condition"] == condition) & (traj_df["model"] == model)]
            ids = np.array(sorted(d["ID"].astype(str).unique()))
            max_n = MAX_TRAJECTORY_WORMS_REAL if model == "Real Data" else MAX_TRAJECTORY_WORMS_SIM
            if len(ids) > max_n:
                ids = rng.choice(ids, size=max_n, replace=False)
            color = MODEL_COLORS.get(model, "black")
            alpha = TRAJ_ALPHA_REAL if model == "Real Data" else TRAJ_ALPHA_SIM
            for worm_id in ids:
                g = d[d["ID"].astype(str) == str(worm_id)].dropna(subset=["x", "y"])
                if len(g) > 1:
                    ax.plot(g["x"].values, g["y"].values, lw=0.45, alpha=alpha, color=color)
            ax.set_title(model, fontsize=9)
            ax.set_xlim(0, WIDTH_REAL + 2 * BIN_SIZE)
            ax.set_ylim(0, HEIGHT_REAL)
            ax.grid(alpha=0.25)
        fig.suptitle(f"Trajectory comparison: {condition}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(out_dir, f"trajectories_5panel_{condition}.png"), dpi=FIG_DPI)
        plt.close(fig)


def plot_across_condition_violin(worm_df: pd.DataFrame, pairwise_stats: pd.DataFrame, feature: str,
                                 output_dir: str) -> None:
    out_dir = ensure_dir(os.path.join(output_dir, "group_difference_violins"))
    fig, axes = plt.subplots(1, len(MODELS), figsize=(20, 4.2), sharey=False)
    xlabels = [CONDITION_LABELS[c] for c in CONDITIONS]

    for ax, model in zip(axes, MODELS):
        arrays = [
            finite_array(worm_df[(worm_df["model"] == model) & (worm_df["condition"] == c)][feature].values)
            for c in CONDITIONS
        ]
        positions = np.arange(len(CONDITIONS))
        color = MODEL_COLORS.get(model, "gray")
        valid_arrays = [a if len(a) else np.array([0.0]) for a in arrays]
        vp = ax.violinplot(valid_arrays, positions=positions, widths=0.7, showmeans=True, showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.85)
        for key in ["cmeans", "cmedians"]:
            if key in vp:
                vp[key].set_color("black")
                vp[key].set_linewidth(1.0)

        for i, arr in enumerate(arrays):
            scat = sample_for_scatter(arr, seed=1234 + i)
            if len(scat):
                jitter = np.random.default_rng(100 + i).normal(0, 0.045, size=len(scat))
                ax.scatter(np.full(len(scat), i) + jitter, scat, s=5, color="black", alpha=0.55, zorder=3)

        apply_common_axes(ax, f"Model: {model}", FEATURE_LABELS[feature], xlabels)

        stats_m = pairwise_stats[(pairwise_stats["feature"] == feature) & (pairwise_stats["model"] == model)]
        if not stats_m.empty:
            y_values = finite_array(np.concatenate([a for a in arrays if len(a)])) if any(len(a) for a in arrays) else np.array([0])
            y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
            y_range = max(y_max - y_min, 1e-6)
            base = y_max + 0.08 * y_range
            step = 0.095 * y_range
            h = 0.025 * y_range
            for k, (ca, cb) in enumerate(PAIR_ORDER):
                row = stats_m[(stats_m["condition_a"] == ca) & (stats_m["condition_b"] == cb)]
                if row.empty:
                    continue
                x1 = CONDITIONS.index(ca)
                x2 = CONDITIONS.index(cb)
                label = str(row.iloc[0]["sig_label"])
                add_sig_bracket(ax, x1, x2, base + step * k, h, label)
            ax.set_ylim(y_min - 0.05 * y_range, base + step * len(PAIR_ORDER) + 0.10 * y_range)

    fig.suptitle(f"Across-condition group differences: {FEATURE_LABELS[feature]}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out_dir, f"onefig_{feature}_5panels.png"), dpi=FIG_DPI)
    plt.close(fig)


def plot_model_vs_real_by_condition(worm_df: pd.DataFrame, feature: str, output_dir: str) -> None:
    out_dir = ensure_dir(os.path.join(output_dir, "model_vs_real_violins"))
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(18, 4.5), sharey=True)
    model_labels = MODELS

    for ax, condition in zip(axes, CONDITIONS):
        arrays = [finite_array(worm_df[(worm_df["condition"] == condition) & (worm_df["model"] == m)][feature].values) for m in MODELS]
        positions = np.arange(len(MODELS))
        vp = ax.violinplot([a if len(a) else np.array([0.0]) for a in arrays], positions=positions,
                           widths=0.72, showmeans=True, showmedians=True, showextrema=False)
        for body, model in zip(vp["bodies"], MODELS):
            body.set_facecolor(MODEL_COLORS.get(model, "gray"))
            body.set_edgecolor("black")
            body.set_alpha(0.82)
        for key in ["cmeans", "cmedians"]:
            if key in vp:
                vp[key].set_color("black")
                vp[key].set_linewidth(1.0)
        for i, arr in enumerate(arrays):
            scat = sample_for_scatter(arr, seed=301 + i)
            if len(scat):
                jitter = np.random.default_rng(400 + i).normal(0, 0.045, size=len(scat))
                ax.scatter(np.full(len(scat), i) + jitter, scat, s=5, color="black", alpha=0.45, zorder=3)

        ax.set_title(condition, fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        if ax is axes[0]:
            ax.set_ylabel(FEATURE_LABELS[feature])

        # Real-vs-each-model tests, Holm-adjusted.
        pvals = [mannwhitney_p(arrays[0], arrays[i]) for i in range(1, len(MODELS))]
        padj = holm_adjust(pvals)
        y_values = finite_array(np.concatenate([a for a in arrays if len(a)])) if any(len(a) for a in arrays) else np.array([0])
        y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
        y_range = max(y_max - y_min, 1e-6)
        base = y_max + 0.08 * y_range
        step = 0.075 * y_range
        h = 0.02 * y_range
        for k, p in enumerate(padj):
            add_sig_bracket(ax, 0, k + 1, base + step * k, h, p_to_label(p), fontsize=7)
        ax.set_ylim(y_min - 0.05 * y_range, base + step * len(padj) + 0.10 * y_range)

    fig.suptitle(f"Real-vs-model distributions by condition: {FEATURE_LABELS[feature]}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out_dir, f"model_vs_real_{feature}.png"), dpi=FIG_DPI)
    plt.close(fig)


def plot_distribution_overlays(frame_df: pd.DataFrame, worm_df: pd.DataFrame, output_dir: str) -> None:
    out_dir = ensure_dir(os.path.join(output_dir, "distribution_overlays"))

    # Frame-level CTX overlay.
    overlay_specs = [
        (frame_df, "CTX", "Frame-level CTX", "distribution_overlay_CTX.png"),
        (frame_df, "abs_agl_velocity", "Frame-level |Angular Velocity|", "distribution_overlay_abs_agl_velocity.png"),
        (worm_df, "mean_CTX", "Per-worm Mean CTX", "distribution_overlay_mean_CTX.png"),
        (worm_df, "net_displacement", "Per-worm Net Displacement", "distribution_overlay_net_displacement.png"),
        (worm_df, "mean_dist_to_midline", "Per-worm Mean Distance to Midline", "distribution_overlay_mean_dist_to_midline.png"),
    ]

    for source_df, col, label, fname in overlay_specs:
        fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(18, 4), sharey=True)
        for ax, condition in zip(axes, CONDITIONS):
            arrays_by_model = {}
            pooled = []
            for model in MODELS:
                d = finite_array(source_df[(source_df["condition"] == condition) & (source_df["model"] == model)][col].values)
                arrays_by_model[model] = d
                if len(d):
                    pooled.append(d)
            if pooled:
                all_vals = finite_array(np.concatenate(pooled))
                lo, hi = np.nanpercentile(all_vals, [0.5, 99.5])
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
                if lo == hi:
                    lo -= 0.5
                    hi += 0.5
                for model in MODELS:
                    d = arrays_by_model.get(model, np.array([]))
                    if len(d) < 2:
                        continue
                    ax.hist(d, bins=45, range=(lo, hi), density=True, histtype="stepfilled",
                            alpha=0.22, color=MODEL_COLORS.get(model, "gray"), label=model)
                    ax.hist(d, bins=45, range=(lo, hi), density=True, histtype="step",
                            alpha=0.8, color=MODEL_COLORS.get(model, "gray"), linewidth=1.0)
            ax.set_title(condition, fontsize=10)
            ax.set_xlabel(label, fontsize=9)
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Density")
        axes[-1].legend(fontsize=8, frameon=True)
        fig.suptitle(f"Distribution overlay: {label}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(out_dir, fname), dpi=FIG_DPI)
        plt.close(fig)


def plot_loss_bars(results_dir: str, output_dir: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    out_dir = ensure_dir(os.path.join(output_dir, "loss_bars"))
    final_paths = [
        os.path.join(results_dir, "ALL_CONDITIONS_results.csv"),
        os.path.join(results_dir, "final_fit", "ALL_CONDITIONS_results.csv"),
        "ALL_CONDITIONS_results.csv",
    ]
    final_csv = next((p for p in final_paths if os.path.exists(p)), None)
    final_rank = pd.DataFrame()
    cv_rank = None

    if final_csv is not None:
        final_df = pd.read_csv(final_csv)
        final_df = final_df[final_df["model"].isin(SIM_MODELS)].copy()
        final_rank = final_df.sort_values(["condition", "loss"])
        final_rank.to_csv(os.path.join(output_dir, "LOSS_RANKING_FINAL.csv"), index=False)

        # Grouped final-loss bar.
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(CONDITIONS))
        width = 0.18
        for i, model in enumerate(SIM_MODELS):
            vals = []
            for c in CONDITIONS:
                row = final_df[(final_df["condition"] == c) & (final_df["model"] == model)]
                vals.append(float(row["loss"].iloc[0]) if len(row) else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width=width, label=model, color=MODEL_COLORS[model], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
        ax.set_ylabel("Final-fit loss")
        ax.set_title("Final-fit loss by condition")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "final_loss_by_condition_bar.png"), dpi=FIG_DPI)
        plt.close(fig)

        # Mean final-loss bar.
        mean_df = final_df.groupby("model").agg(mean_loss=("loss", "mean"), sem_loss=("loss", sem)).reset_index()
        mean_df = mean_df.sort_values("mean_loss")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        xpos = np.arange(len(mean_df))
        ax.bar(xpos, mean_df["mean_loss"], yerr=mean_df["sem_loss"],
               color=[MODEL_COLORS[m] for m in mean_df["model"]], alpha=0.85, capsize=4)
        ax.set_xticks(xpos)
        ax.set_xticklabels(mean_df["model"], rotation=25, ha="right")
        ax.set_ylabel("Mean final-fit loss across conditions")
        ax.set_title("Average final-fit loss")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "final_loss_mean_bar.png"), dpi=FIG_DPI)
        plt.close(fig)
    else:
        print("[WARNING] Could not find ALL_CONDITIONS_results.csv for final-loss plots")

    cv_paths = [
        os.path.join(results_dir, "CV_MODEL_SUMMARY.csv"),
        os.path.join(results_dir, "cross_validation", "CV_MODEL_SUMMARY.csv"),
    ]
    cv_csv = next((p for p in cv_paths if os.path.exists(p)), None)
    if cv_csv is not None:
        cv_df = pd.read_csv(cv_csv)
        cv_df = cv_df[cv_df["model"].isin(SIM_MODELS)].copy()
        if "test_data_loss_mean" in cv_df.columns:
            cv_rank = cv_df.sort_values(["condition", "test_data_loss_mean"])
            cv_rank.to_csv(os.path.join(output_dir, "LOSS_RANKING_CV.csv"), index=False)

            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(CONDITIONS))
            width = 0.18
            for i, model in enumerate(SIM_MODELS):
                vals, errs = [], []
                for c in CONDITIONS:
                    row = cv_df[(cv_df["condition"] == c) & (cv_df["model"] == model)]
                    vals.append(float(row["test_data_loss_mean"].iloc[0]) if len(row) else np.nan)
                    errs.append(float(row["test_data_loss_sem"].iloc[0]) if len(row) and "test_data_loss_sem" in row else np.nan)
                ax.bar(x + (i - 1.5) * width, vals, yerr=errs, width=width, label=model,
                       color=MODEL_COLORS[model], alpha=0.85, capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
            ax.set_ylabel("Held-out test data loss")
            ax.set_title("Cross-validated held-out loss by condition")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "cv_test_loss_by_condition_bar.png"), dpi=FIG_DPI)
            plt.close(fig)

            cv_mean = cv_df.groupby("model").agg(
                mean_test_loss=("test_data_loss_mean", "mean"),
                sem_test_loss=("test_data_loss_mean", sem),
            ).reset_index().sort_values("mean_test_loss")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            xpos = np.arange(len(cv_mean))
            ax.bar(xpos, cv_mean["mean_test_loss"], yerr=cv_mean["sem_test_loss"],
                   color=[MODEL_COLORS[m] for m in cv_mean["model"]], alpha=0.85, capsize=4)
            ax.set_xticks(xpos)
            ax.set_xticklabels(cv_mean["model"], rotation=25, ha="right")
            ax.set_ylabel("Mean held-out test loss across conditions")
            ax.set_title("Average cross-validated held-out loss")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "cv_test_loss_mean_bar.png"), dpi=FIG_DPI)
            plt.close(fig)
    else:
        print("[INFO] No CV_MODEL_SUMMARY.csv found. Skipping CV loss plots.")

    return final_rank, cv_rank


def plot_group_capture_bars(feature_scores: pd.DataFrame, summary_scores: pd.DataFrame, output_dir: str) -> None:
    out_dir = ensure_dir(os.path.join(output_dir, "group_capture"))
    if summary_scores.empty:
        return
    sorted_summary = summary_scores.sort_values("mean_capture_score", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xpos = np.arange(len(sorted_summary))
    ax.bar(xpos, sorted_summary["mean_capture_score"],
           yerr=sorted_summary["sem_capture_score"],
           color=[MODEL_COLORS[m] for m in sorted_summary["model"]], alpha=0.85, capsize=4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(sorted_summary["model"], rotation=25, ha="right")
    ax.set_ylabel("Group-difference capture score")
    ax.set_title("Which model best captures real between-condition differences?")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_group_capture_score_bar.png"), dpi=FIG_DPI)
    plt.close(fig)

    if not feature_scores.empty:
        pivot = feature_scores.pivot(index="feature", columns="model", values="capture_score").reindex(FEATURES)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(pivot.index))
        width = 0.18
        for i, model in enumerate(SIM_MODELS):
            vals = pivot[model].values if model in pivot.columns else np.full(len(pivot.index), np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width=width, label=model, color=MODEL_COLORS[model], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in pivot.index], rotation=25, ha="right")
        ax.set_ylabel("Capture score")
        ax.set_title("Group-difference capture score by feature")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "group_capture_score_by_feature_bar.png"), dpi=FIG_DPI)
        plt.close(fig)


# =========================
# REPORTING
# =========================
def write_summary_text(output_dir: str,
                       final_rank: pd.DataFrame,
                       cv_rank: Optional[pd.DataFrame],
                       group_summary: pd.DataFrame) -> None:
    lines = []
    lines.append("Fitting-result analysis summary")
    lines.append("================================")
    lines.append("")

    if final_rank is not None and not final_rank.empty:
        lines.append("Final-fit loss ranking")
        for condition in CONDITIONS:
            g = final_rank[final_rank["condition"] == condition].sort_values("loss")
            if len(g):
                best = g.iloc[0]
                lines.append(f"- {condition}: best final-fit loss = {best['model']} ({best['loss']:.4g})")
        mean_final = final_rank.groupby("model")["loss"].mean().sort_values()
        if len(mean_final):
            lines.append(f"- Across-condition mean final-fit loss winner: {mean_final.index[0]} ({mean_final.iloc[0]:.4g})")
        lines.append("")

    if cv_rank is not None and not cv_rank.empty and "test_data_loss_mean" in cv_rank.columns:
        lines.append("Cross-validated held-out loss ranking")
        for condition in CONDITIONS:
            g = cv_rank[cv_rank["condition"] == condition].sort_values("test_data_loss_mean")
            if len(g):
                best = g.iloc[0]
                lines.append(f"- {condition}: best held-out loss = {best['model']} ({best['test_data_loss_mean']:.4g})")
        mean_cv = cv_rank.groupby("model")["test_data_loss_mean"].mean().sort_values()
        if len(mean_cv):
            lines.append(f"- Across-condition mean held-out loss winner: {mean_cv.index[0]} ({mean_cv.iloc[0]:.4g})")
        lines.append("")
    else:
        lines.append("Cross-validated held-out loss ranking")
        lines.append("- CV_MODEL_SUMMARY.csv was not found, so no held-out-loss ranking was computed.")
        lines.append("")

    if group_summary is not None and not group_summary.empty:
        lines.append("Group-difference capture ranking")
        for _, row in group_summary.sort_values("rank_by_group_capture").iterrows():
            lines.append(
                f"- rank {int(row['rank_by_group_capture'])}: {row['model']} "
                f"score={row['mean_capture_score']:.3f}, "
                f"direction_match={row['mean_direction_match_rate']:.3f}, "
                f"sig_match={row['mean_significance_match_rate']:.3f}, "
                f"pattern_mse_z={row['mean_pattern_mse_z']:.3f}"
            )
        best = group_summary.sort_values("rank_by_group_capture").iloc[0]
        lines.append("")
        lines.append(f"Best group-difference capture model by this diagnostic: {best['model']}.")
        lines.append("")

    lines.append("Interpretation notes")
    lines.append("- Final-fit loss measures how well the model fits the same data used for fitting.")
    lines.append("- Held-out CV loss is stricter and should be preferred for model selection when available.")
    lines.append("- The group-capture score asks whether the simulated data preserve real between-condition direction and significance patterns.")
    lines.append("- If loss and group-capture disagree, inspect the feature-specific capture table and plots before choosing a main model.")

    with open(os.path.join(output_dir, "analysis_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# MAIN
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze V13 fitting trajectories and model results.")
    parser.add_argument("--condition-pkl", default="condition_dict.pkl", help="Path to condition_dict.pkl")
    parser.add_argument("--results-dir", default="SingleConc_resultsV13_CV_features", help="Path to V13 result directory")
    parser.add_argument("--output-dir", default="SingleConc_resultsV13_analysis", help="Output directory for analysis figures and CSVs")
    parser.add_argument("--df-selection", nargs="*", type=int, default=DEFAULT_DF_SELECTION,
                        help="Original dataframe indices used for real-data comparison. Default: 10 11 ... 19")
    parser.add_argument("--save-large-tables", action="store_true",
                        help="Also save FRAME_FEATURE_TABLE.csv and TRAJECTORY_TABLE.csv. These can be large.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    print("[INFO] Loading real and simulated data...")
    frame_df, traj_df, worm_df = load_all_analysis_data(
        condition_pkl=args.condition_pkl,
        results_dir=args.results_dir,
        df_selection=args.df_selection,
    )

    print("[INFO] Saving per-worm feature table...")
    worm_df.to_csv(os.path.join(output_dir, "PER_WORM_FEATURE_TABLE.csv"), index=False)
    if args.save_large_tables:
        print("[INFO] Saving large frame and trajectory tables...")
        frame_df.to_csv(os.path.join(output_dir, "FRAME_FEATURE_TABLE.csv"), index=False)
        traj_df.to_csv(os.path.join(output_dir, "TRAJECTORY_TABLE.csv"), index=False)
    else:
        print("[INFO] Skipping large frame/trajectory table export. Use --save-large-tables if needed.")

    print("[INFO] Computing pairwise group statistics...")
    pairwise = compute_all_pairwise_stats(worm_df)
    pairwise.to_csv(os.path.join(output_dir, "PAIRWISE_GROUP_STATS.csv"), index=False)

    print("[INFO] Computing group-difference capture scores...")
    feature_scores, group_summary = compute_group_capture_scores(worm_df, pairwise)
    feature_scores.to_csv(os.path.join(output_dir, "GROUP_DIFFERENCE_CAPTURE_BY_FEATURE.csv"), index=False)
    group_summary.to_csv(os.path.join(output_dir, "MODEL_GROUP_CAPTURE_SUMMARY.csv"), index=False)

    print("[INFO] Plotting trajectories...")
    plot_trajectories_5panel(traj_df, output_dir)

    print("[INFO] Plotting across-condition feature violins...")
    for feature in FEATURES:
        if feature in worm_df.columns:
            plot_across_condition_violin(worm_df, pairwise, feature, output_dir)

    print("[INFO] Plotting model-vs-real feature violins...")
    for feature in FEATURES:
        if feature in worm_df.columns:
            plot_model_vs_real_by_condition(worm_df, feature, output_dir)

    print("[INFO] Plotting distribution overlays...")
    plot_distribution_overlays(frame_df, worm_df, output_dir)

    print("[INFO] Plotting loss bars...")
    final_rank, cv_rank = plot_loss_bars(args.results_dir, output_dir)

    print("[INFO] Plotting group-capture bars...")
    plot_group_capture_bars(feature_scores, group_summary, output_dir)

    print("[INFO] Writing summary...")
    write_summary_text(output_dir, final_rank, cv_rank, group_summary)

    print(f"[INFO] Analysis complete. Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
