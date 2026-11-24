# -*- coding: utf-8 -*-
"""
SingleConc_V13_CV_features.py

Built from SingleConc_V12_KRW_CDF_conditional_value.py and the uploaded SingleConc.py.

Purpose
-------
Fit and compare four C. elegans salt-learning behavior models:
    RW, RW_switch, KRW, KRW_switch

Key model assumptions
---------------------
1. RW / RW_switch are deterministic value models:
       U_low = w_low, U_high = w_high
       target = argmax(U_low, U_high)
   There is no variance denominator in RW.

2. KRW / KRW_switch are probabilistic value models:
       U_low  ~ N(w_low,  c_low^2)
       U_high ~ N(w_high, c_high^2)
       P(high) = Phi((w_high - w_low) / sqrt(c_low^2 + c_high^2))
   A zero-variance branch makes KRW reduce to RW when c_low and c_high approach zero.

3. KRW still keeps c_low and c_high separately identifiable in the movement layer:
   after target selection, the selected target's conditional expected latent value
       E[U_selected | selected target wins]
   is passed to non-switch keep/turn, WV angle, and optionally PIR transition logic.

4. Behavior state remains two-state only:
       state 0 = WV / run-like steering
       state 1 = PIR
   This version does NOT expand the model to an explicit Run/PIR/Pause three-state model.
   The small fixed pause process inherited from the original script is retained as a
   non-fitted empirical transient, not a separate fitted state.

Major improvements over V12
---------------------------
- Added abs_agl_velocity to the fitted frame-level loss.
- Added mean_dist_to_midline to the fitted worm-level loss.
- Kept raw CTX mean penalty, doubled when real and simulated CTX means have opposite signs.
- Fixed the WV unit issue: k_wv_deg is interpreted as degrees per frame per reward unit.
- Replaced rigid reward-gated PR with optional separable PR logit:
      logit(PR) = a0 + a_dr*(-dR_norm) + a_reward*reward
                  + a_interaction*(-dR_norm)*reward
- Added train/test cross-validation over original dataframe indices.
- Added repeated held-out evaluation with fixed seeds.
- Random-search top candidates are recomputed under the full objective before Bayesian optimization.
- Saves fold-level, condition-level, and group-level validation summaries.

Input
-----
condition_dict.pkl in the working directory.

Output
------
SingleConc_resultsV13_CV_features/
    final_fit/
    cross_validation/
    ALL_CONDITIONS_results.csv
    CV_ALL_FOLDS_results.csv
    CV_MODEL_SUMMARY.csv
    CV_GROUP_PATTERN_SCORES.csv
"""

import os
import sys
import math
import random
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance, norm
from scipy.special import log_ndtr

try:
    from skopt import gp_minimize
    from skopt.space import Real
except ImportError:
    gp_minimize = None

    class Real:  # minimal fallback for random-search-only execution
        def __init__(self, low, high, name=None):
            self.low = float(low)
            self.high = float(high)
            self.name = name

        def rvs(self):
            return [np.random.uniform(self.low, self.high)]

from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="skopt")


# =========================
# CONFIG
# =========================
RUN_MODE = "quick"  # quick or full

CONDITIONS_TO_FIT = [
    "3g-1gNa-1h",
    "3g-1gNa-4h",
    "1gNa-end(8x)",
]

# Which original dataframe indices to use per condition. Your recent analyses used 10-19.
DF_SELECTION = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Run final all-data fit and/or train/test cross-validation.
RUN_FINAL_FIT = True
RUN_CROSS_VALIDATION = True
CV_N_SPLITS = 5
CV_SEED = 202606

FRAME_RATE_REAL = 20.0
DT_REAL = 1.0 / FRAME_RATE_REAL
FIT_MINUTES = 12
MAX_TIMESTAMP = int(FIT_MINUTES * 60 * FRAME_RATE_REAL)

WIDTH_REAL = 4100
HEIGHT_REAL = 3500
PLATE_WIDTH_CM = 10.0
BIN_SIZE = 10
MIDLINE_X = (WIDTH_REAL + 2 * BIN_SIZE) / 2.0

PIXEL_LENGTH_CM_PER_PX = PLATE_WIDTH_CM / WIDTH_REAL
PX_PER_CM = WIDTH_REAL / PLATE_WIDTH_CM

minNaCl = 1
maxNaCl = 5
stimulus = [1.0, 5.0]

SIGMA = 10
BUZZ = 5

# Fitted features.
FRAME_FEATURES = ["agl_velocity", "abs_agl_velocity", "CTX"]
WORM_FEATURES = ["net_displacement", "mean_dist_to_midline"]
FEATURE_WEIGHTS = {
    "agl_velocity": 1.0,
    "abs_agl_velocity": 1.0,
    "CTX": 1.5,
    "net_displacement": 0.4,
    "mean_dist_to_midline": 0.8,
}

MODEL_NAMES = ["RW", "RW_switch", "KRW", "KRW_switch"]
MODEL_COLORS = {
    "real": "gray",
    "RW": "royalblue",
    "RW_switch": "deepskyblue",
    "KRW": "tomato",
    "KRW_switch": "darkred",
}

# KRW uncertainty bounds.
C_MIN = 1e-6
C_MAX = 5.0
CHOICE_ZERO_VARIANCE_EPS = 1e-8
DETERMINISTIC_TIE_EPS = 1e-9
SELECTED_U_CLIP = (-20.0, 20.0)
REWARD_CLIP = (0.0, 8.0)

# PR transition form.
USE_SEPARABLE_PR_LOGIT = True
# If False, use old form with optional reward multiplication:
#     sigmoid(a0 + a1*(-dR_norm)*selected_reward)
USE_REWARD_GATED_PR_LEGACY = True

# Raw CTX mean penalty.
CTX_MEAN_PENALTY_WEIGHT = 1.0
CTX_OPPOSITE_SIGN_MULTIPLIER = 2.0
CTX_SIGN_EPS = 1e-3

USE_CTX_DEADZONE = True
CTX_DEADZONE_SPEED_Q = 0.08
CTX_DEADZONE_MIN = 1e-6

LARGE_LOSS = 1e6
SPEED_BINS = 50

INITIAL_SPEED_RAW = 50.0
FINAL_SPEED_RAW = 35.0

MIN_PIR_STEPS = 35
PIR_STEPS_STD = 17.0
DEFAULT_SIGMA_PIR_DEG = 70.0
WV_TRY_EPS_DEG = 8.0
WV_TEMPERATURE = 0.3
PIR_MEAN_ANGLE = 180.0

if RUN_MODE == "quick":
    NUM_WORMS_OPT_FINAL = 60
    NUM_WORMS_OPT_CV = 50
    NUM_WORMS_PLOT = 8

    FINAL_RANDOM_SEARCH_ITERS = 800
    FINAL_N_CALLS_RW = 500
    FINAL_N_CALLS_KRW = 1000
    FINAL_OBJECTIVE_REPEATS = 1

    CV_RANDOM_SEARCH_ITERS = 300
    CV_N_CALLS_RW = 250
    CV_N_CALLS_KRW = 500
    CV_OBJECTIVE_REPEATS = 1

    EVAL_REPEATS = 5
    N_JOBS = 8
    SHOW_PLOTS = False
else:
    NUM_WORMS_OPT_FINAL = 80
    NUM_WORMS_OPT_CV = 60
    NUM_WORMS_PLOT = 24

    FINAL_RANDOM_SEARCH_ITERS = 1200
    FINAL_N_CALLS_RW = 700
    FINAL_N_CALLS_KRW = 1400
    FINAL_OBJECTIVE_REPEATS = 3

    CV_RANDOM_SEARCH_ITERS = 600
    CV_N_CALLS_RW = 400
    CV_N_CALLS_KRW = 800
    CV_OBJECTIVE_REPEATS = 2

    EVAL_REPEATS = 8
    N_JOBS = 8
    SHOW_PLOTS = True

RANDOM_TOPK_FULL_RECOMPUTE = 10
BASE_OUTPUT_DIR = "SingleConc_resultsV13_CV_features"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# =========================
# UTILS
# =========================
def safe_sigmoid(x):
    x_clipped = np.clip(x, -6, 6)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def safe_large_loss(reason, val=LARGE_LOSS):
    print(f"[LOSS_GUARD] {reason} -> {val}")
    return float(val)


def compute_delta(data, bins, scale_factor=1.0):
    return (data[bins:] - data[:-bins]) * scale_factor


def smooth_velocity(x, y, timestamps, bins, pixel_length_cm_per_px):
    delta_x = compute_delta(x, bins, pixel_length_cm_per_px)
    delta_y = compute_delta(y, bins, pixel_length_cm_per_px)
    delta_t = compute_delta(timestamps, bins)
    velocity = np.column_stack((delta_x / delta_t, delta_y / delta_t))
    speed = np.linalg.norm(velocity, axis=1)
    return velocity, speed, delta_t


def ang_cal(vec_1, vec_2):
    dot_pro = np.dot(vec_1, vec_2)
    mod_1 = np.sqrt(np.dot(vec_1, vec_1))
    mod_2 = np.sqrt(np.dot(vec_2, vec_2))
    if mod_1 == 0 or mod_2 == 0:
        return 0
    cos = dot_pro / (mod_1 * mod_2)
    if np.isnan(cos):
        return np.nan
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(round(cos, 1))


def clws_delta_phi(vec_1, vec_2, vec_0=[1, 0]):
    vec_0 = np.array(vec_0)
    agl_1 = ang_cal(vec_1, vec_2)
    agl_2 = ang_cal(vec_0, vec_1)
    agl_3 = ang_cal(vec_0, vec_2)
    if agl_1 == 0:
        agl_1 = 0
    elif agl_1 == np.pi:
        agl_1 = np.pi
    elif vec_1[1] >= 0 and vec_2[1] >= 0:
        agl_1 = (-1) * np.abs(agl_1) if (agl_3 - agl_2) > 0 else np.abs(agl_1)
    elif vec_1[1] <= 0 and vec_2[1] <= 0:
        agl_1 = np.abs(agl_1) if (agl_3 - agl_2) > 0 else (-1) * np.abs(agl_1)
    elif vec_1[1] >= 0 and vec_2[1] <= 0:
        agl_1 = (-1) * np.abs(agl_1) if (agl_3 + agl_2) > np.pi else np.abs(agl_1)
    elif vec_1[1] <= 0 and vec_2[1] >= 0:
        agl_1 = (-1) * np.abs(agl_1) if (agl_3 + agl_2) < np.pi else np.abs(agl_1)
    return agl_1 * 180 / np.pi


def calc_concentration(x_pixel):
    return minNaCl + x_pixel * (maxNaCl - minNaCl) / WIDTH_REAL


def compute_ctx_left_positive(velocity_spd, deadzone_speed=None):
    vx = velocity_spd[:, 0]
    mod_v = np.linalg.norm(velocity_spd, axis=1)
    ctx = np.divide(-vx, mod_v, out=np.zeros_like(vx), where=(mod_v != 0))
    if deadzone_speed is not None and deadzone_speed > 0:
        ctx = ctx.copy()
        ctx[mod_v < deadzone_speed] = 0.0
    return ctx


def speed_change_scaled(timestep, total_steps):
    change_duration = total_steps / 3.0
    if timestep < change_duration:
        return INITIAL_SPEED_RAW - (INITIAL_SPEED_RAW - FINAL_SPEED_RAW) * timestep / change_duration
    return FINAL_SPEED_RAW


def is_single_concentration_condition(condition_name):
    return condition_name in ["3g-1gNa-1h", "3g-1gNa-4h"]


def is_probabilistic_model(model_name):
    return ("KRW" in model_name) or ("full" in model_name)


def deterministic_choose_high_probability(w_low, w_high, tie_eps=DETERMINISTIC_TIE_EPS):
    if w_high > w_low + tie_eps:
        return 1.0
    if w_low > w_high + tie_eps:
        return 0.0
    return 0.5


def stable_softplus(x):
    x = float(x)
    return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))


def reward_transform(selected_u):
    r = stable_softplus(selected_u)
    return float(np.clip(r, REWARD_CLIP[0], REWARD_CLIP[1]))


def normal_pdf_over_cdf(z):
    z = float(np.clip(z, -50.0, 50.0))
    log_pdf = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
    log_cdf = float(log_ndtr(z))
    return float(np.exp(np.clip(log_pdf - log_cdf, -50.0, 50.0)))


def signs_opposite(a, b, eps=CTX_SIGN_EPS):
    if abs(a) <= eps or abs(b) <= eps:
        return False
    return (a * b) < 0


def sem(x):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / math.sqrt(len(arr)))


# =========================
# TARGET SELECTION
# =========================
def choose_target_rw_hard(w_low, w_high, rng=np.random):
    p_high = deterministic_choose_high_probability(w_low, w_high)
    if p_high == 1.0:
        select_dist = 1
    elif p_high == 0.0:
        select_dist = 0
    else:
        select_dist = int(rng.rand() < 0.5)

    selected_u = w_high if select_dist == 1 else w_low
    return {
        "select_dist": int(select_dist),
        "selected_U": float(selected_u),
        "U_low_effective": float(w_low),
        "U_high_effective": float(w_high),
        "p_high": float(p_high),
        "sigma_delta": 0.0,
    }


def choose_target_krw_cdf_conditional_value(w_low, w_high, c_low, c_high, rng=np.random):
    c_low = max(float(c_low), 0.0)
    c_high = max(float(c_high), 0.0)
    sigma_delta = math.sqrt(c_low * c_low + c_high * c_high)

    if sigma_delta <= CHOICE_ZERO_VARIANCE_EPS:
        return choose_target_rw_hard(w_low, w_high, rng=rng)

    alpha = (w_high - w_low) / sigma_delta
    p_high = float(norm.cdf(np.clip(alpha, -12.0, 12.0)))
    p_high = float(np.clip(p_high, 0.0, 1.0))
    select_dist = int(rng.rand() < p_high)

    if select_dist == 1:
        lam = normal_pdf_over_cdf(alpha)
        selected_u = w_high + (c_high * c_high / sigma_delta) * lam
        u_high_eff = selected_u
        u_low_eff = w_low - (c_low * c_low / sigma_delta) * lam
    else:
        lam = normal_pdf_over_cdf(-alpha)
        selected_u = w_low + (c_low * c_low / sigma_delta) * lam
        u_low_eff = selected_u
        u_high_eff = w_high - (c_high * c_high / sigma_delta) * lam

    selected_u = float(np.clip(selected_u, SELECTED_U_CLIP[0], SELECTED_U_CLIP[1]))
    u_low_eff = float(np.clip(u_low_eff, SELECTED_U_CLIP[0], SELECTED_U_CLIP[1]))
    u_high_eff = float(np.clip(u_high_eff, SELECTED_U_CLIP[0], SELECTED_U_CLIP[1]))

    return {
        "select_dist": int(select_dist),
        "selected_U": selected_u,
        "U_low_effective": u_low_eff,
        "U_high_effective": u_high_eff,
        "p_high": float(p_high),
        "sigma_delta": float(sigma_delta),
    }


def choose_target_and_value(w_low, w_high, c_low, c_high, model_name, rng=np.random):
    if is_probabilistic_model(model_name):
        return choose_target_krw_cdf_conditional_value(w_low, w_high, c_low, c_high, rng=rng)
    return choose_target_rw_hard(w_low, w_high, rng=rng)


def selected_option_uncertainty(params, select_dist, model_name):
    if not is_probabilistic_model(model_name):
        return 0.0
    return float(params.get("c1", 0.0) if select_dist == 0 else params.get("c2", 0.0))


# =========================
# DATA PREP
# =========================
def select_dfs(condition_dict, target_condition, df_selection):
    try:
        all_dfs = condition_dict[target_condition]
    except KeyError:
        return [], []
    if df_selection is None:
        return all_dfs, list(range(len(all_dfs)))
    if isinstance(df_selection, int):
        n = min(df_selection, len(all_dfs))
        return all_dfs[:n], list(range(n))
    if isinstance(df_selection, (list, tuple)):
        idxs = [int(i) for i in df_selection if 0 <= int(i) < len(all_dfs)]
        return [all_dfs[i] for i in idxs], idxs
    raise TypeError("df_selection must be None/int/list/tuple")


def merge_dfs_with_unique_ids(dfs, idxs):
    merged = []
    for i, df_obj in zip(idxs, dfs):
        df_copy = df_obj.copy()
        df_copy["orig_id"] = df_copy["ID"]
        df_copy["ID"] = df_copy["ID"].astype(str) + f"_df{i}"
        df_copy["df_index"] = i
        merged.append(df_copy)
    if not merged:
        return pd.DataFrame()
    return pd.concat(merged, ignore_index=True)


def compute_worm_summary_from_traj(real_traj_df):
    rows = []
    for worm_id, group in real_traj_df.groupby("ID"):
        g = group.dropna(subset=["x", "y"])
        if len(g) > 1:
            start = g.iloc[0][["x", "y"]].values.astype(float)
            end = g.iloc[-1][["x", "y"]].values.astype(float)
            nd = float(np.linalg.norm(end - start))
            mean_mid = float(np.nanmean(np.abs(g["x"].values.astype(float) - MIDLINE_X)))
        else:
            nd = 0.0
            mean_mid = 0.0
        rows.append({
            "ID": worm_id,
            "net_displacement": nd,
            "mean_dist_to_midline": mean_mid,
        })
    return pd.DataFrame(rows)


def prepare_one_df(df_in, df_idx, max_timestamp=14400, speed_bins=50, add_id_suffix=True):
    d = df_in.copy()
    if add_id_suffix:
        d["ID"] = d["ID"].astype(str) + f"_df{df_idx}"
    if "Timestamp" in d.columns:
        n0 = len(d)
        d = d[d["Timestamp"] <= max_timestamp]
        print(f"[INFO][{df_idx}] truncation: {len(d)}/{n0}")

    required = ["speed", "agl_velocity", "CTX"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing real-data columns: {missing}")

    real_df = d[required].copy()
    real_df["abs_agl_velocity"] = np.abs(real_df["agl_velocity"].values.astype(float))
    real_df = real_df[["speed"] + FRAME_FEATURES].dropna()

    x_col = "X" if "X" in d.columns else "x"
    y_col = "Y" if "Y" in d.columns else "y"
    real_traj_df = d[[x_col, y_col, "ID"]].rename(columns={x_col: "x", y_col: "y"}).dropna()
    worm_summary = compute_worm_summary_from_traj(real_traj_df)

    speed_profile = None
    if "speed" in d.columns and "Timestamp" in d.columns:
        max_ts = d["Timestamp"].max()
        nt = d["Timestamp"] / max_ts if max_ts > 0 else d["Timestamp"]
        bins = np.linspace(0, 1, speed_bins + 1)
        centers = (bins[:-1] + bins[1:]) / 2
        means = []
        for i in range(speed_bins):
            m = (nt >= bins[i]) & (nt < bins[i + 1])
            means.append(d.loc[m, "speed"].mean() if m.sum() > 0 else np.nan)
        means = np.array(means, dtype=float)
        if np.isnan(means).any():
            ok = ~np.isnan(means)
            if ok.sum() >= 2:
                f = interp1d(centers[ok], means[ok], kind="linear", fill_value="extrapolate")
                means = f(centers)
            elif ok.sum() == 1:
                means = np.full_like(centers, means[ok][0], dtype=float)
            else:
                means = np.array([speed_change_scaled(i, speed_bins) for i in range(speed_bins)], dtype=float)
        speed_profile = (centers, means)

    deadzone_speed = None
    if USE_CTX_DEADZONE and "speed" in d.columns:
        s = d["speed"].dropna().values
        if len(s) > 10:
            deadzone_speed = max(np.quantile(s, CTX_DEADZONE_SPEED_Q), CTX_DEADZONE_MIN)

    return real_df, real_traj_df, speed_profile, deadzone_speed, worm_summary


# =========================
# WV DIRECTION
# =========================
def WV_direction_probe_stochastic(cur_x, rotate_angle, sample_conc, v_px_s, dt, temp=WV_TEMPERATURE):
    eps = math.radians(WV_TRY_EPS_DEG)
    dxp = v_px_s * dt * math.cos(rotate_angle + eps)
    dxm = v_px_s * dt * math.cos(rotate_angle - eps)
    newxp = cur_x + dxp
    newxm = cur_x + dxm
    dRp = abs(sample_conc - calc_concentration(cur_x)) - abs(sample_conc - calc_concentration(newxp))
    dRm = abs(sample_conc - calc_concentration(cur_x)) - abs(sample_conc - calc_concentration(newxm))
    p_turn_right = 1.0 / (1.0 + np.exp((dRm - dRp) / temp))
    return 1.0 if np.random.rand() < p_turn_right else -1.0


# =========================
# SIMULATION
# =========================
def simulate_single_particle(params, model_name, total_steps, dt, speed_profile, deadzone_speed, seed=None,
                             return_xy=False):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    w1, w2 = float(params["w1"]), float(params["w2"])
    c1 = float(params.get("c1", 0.0))
    c2 = float(params.get("c2", 0.0))
    amp_scale = float(params.get("amp_scale", 1.0))

    is_switch = "switch" in model_name
    is_ar1 = "AR1" in model_name

    # Switch parameters.
    a0 = float(params.get("a0", -1.2))
    a1 = float(params.get("a1", 5.0))  # legacy
    a_dr = float(params.get("a_dr", a1))
    a_reward = float(params.get("a_reward", 0.0))
    a_interaction = float(params.get("a_interaction", 0.0))
    k_wv_deg = float(params.get("k_wv_deg", params.get("k_wv", 1.0)))
    sigma_wv_deg = float(params.get("sigma_wv_deg", 5.0))
    sigma_pir_deg = float(params.get("sigma_pir_deg", DEFAULT_SIGMA_PIR_DEG))
    pir_mean_angle = float(params.get("pir_mean_angle", PIR_MEAN_ANGLE))
    min_pir_steps = int(params.get("min_pir_steps", MIN_PIR_STEPS))

    cur_x = random.gauss(MIDLINE_X, SIGMA)
    cur_y = random.gauss((HEIGHT_REAL + 2 * BIN_SIZE) / 2.0, SIGMA)
    rotate_angle = math.radians(random.random() * 360)

    grad = (maxNaCl - minNaCl) / WIDTH_REAL
    lo_x, hi_x = BIN_SIZE, WIDTH_REAL + BIN_SIZE
    lo_y, hi_y = BIN_SIZE, HEIGHT_REAL + BIN_SIZE

    if speed_profile is not None:
        centers, means = speed_profile
        fspd = interp1d(centers, means, kind="linear", fill_value=(means[0], means[-1]), bounds_error=False)
        get_speed = lambda t: float(fspd(t / total_steps))
    else:
        get_speed = lambda t: speed_change_scaled(t, total_steps)

    xs, ys = [], []
    state = 0
    ar_turn = 0.0
    pir_left = 0

    DECISION_INTERVAL_SEC = 1.5
    DECISION_STEPS = max(1, int(DECISION_INTERVAL_SEC / dt))
    decision_timer = 0

    select_dist = 0
    sample_conc = stimulus[select_dist]
    selected_U = w1
    selected_reward = reward_transform(selected_U)
    p45 = 0.0
    selected_uncertainty = 0.0
    U_0_eff, U_45_eff = w1, w2
    sigma_delta = 0.0

    pir_direction = 1
    pir_step_angle = 0.0
    pause_timer = 0

    if return_xy:
        select_dist_trace, selected_U_trace, selected_reward_trace = [], [], []
        p45_trace, state_trace, selected_uncertainty_trace = [], [], []
        U0_trace, U45_trace, sigma_delta_trace = [], [], []

    for t in range(total_steps):
        current_conc = calc_concentration(cur_x)

        if decision_timer <= 0:
            choice = choose_target_and_value(
                w_low=w1,
                w_high=w2,
                c_low=c1,
                c_high=c2,
                model_name=model_name,
                rng=np.random,
            )
            select_dist = choice["select_dist"]
            selected_U = choice["selected_U"]
            U_0_eff = choice["U_low_effective"]
            U_45_eff = choice["U_high_effective"]
            p45 = choice["p_high"]
            sigma_delta = choice["sigma_delta"]
            selected_reward = reward_transform(selected_U)
            selected_uncertainty = selected_option_uncertainty({"c1": c1, "c2": c2}, select_dist, model_name)
            sample_conc = stimulus[select_dist]
            decision_timer = DECISION_STEPS

        v_cm_s = max(get_speed(t), 0.0)

        # Retain original small non-fitted transient pause. This is not a fitted third behavior state.
        if pause_timer > 0:
            v_cm_s = 0.0
            pause_timer -= 1
        elif random.random() < 0.005:
            pause_timer = random.randint(10, 30)
            v_cm_s = 0.0

        v_px_s = v_cm_s * PX_PER_CM
        dx = v_px_s * dt * math.cos(rotate_angle)
        dy = v_px_s * dt * math.sin(rotate_angle)
        new_x = cur_x + dx
        new_y = cur_y + dy

        if new_x <= lo_x or new_x >= hi_x:
            rotate_angle = math.pi - rotate_angle + math.radians(random.gauss(0, 30))
            new_x = np.clip(new_x, lo_x, hi_x)
            ar_turn = 0.0
        if new_y <= lo_y or new_y >= hi_y:
            rotate_angle = -rotate_angle + math.radians(random.gauss(0, 30))
            new_y = np.clip(new_y, lo_y, hi_y)
            ar_turn = 0.0

        delta_R = abs(sample_conc - current_conc) - abs(sample_conc - calc_concentration(new_x))
        dr = delta_R / max(dt, 1e-9)

        if not is_switch:
            AMP = amp_scale / (max(v_px_s, 1e-9) * grad + 1e-9)
            s = safe_sigmoid(dr * AMP)
            logit = selected_reward * (s - 0.5) * 2.0
            p_keep = safe_sigmoid(logit)

            if np.random.rand() < p_keep:
                dtheta = math.radians(random.gauss(0, BUZZ))
            else:
                dtheta = math.radians(random.gauss(0, 90.0))

            if is_ar1:
                rho_ar = float(params.get("rho_ar", 0.95))
                sigma_ar = float(params.get("sigma_ar", 1.0))
                rho_ar = min(max(rho_ar, 0.0), 0.999)
                ar_turn = rho_ar * ar_turn + math.sqrt(max(1.0 - rho_ar * rho_ar, 0.0)) * dtheta
                rotate_angle += sigma_ar * ar_turn
            else:
                rotate_angle += dtheta
        else:
            if state == 0:
                dr_normalized = dr / (max(v_px_s, 1e-9) * grad + 1e-9)
                dr_signal = -dr_normalized
                if USE_SEPARABLE_PR_LOGIT:
                    logit_pr = a0 + a_dr * dr_signal + a_reward * selected_reward + a_interaction * dr_signal * selected_reward
                    p_wv2pir = safe_sigmoid(logit_pr)
                else:
                    if USE_REWARD_GATED_PR_LEGACY:
                        p_wv2pir = safe_sigmoid(a0 + a1 * dr_signal * selected_reward)
                    else:
                        p_wv2pir = safe_sigmoid(a0 + a1 * dr_signal)

                if np.random.rand() < p_wv2pir:
                    state = 1
                    pir_left = int(round(np.random.normal(min_pir_steps, PIR_STEPS_STD)))
                    pir_left = max(10, min(70, pir_left))
                    pir_direction = random.choice([1, -1])
                    pir_total_angle = random.gauss(pir_mean_angle, sigma_pir_deg)
                    pir_total_angle = max(120.0, min(240.0, pir_total_angle))
                    pir_step_angle = math.radians(pir_total_angle / pir_left)
                else:
                    sign = WV_direction_probe_stochastic(cur_x, rotate_angle, sample_conc, v_px_s, dt,
                                                         temp=WV_TEMPERATURE)
                    # k_wv_deg is explicitly in degrees per frame per reward unit.
                    dth = sign * math.radians(k_wv_deg * selected_reward) + math.radians(random.gauss(0, sigma_wv_deg))
                    rotate_angle += dth
            elif state == 1:
                dth = pir_direction * pir_step_angle
                rotate_angle += dth
                if pir_left > 0:
                    pir_left -= 1
                if pir_left == 0:
                    state = 0

        xs.append(new_x)
        ys.append(HEIGHT_REAL - new_y)
        if return_xy:
            select_dist_trace.append(select_dist)
            selected_U_trace.append(selected_U)
            selected_reward_trace.append(selected_reward)
            p45_trace.append(p45)
            state_trace.append(state)
            selected_uncertainty_trace.append(selected_uncertainty)
            U0_trace.append(U_0_eff)
            U45_trace.append(U_45_eff)
            sigma_delta_trace.append(sigma_delta)

        cur_x, cur_y = new_x, new_y
        decision_timer -= 1

    xs = np.array(xs)
    ys = np.array(ys)
    ts = np.arange(total_steps) * dt

    bins_spd, bins_agl, hlf = 2, 20, 10
    vel_spd, speed_spd, _ = smooth_velocity(xs, ys, ts, bins_spd, PIXEL_LENGTH_CM_PER_PX)
    vel_spd = np.pad(vel_spd, ((1, 1), (0, 0)), constant_values=np.nan)
    speed_spd = np.pad(speed_spd, (1, 1), constant_values=np.nan)

    vel_agl, _, t_agl = smooth_velocity(xs, ys, ts, bins_agl, PIXEL_LENGTH_CM_PER_PX)
    agl = np.zeros(len(vel_agl))
    valid = np.arange(hlf, len(t_agl) - hlf)
    for i in valid:
        dphi = clws_delta_phi(vel_agl[i - hlf], vel_agl[i + hlf])
        dt_loc = t_agl[i] * hlf
        agl[i] = dphi / dt_loc
    agl = np.pad(agl, (hlf, hlf), constant_values=np.nan)

    ctx = compute_ctx_left_positive(vel_spd, deadzone_speed=deadzone_speed)
    net_displacement = float(np.linalg.norm([xs[-1] - xs[0], ys[-1] - ys[0]]))
    mean_dist_to_midline = float(np.nanmean(np.abs(xs - MIDLINE_X)))

    out = pd.DataFrame({
        "speed": speed_spd,
        "agl_velocity": agl,
        "abs_agl_velocity": np.abs(agl),
        "CTX": ctx,
        "net_displacement": net_displacement,
        "mean_dist_to_midline": mean_dist_to_midline,
    })
    if return_xy:
        out["x"], out["y"] = xs, ys
        out["timestep"] = np.arange(total_steps)
        out["select_dist"] = select_dist_trace
        out["selected_U"] = selected_U_trace
        out["selected_reward"] = selected_reward_trace
        out["p_high"] = p45_trace
        out["state"] = state_trace
        out["selected_uncertainty"] = selected_uncertainty_trace
        out["U_low_effective"] = U0_trace
        out["U_high_effective"] = U45_trace
        out["sigma_delta"] = sigma_delta_trace
    return out.dropna()


def simulate_population(params, model_name, num_worms, total_steps, dt, speed_profile, deadzone_speed,
                        n_jobs=1, seed_base=1234, return_xy=False):
    seeds = [seed_base + i for i in range(int(num_worms))]

    def one(k, s):
        d = simulate_single_particle(params, model_name, total_steps, dt, speed_profile, deadzone_speed,
                                     seed=s, return_xy=return_xy)
        d = d.copy()
        d["ID"] = k
        return d

    if n_jobs > 1:
        arr = Parallel(n_jobs=n_jobs)(delayed(one)(k, s) for k, s in enumerate(seeds))
    else:
        arr = [one(k, s) for k, s in enumerate(seeds)]
    return pd.concat(arr, ignore_index=True).dropna()


# =========================
# LOSS
# =========================
class FeatureStandardizer:
    def __init__(self, fit_real_df, fit_worm_summary,
                 target_real_df=None, target_worm_summary=None):
        self.fit_real_df = fit_real_df.copy()
        self.fit_worm_summary = fit_worm_summary.copy()
        self.target_real_df = fit_real_df.copy() if target_real_df is None else target_real_df.copy()
        self.target_worm_summary = fit_worm_summary.copy() if target_worm_summary is None else target_worm_summary.copy()

        self.frame_scaler = StandardScaler()
        fit_frame = self.fit_real_df[FRAME_FEATURES].dropna().values
        target_frame = self.target_real_df[FRAME_FEATURES].dropna().values
        if len(fit_frame) < 2 or len(target_frame) < 2:
            raise ValueError("Not enough frame-level real data for standardization")
        self.frame_scaler.fit(fit_frame)
        self.real_frame_scaled = self.frame_scaler.transform(target_frame)

        self.worm_scaler = StandardScaler()
        fit_worm = self.fit_worm_summary[WORM_FEATURES].dropna().values
        target_worm = self.target_worm_summary[WORM_FEATURES].dropna().values
        if len(fit_worm) < 1 or len(target_worm) < 1:
            raise ValueError("Not enough worm-level real data for standardization")
        self.worm_scaler.fit(fit_worm)
        self.real_worm_scaled = self.worm_scaler.transform(target_worm)

        self.real_ctx_raw = self.target_real_df["CTX"].dropna().values.astype(float)
        self.real_ctx_mean_raw = float(np.nanmean(self.real_ctx_raw)) if len(self.real_ctx_raw) else 0.0
        self.real_ctx_std_raw = float(np.nanstd(self.real_ctx_raw)) if len(self.real_ctx_raw) else 1.0
        self.real_ctx_std_raw = max(self.real_ctx_std_raw, 1e-9)

    def transform(self, sim_df):
        sim_frame_df = sim_df[FRAME_FEATURES].dropna()
        if sim_frame_df.empty:
            raise ValueError("empty simulated frame feature table")
        scaled_frame = self.frame_scaler.transform(sim_frame_df.values)

        sim_worm = sim_df.groupby("ID")[WORM_FEATURES].first().dropna()
        if sim_worm.empty:
            raise ValueError("empty simulated worm feature table")
        scaled_worm = self.worm_scaler.transform(sim_worm.values)
        return scaled_frame, scaled_worm, sim_worm


def compute_loss(stdz, sim_df, return_components=False):
    if sim_df is None or sim_df.empty:
        val = safe_large_loss("empty sim")
        return (val, {"total_loss": val}) if return_components else val
    try:
        scaled_frame, scaled_worm, sim_worm = stdz.transform(sim_df)
        components = {}
        loss = 0.0

        for i, f in enumerate(FRAME_FEATURES):
            d = wasserstein_distance(stdz.real_frame_scaled[:, i], scaled_frame[:, i])
            weighted = FEATURE_WEIGHTS.get(f, 1.0) * d
            components[f"wasserstein_{f}"] = float(d)
            components[f"weighted_{f}"] = float(weighted)
            loss += weighted

        for i, f in enumerate(WORM_FEATURES):
            d = wasserstein_distance(stdz.real_worm_scaled[:, i], scaled_worm[:, i])
            weighted = FEATURE_WEIGHTS.get(f, 1.0) * d
            components[f"wasserstein_{f}"] = float(d)
            components[f"weighted_{f}"] = float(weighted)
            loss += weighted

        real_ctx_mean = stdz.real_ctx_mean_raw
        sim_ctx_mean = float(np.nanmean(sim_df["CTX"].values.astype(float)))
        mean_diff_z = abs(sim_ctx_mean - real_ctx_mean) / stdz.real_ctx_std_raw
        opposite = signs_opposite(real_ctx_mean, sim_ctx_mean, eps=CTX_SIGN_EPS)
        multiplier = CTX_OPPOSITE_SIGN_MULTIPLIER if opposite else 1.0
        ctx_mean_penalty = CTX_MEAN_PENALTY_WEIGHT * multiplier * mean_diff_z
        loss += ctx_mean_penalty

        components["real_CTX_mean_raw"] = float(real_ctx_mean)
        components["sim_CTX_mean_raw"] = float(sim_ctx_mean)
        components["CTX_mean_diff_z"] = float(mean_diff_z)
        components["CTX_opposite_sign"] = bool(opposite)
        components["CTX_mean_penalty_multiplier"] = float(multiplier)
        components["CTX_mean_penalty"] = float(ctx_mean_penalty)

        # Raw diagnostic means.
        components["real_abs_agl_velocity_mean_raw"] = float(stdz.target_real_df["abs_agl_velocity"].mean())
        components["sim_abs_agl_velocity_mean_raw"] = float(sim_df["abs_agl_velocity"].mean())
        for f in WORM_FEATURES:
            components[f"real_{f}_mean_raw"] = float(stdz.target_worm_summary[f].mean())
            components[f"sim_{f}_mean_raw"] = float(sim_worm[f].mean())

        if not np.isfinite(loss):
            val = safe_large_loss("non-finite")
            return (val, {"total_loss": val}) if return_components else val

        components["data_loss"] = float(loss)
        components["total_loss"] = float(loss)
        return (float(loss), components) if return_components else float(loss)
    except Exception as e:
        val = safe_large_loss(str(e))
        return (val, {"total_loss": val, "error": str(e)}) if return_components else val


# =========================
# PARAMETERIZATION & OPTIMIZATION
# =========================
def search_space(model_name, condition_name="1gNa-end(8x)"):
    is_prob = is_probabilistic_model(model_name)
    is_switch = "switch" in model_name
    is_single = is_single_concentration_condition(condition_name)

    spaces = [Real(np.log(0.5), np.log(5.0), name="log_w1")]
    if not is_single:
        spaces.append(Real(np.log(0.5), np.log(5.0), name="log_w2"))

    if is_prob:
        spaces.append(Real(np.log(C_MIN), np.log(C_MAX), name="log_c1"))
        if not is_single:
            spaces.append(Real(np.log(C_MIN), np.log(C_MAX), name="log_c2"))

    if not is_switch:
        spaces.append(Real(np.log(1.0), np.log(100.0), name="log_amp_scale"))

    if is_switch:
        spaces.append(Real(-6.0, -1.0, name="a0"))
        if USE_SEPARABLE_PR_LOGIT:
            spaces += [
                Real(np.log(0.1), np.log(50.0), name="log_a_dr"),
                Real(-3.0, 3.0, name="a_reward"),
                Real(-10.0, 10.0, name="a_interaction"),
            ]
        else:
            spaces.append(Real(np.log(0.1), np.log(100.0), name="log_a1"))
        spaces += [
            Real(np.log(0.1), np.log(20.0), name="log_k_wv_deg"),
            Real(np.log(1.0), np.log(15.0), name="log_sig_wv"),
            Real(np.log(15.0), np.log(45.0), name="log_sigma_pir_deg"),
        ]
    return spaces


def decode_params(x, model_name, condition_name="1gNa-end(8x)"):
    is_prob = is_probabilistic_model(model_name)
    is_switch = "switch" in model_name
    is_single = is_single_concentration_condition(condition_name)

    idx = 0
    w1 = float(np.exp(x[idx])); idx += 1
    if is_single:
        w2 = 0.0
    else:
        w2 = float(np.exp(x[idx])); idx += 1

    p = {"w1": w1, "w2": w2}

    if is_prob:
        c1 = float(np.exp(x[idx])); idx += 1
        if is_single:
            c2 = 0.0
        else:
            c2 = float(np.exp(x[idx])); idx += 1
        p.update({
            "c1": c1,
            "c2": c2,
            "c_decision": math.sqrt(c1 * c1 + c2 * c2),
            "uncertainty_mode": "cdf_conditional_c1c2",
        })
    else:
        p.update({"c1": 0.0, "c2": 0.0, "c_decision": 0.0, "uncertainty_mode": "deterministic"})

    if not is_switch:
        p["amp_scale"] = float(np.exp(x[idx])); idx += 1
    else:
        p["amp_scale"] = 1.0

    if is_switch:
        a0 = float(x[idx]); idx += 1
        if USE_SEPARABLE_PR_LOGIT:
            a_dr = float(np.exp(x[idx])); idx += 1
            a_reward = float(x[idx]); idx += 1
            a_interaction = float(x[idx]); idx += 1
            p.update({"a0": a0, "a_dr": a_dr, "a_reward": a_reward,
                      "a_interaction": a_interaction, "a1": a_dr})
        else:
            a1 = float(np.exp(x[idx])); idx += 1
            p.update({"a0": a0, "a1": a1, "a_dr": a1,
                      "a_reward": 0.0, "a_interaction": 0.0})

        k_wv_deg = float(np.exp(x[idx])); idx += 1
        sigma_wv_deg = float(np.exp(x[idx])); idx += 1
        sigma_pir_deg = float(np.exp(x[idx])); idx += 1
        p.update({
            "k_wv_deg": k_wv_deg,
            "k_wv": k_wv_deg,  # backward-compatible label, but interpreted as degrees in this script
            "sigma_wv_deg": sigma_wv_deg,
            "sigma_pir_deg": sigma_pir_deg,
            "pir_mean_angle": PIR_MEAN_ANGLE,
            "min_pir_steps": MIN_PIR_STEPS,
        })
    else:
        p.update({
            "a0": 0.0,
            "a1": 5.0,
            "a_dr": 5.0,
            "a_reward": 0.0,
            "a_interaction": 0.0,
            "k_wv_deg": 1.0,
            "k_wv": 1.0,
            "sigma_wv_deg": 5.0,
            "sigma_pir_deg": DEFAULT_SIGMA_PIR_DEG,
            "pir_mean_angle": PIR_MEAN_ANGLE,
            "min_pir_steps": MIN_PIR_STEPS,
        })
    return p


def num_model_parameters(model_name, condition_name):
    return len(search_space(model_name, condition_name))


def parameter_regularization(p, model_name):
    regularization = 0.0

    if is_probabilistic_model(model_name):
        c1 = float(p.get("c1", 0.0))
        c2 = float(p.get("c2", 0.0))
        w1_val = max(abs(float(p.get("w1", 1.0))), 0.1)
        w2_val = max(abs(float(p.get("w2", 1.0))), 0.1)
        if c1 > w1_val * 2.0:
            regularization += 0.1 * (c1 - w1_val * 2.0) ** 2
        if c2 > w2_val * 2.0:
            regularization += 0.1 * (c2 - w2_val * 2.0) ** 2

    if "switch" in model_name:
        a_dr = float(p.get("a_dr", p.get("a1", 0.0)))
        if a_dr > 40.0:
            regularization += 0.03 * (a_dr - 40.0) ** 2
        if a_dr < 0.2:
            regularization += 0.05 * (0.2 - a_dr) ** 2

        a_reward = float(p.get("a_reward", 0.0))
        a_interaction = float(p.get("a_interaction", 0.0))
        regularization += 0.01 * (a_reward ** 2)
        regularization += 0.002 * (a_interaction ** 2)

        k_wv_deg = float(p.get("k_wv_deg", p.get("k_wv", 1.0)))
        if k_wv_deg < 0.2:
            regularization += 0.05 * (0.2 - k_wv_deg) ** 2
        if k_wv_deg > 15.0:
            regularization += 0.05 * (k_wv_deg - 15.0) ** 2

        sigma_wv = float(p.get("sigma_wv_deg", 5.0))
        if sigma_wv > 12.0:
            regularization += 0.05 * (sigma_wv - 12.0) ** 2
        if sigma_wv < 1.5:
            regularization += 0.05 * (1.5 - sigma_wv) ** 2

        sigma_pir = float(p.get("sigma_pir_deg", 35.0))
        if sigma_pir > 42.0:
            regularization += 0.1 * (sigma_pir - 42.0) ** 2
        if sigma_pir < 15.0:
            regularization += 0.1 * (15.0 - sigma_pir) ** 2
        if sigma_pir < sigma_wv * 3.0:
            regularization += 0.1 * (sigma_wv * 3.0 - sigma_pir) ** 2
    else:
        amp_scale = float(p.get("amp_scale", 1.0))
        if amp_scale > 50.0:
            regularization += 0.1 * (amp_scale - 50.0) ** 2
        if amp_scale < 2.0:
            regularization += 0.1 * (2.0 - amp_scale) ** 2

    return float(regularization)


def evaluate_params(p, model_name, stdz, speed_profile, deadzone_speed,
                    num_worms, repeats, seed_base=10000,
                    include_regularization=True, return_components=False):
    losses = []
    comp_rows = []
    for j in range(int(repeats)):
        sim = simulate_population(
            p, model_name, int(num_worms), MAX_TIMESTAMP, DT_REAL,
            speed_profile, deadzone_speed, n_jobs=N_JOBS,
            seed_base=seed_base + j * 100, return_xy=False
        )
        if return_components:
            data_loss, comps = compute_loss(stdz, sim, return_components=True)
            comp_rows.append(comps)
        else:
            data_loss = compute_loss(stdz, sim, return_components=False)
        losses.append(float(data_loss))

    data_loss_mean = float(np.mean(losses))
    reg = parameter_regularization(p, model_name) if include_regularization else 0.0
    total = data_loss_mean + reg

    if not return_components:
        return total

    avg_comps = {}
    if comp_rows:
        keys = sorted(set().union(*[c.keys() for c in comp_rows]))
        for k in keys:
            vals = [c.get(k, np.nan) for c in comp_rows]
            if all(isinstance(v, (bool, np.bool_)) for v in vals if pd.notna(v)):
                avg_comps[k] = bool(np.nanmean(vals) >= 0.5)
            else:
                try:
                    avg_comps[k] = float(np.nanmean(vals))
                except Exception:
                    avg_comps[k] = vals[0]
    avg_comps["data_loss_mean"] = data_loss_mean
    avg_comps["data_loss_sd"] = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
    avg_comps["regularization"] = float(reg)
    avg_comps["objective_loss"] = float(total)
    return total, avg_comps


def objective_from_x(x, model_name, stdz, speed_profile, deadzone_speed, condition_name,
                     num_worms, repeats):
    p = decode_params(x, model_name, condition_name)
    return evaluate_params(p, model_name, stdz, speed_profile, deadzone_speed,
                           num_worms=num_worms, repeats=repeats,
                           seed_base=10000, include_regularization=True,
                           return_components=False)


def optimize_one_model(model_name, stdz, speed_profile, deadzone_speed,
                       condition_name, fit_config):
    space = search_space(model_name, condition_name)
    dim = len(space)
    print(f"  Optimizing {model_name} (dim={dim})")

    num_worms = int(fit_config["num_worms"])
    repeats = int(fit_config["objective_repeats"])
    random_iters = int(fit_config["random_search_iters"])
    n_calls = int(fit_config["n_calls_krw"] if is_probabilistic_model(model_name) else fit_config["n_calls_rw"])

    print(f"    Stage 1: random prescreen ({random_iters} iterations)")
    random_results = []
    for i in range(random_iters):
        x_random = []
        for s in space:
            v = s.rvs()
            if isinstance(v, (list, tuple, np.ndarray)):
                x_random.append(float(np.asarray(v).ravel()[0]))
            else:
                x_random.append(float(v))

        p_tmp = decode_params(x_random, model_name, condition_name)
        prescreen_loss = evaluate_params(
            p_tmp, model_name, stdz, speed_profile, deadzone_speed,
            num_worms=max(4, num_worms // 2), repeats=1,
            seed_base=12345 + i, include_regularization=True,
            return_components=False
        )
        random_results.append((float(prescreen_loss), x_random))

        if (i + 1) % 50 == 0:
            current_best = min(loss for loss, _ in random_results)
            print(f"      random {i + 1}/{random_iters}, best prescreen loss: {current_best:.4f}")

    random_results.sort(key=lambda z: z[0])

    # Recompute top-k random candidates under the full objective to avoid a lucky half-worm/seed artifact.
    top_k = min(RANDOM_TOPK_FULL_RECOMPUTE, len(random_results))
    recomputed = []
    for prescreen_loss, x_candidate in random_results[:top_k]:
        full_loss = objective_from_x(
            x_candidate, model_name, stdz, speed_profile, deadzone_speed, condition_name,
            num_worms=num_worms, repeats=repeats
        )
        recomputed.append((full_loss, prescreen_loss, x_candidate))
    recomputed.sort(key=lambda z: z[0])

    best_random_loss = float(recomputed[0][0])
    best_prescreen_loss = float(recomputed[0][1])
    best_random_x = recomputed[0][2]
    print(f"    Best prescreen loss among top-{top_k}: {best_prescreen_loss:.4f}")
    print(f"    Best random loss after full-objective recompute: {best_random_loss:.4f}")

    if gp_minimize is None:
        print("    [WARNING] scikit-optimize unavailable; returning best recomputed random candidate.")
        best = decode_params(best_random_x, model_name, condition_name)
        best["loss"] = best_random_loss
        return best

    print("    Stage 2: Bayesian optimization")
    res = gp_minimize(
        lambda x: objective_from_x(x, model_name, stdz, speed_profile, deadzone_speed, condition_name,
                                   num_worms=num_worms, repeats=repeats),
        space,
        x0=[best_random_x],
        y0=[best_random_loss],
        n_calls=n_calls,
        n_initial_points=max(5, min(10, random_iters // 20)),
        random_state=42,
        verbose=False,
        acq_func="EI",
    )

    best = decode_params(res.x, model_name, condition_name)
    best["loss"] = float(res.fun)
    print(f"    Final best objective loss: {best['loss']:.4f}")
    return best


def deterministic_to_probabilistic_params(p_det):
    p = dict(p_det)
    p["c1"] = 0.0
    p["c2"] = 0.0
    p["c_decision"] = 0.0
    p["uncertainty_mode"] = "nested_deterministic"
    return p


def final_fit_config():
    return {
        "num_worms": NUM_WORMS_OPT_FINAL,
        "objective_repeats": FINAL_OBJECTIVE_REPEATS,
        "random_search_iters": FINAL_RANDOM_SEARCH_ITERS,
        "n_calls_rw": FINAL_N_CALLS_RW,
        "n_calls_krw": FINAL_N_CALLS_KRW,
    }


def cv_fit_config():
    return {
        "num_worms": NUM_WORMS_OPT_CV,
        "objective_repeats": CV_OBJECTIVE_REPEATS,
        "random_search_iters": CV_RANDOM_SEARCH_ITERS,
        "n_calls_rw": CV_N_CALLS_RW,
        "n_calls_krw": CV_N_CALLS_KRW,
    }


# =========================
# PLOTTING
# =========================
def plot_real_trajectories(condition_name, real_traj_df, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for _, g in real_traj_df.groupby("ID"):
        ax.plot(g["x"], g["y"], lw=0.5, alpha=0.6, color="gray")
    ax.set_title(f"Real Trajectories: {condition_name}", fontsize=14)
    ax.set_xlim(0, WIDTH_REAL)
    ax.set_ylim(0, HEIGHT_REAL)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{condition_name}_real_trajectories.png"), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_real_features(condition_name, real_df, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, feat in enumerate(["speed", "agl_velocity", "abs_agl_velocity", "CTX"]):
        ax = axes[i]
        ax.hist(real_df[feat], bins=60, density=True, alpha=0.7, color="gray", edgecolor="black")
        ax.set_title(f"Real Data: {feat}", fontsize=12)
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{condition_name}_real_features.png"), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_traj_5panel(condition_name, real_traj_df, traj_dict, output_dir):
    fig, axes = plt.subplots(1, 1 + len(MODEL_NAMES), figsize=(6 * (1 + len(MODEL_NAMES)), 6))
    names = ["real"] + MODEL_NAMES
    for k, name in enumerate(names):
        ax = axes[k]
        if name == "real":
            for _, g in real_traj_df.groupby("ID"):
                ax.plot(g["x"], g["y"], lw=0.5, alpha=0.6, color=MODEL_COLORS[name])
        else:
            d = traj_dict[name]
            for _, g in d.groupby("ID"):
                ax.plot(g["x"], g["y"], lw=0.5, alpha=0.65, color=MODEL_COLORS.get(name, "black"))
        ax.set_title(name)
        ax.set_xlim(0, WIDTH_REAL)
        ax.set_ylim(0, HEIGHT_REAL)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{condition_name}_traj_5panel.png"), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_feature_4models(condition_name, real_df, sim_feat_dict, output_dir):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    det_family = [m for m in MODEL_NAMES if not is_probabilistic_model(m)]
    prob_family = [m for m in MODEL_NAMES if is_probabilistic_model(m)]
    for i, feat in enumerate(FRAME_FEATURES):
        ax = axes[i, 0]
        ax.hist(real_df[feat], bins=60, density=True, alpha=0.5, color="gray", label="Real")
        for m in det_family:
            ax.hist(sim_feat_dict[m][feat], bins=60, density=True, alpha=0.45,
                    color=MODEL_COLORS.get(m, None), label=m)
        ax.set_title(f"{feat}: real vs deterministic family")
        ax.legend()

        ax = axes[i, 1]
        ax.hist(real_df[feat], bins=60, density=True, alpha=0.5, color="gray", label="Real")
        for m in prob_family:
            ax.hist(sim_feat_dict[m][feat], bins=60, density=True, alpha=0.45,
                    color=MODEL_COLORS.get(m, None), label=m)
        ax.set_title(f"{feat}: real vs probabilistic family")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{condition_name}_feature_compare.png"), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_worm_feature_compare(condition_name, real_worm_summary, sim_traj_dict, output_dir, feature):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    data_to_plot = [real_worm_summary[feature].dropna().values]
    labels = ["Real"]
    colors = ["gray"]
    for name in MODEL_NAMES:
        if name in sim_traj_dict:
            sim_worm = sim_traj_dict[name].groupby("ID")[feature].first().dropna().values
            data_to_plot.append(sim_worm)
            labels.append(name)
            colors.append(MODEL_COLORS.get(name, "black"))
    if not data_to_plot:
        return
    parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(1)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"{feature} Comparison: {condition_name}", fontsize=12)
    ax.set_ylabel(f"{feature} (pixels)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{condition_name}_{feature}_compare.png"), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# =========================
# FITTING WORKFLOWS
# =========================
def prepare_condition_subset(condition_name, dfs, idxs, label):
    merged = merge_dfs_with_unique_ids(dfs, idxs)
    if merged.empty:
        raise ValueError(f"No data to prepare for {condition_name}, {label}")
    return prepare_one_df(
        merged, df_idx=f"{condition_name}_{label}",
        max_timestamp=MAX_TIMESTAMP, speed_bins=SPEED_BINS,
        add_id_suffix=False,
    )


def fit_models_on_dataset(condition_name, train_real_df, train_worm_summary,
                          speed_profile, deadzone_speed, output_dir, fit_config):
    stdz_train = FeatureStandardizer(train_real_df, train_worm_summary)
    best_params = {}
    rows = []
    for m in MODEL_NAMES:
        print(f"[STAGE][{condition_name}] optimize {m}")
        best = optimize_one_model(m, stdz_train, speed_profile, deadzone_speed,
                                  condition_name, fit_config)
        best_params[m] = best
        rows.append({
            "condition": condition_name,
            "model": m,
            "loss": best.get("loss", np.nan),
            "n_params": num_model_parameters(m, condition_name),
            **best,
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, f"{condition_name}_fit_4models_results.csv"), index=False)
    return best_params, pd.DataFrame(rows), stdz_train


def evaluate_best_params_on_target(condition_name, best_params, fit_real_df, fit_worm_summary,
                                   target_real_df, target_worm_summary,
                                   speed_profile, deadzone_speed,
                                   num_worms, repeats, seed_base, split_label):
    stdz_target = FeatureStandardizer(
        fit_real_df=fit_real_df,
        fit_worm_summary=fit_worm_summary,
        target_real_df=target_real_df,
        target_worm_summary=target_worm_summary,
    )
    rows = []
    for m in MODEL_NAMES:
        p_use = best_params[m]
        objective_loss, comps = evaluate_params(
            p_use, m, stdz_target, speed_profile, deadzone_speed,
            num_worms=num_worms, repeats=repeats,
            seed_base=seed_base, include_regularization=True,
            return_components=True
        )
        data_loss = comps.get("data_loss_mean", np.nan)
        rows.append({
            "condition": condition_name,
            "split": split_label,
            "model": m,
            "data_loss": data_loss,
            "objective_loss": objective_loss,
            "regularization": comps.get("regularization", np.nan),
            "n_params": num_model_parameters(m, condition_name),
            **{f"comp_{k}": v for k, v in comps.items()},
        })
    return pd.DataFrame(rows)


def run_final_fit(condition_dict):
    final_root = os.path.join(BASE_OUTPUT_DIR, "final_fit")
    os.makedirs(final_root, exist_ok=True)
    all_results = []

    for condition_name in CONDITIONS_TO_FIT:
        print(f"\n{'=' * 60}\n[FINAL FIT] {condition_name}\n{'=' * 60}")
        output_dir = os.path.join(final_root, condition_name.replace(" ", "_"))
        os.makedirs(output_dir, exist_ok=True)

        selected_dfs, selected_idxs = select_dfs(condition_dict, condition_name, DF_SELECTION)
        if not selected_dfs:
            print(f"[WARNING] No data found for condition: {condition_name}")
            continue

        real_df, real_traj_df, speed_profile, deadzone_speed, worm_summary = prepare_condition_subset(
            condition_name, selected_dfs, selected_idxs, label="final"
        )
        num_real_worms = real_traj_df["ID"].nunique()
        print(f"[INFO] Real trajectory worms used for output: {num_real_worms}")

        plot_real_trajectories(condition_name, real_traj_df, output_dir)
        plot_real_features(condition_name, real_df, output_dir)

        best_params, res_df, stdz_train = fit_models_on_dataset(
            condition_name, real_df, worm_summary, speed_profile, deadzone_speed,
            output_dir, final_fit_config()
        )
        all_results.extend(res_df.to_dict("records"))

        traj_dict = {}
        sim_feat_dict = {}
        component_rows = []
        for m in MODEL_NAMES:
            p_use = best_params[m].copy()
            sim_xy = simulate_population(
                p_use, m, num_real_worms, MAX_TIMESTAMP, DT_REAL,
                speed_profile, deadzone_speed, n_jobs=N_JOBS,
                seed_base=2026000, return_xy=True
            )
            traj_dict[m] = sim_xy
            sim_feat_dict[m] = sim_xy[["speed"] + FRAME_FEATURES].dropna()
            sim_xy.to_csv(os.path.join(output_dir, f"{condition_name}_{m}_trajectories.csv"), index=False)
            loss_val, comps = compute_loss(stdz_train, sim_xy, return_components=True)
            component_rows.append({"condition": condition_name, "model": m, **comps})

        plot_traj_5panel(condition_name, real_traj_df, traj_dict, output_dir)
        plot_feature_4models(condition_name, real_df, sim_feat_dict, output_dir)
        plot_worm_feature_compare(condition_name, worm_summary, traj_dict, output_dir, "net_displacement")
        plot_worm_feature_compare(condition_name, worm_summary, traj_dict, output_dir, "mean_dist_to_midline")

        pd.DataFrame(component_rows).to_csv(
            os.path.join(output_dir, f"{condition_name}_loss_components.csv"), index=False
        )

        nested_rows = []
        nested_pairs = [("RW", "KRW"), ("RW_switch", "KRW_switch")]
        for det_model, prob_model in nested_pairs:
            if det_model in best_params and prob_model in best_params:
                p_nested = deterministic_to_probabilistic_params(best_params[det_model])
                sim_nested = simulate_population(
                    p_nested, prob_model, NUM_WORMS_OPT_FINAL, MAX_TIMESTAMP, DT_REAL,
                    speed_profile, deadzone_speed, n_jobs=N_JOBS,
                    seed_base=303000, return_xy=False
                )
                nested_loss = compute_loss(stdz_train, sim_nested)
                nested_rows.append({
                    "condition": condition_name,
                    "deterministic_model": det_model,
                    "probabilistic_path": prob_model,
                    "embedded_loss": nested_loss,
                    "deterministic_best_loss": best_params[det_model].get("loss", np.nan),
                    "probabilistic_best_loss": best_params[prob_model].get("loss", np.nan),
                })
        pd.DataFrame(nested_rows).to_csv(
            os.path.join(output_dir, f"{condition_name}_nested_sanity_check.csv"), index=False
        )

    if all_results:
        global_res = pd.DataFrame(all_results)
        global_res.to_csv(os.path.join(final_root, "ALL_CONDITIONS_results.csv"), index=False)
        global_res.to_csv(os.path.join(BASE_OUTPUT_DIR, "ALL_CONDITIONS_results.csv"), index=False)
    return pd.DataFrame(all_results)


# =========================
# CROSS-VALIDATION
# =========================
def make_cv_splits(idxs, n_splits=5, seed=202606):
    idxs = list(idxs)
    rng = np.random.default_rng(seed)
    shuffled = np.array(idxs, dtype=int)
    rng.shuffle(shuffled)
    n_splits = max(2, min(int(n_splits), len(shuffled)))
    folds = np.array_split(shuffled, n_splits)
    splits = []
    all_set = set(shuffled.tolist())
    for k, test_arr in enumerate(folds):
        test_idxs = sorted(test_arr.tolist())
        train_idxs = sorted(list(all_set - set(test_idxs)))
        if train_idxs and test_idxs:
            splits.append((k, train_idxs, test_idxs))
    return splits


def subset_dfs_by_indices(all_condition_dfs, idxs):
    return [all_condition_dfs[i] for i in idxs]


def summarize_cv_results(cv_df):
    if cv_df.empty:
        return pd.DataFrame()
    summary = []
    for (condition, model), g in cv_df.groupby(["condition", "model"]):
        test_g = g[g["split"] == "test"]
        train_g = g[g["split"] == "train"]
        summary.append({
            "condition": condition,
            "model": model,
            "n_folds": int(test_g["fold"].nunique()) if "fold" in test_g else len(test_g),
            "n_params": int(test_g["n_params"].iloc[0]) if len(test_g) else np.nan,
            "train_data_loss_mean": float(train_g["data_loss"].mean()) if len(train_g) else np.nan,
            "train_data_loss_sem": sem(train_g["data_loss"].values) if len(train_g) else np.nan,
            "test_data_loss_mean": float(test_g["data_loss"].mean()) if len(test_g) else np.nan,
            "test_data_loss_sem": sem(test_g["data_loss"].values) if len(test_g) else np.nan,
            "test_objective_loss_mean": float(test_g["objective_loss"].mean()) if len(test_g) else np.nan,
            "test_regularization_mean": float(test_g["regularization"].mean()) if len(test_g) else np.nan,
        })
    return pd.DataFrame(summary).sort_values(["condition", "test_data_loss_mean"])


def zscore_vec(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def ordered_ctx_penalty(mean_1h, mean_4h, mean_end, margin=0.005):
    # Expected low-salt pattern from current real data: CTX_4h > CTX_end > CTX_1h.
    penalty = 0.0
    penalty += max(0.0, mean_1h + margin - mean_end) ** 2
    penalty += max(0.0, mean_end + margin - mean_4h) ** 2
    return float(penalty)


def compute_group_pattern_scores(cv_df):
    if cv_df.empty:
        return pd.DataFrame()
    test_df = cv_df[cv_df["split"] == "test"].copy()
    rows = []
    required_conditions = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
    for (fold, model), g in test_df.groupby(["fold", "model"]):
        if not set(required_conditions).issubset(set(g["condition"])):
            continue
        g = g.set_index("condition")
        real_ctx = [g.loc[c, "comp_real_CTX_mean_raw"] for c in required_conditions]
        sim_ctx = [g.loc[c, "comp_sim_CTX_mean_raw"] for c in required_conditions]
        real_mid = [g.loc[c, "comp_real_mean_dist_to_midline_mean_raw"] for c in required_conditions]
        sim_mid = [g.loc[c, "comp_sim_mean_dist_to_midline_mean_raw"] for c in required_conditions]
        real_abs_agl = [g.loc[c, "comp_real_abs_agl_velocity_mean_raw"] for c in required_conditions]
        sim_abs_agl = [g.loc[c, "comp_sim_abs_agl_velocity_mean_raw"] for c in required_conditions]

        rows.append({
            "fold": fold,
            "model": model,
            "ctx_pattern_mse_z": float(np.mean((zscore_vec(real_ctx) - zscore_vec(sim_ctx)) ** 2)),
            "midline_pattern_mse_z": float(np.mean((zscore_vec(real_mid) - zscore_vec(sim_mid)) ** 2)),
            "abs_agl_pattern_mse_z": float(np.mean((zscore_vec(real_abs_agl) - zscore_vec(sim_abs_agl)) ** 2)),
            "ctx_order_penalty": ordered_ctx_penalty(sim_ctx[0], sim_ctx[1], sim_ctx[2]),
            "real_CTX_1h": real_ctx[0],
            "real_CTX_4h": real_ctx[1],
            "real_CTX_end": real_ctx[2],
            "sim_CTX_1h": sim_ctx[0],
            "sim_CTX_4h": sim_ctx[1],
            "sim_CTX_end": sim_ctx[2],
        })
    score_df = pd.DataFrame(rows)
    if not score_df.empty:
        summary_rows = []
        for model, g in score_df.groupby("model"):
            summary_rows.append({
                "fold": "mean",
                "model": model,
                "ctx_pattern_mse_z": float(g["ctx_pattern_mse_z"].mean()),
                "ctx_pattern_mse_z_sem": sem(g["ctx_pattern_mse_z"].values),
                "midline_pattern_mse_z": float(g["midline_pattern_mse_z"].mean()),
                "midline_pattern_mse_z_sem": sem(g["midline_pattern_mse_z"].values),
                "abs_agl_pattern_mse_z": float(g["abs_agl_pattern_mse_z"].mean()),
                "abs_agl_pattern_mse_z_sem": sem(g["abs_agl_pattern_mse_z"].values),
                "ctx_order_penalty": float(g["ctx_order_penalty"].mean()),
                "ctx_order_penalty_sem": sem(g["ctx_order_penalty"].values),
            })
        score_df = pd.concat([score_df, pd.DataFrame(summary_rows)], ignore_index=True)
    return score_df


def run_cross_validation(condition_dict):
    cv_root = os.path.join(BASE_OUTPUT_DIR, "cross_validation")
    os.makedirs(cv_root, exist_ok=True)
    all_cv_rows = []
    all_param_rows = []

    for condition_name in CONDITIONS_TO_FIT:
        print(f"\n{'=' * 60}\n[CROSS-VALIDATION] {condition_name}\n{'=' * 60}")
        condition_output = os.path.join(cv_root, condition_name.replace(" ", "_"))
        os.makedirs(condition_output, exist_ok=True)

        selected_dfs, selected_idxs = select_dfs(condition_dict, condition_name, DF_SELECTION)
        if not selected_dfs:
            print(f"[WARNING] No data found for condition: {condition_name}")
            continue

        # Map back to the full condition_dict indices.
        all_condition_dfs = condition_dict[condition_name]
        splits = make_cv_splits(selected_idxs, n_splits=CV_N_SPLITS, seed=CV_SEED)
        print(f"[INFO] CV splits: {[(k, tr, te) for k, tr, te in splits]}")

        for fold, train_idxs, test_idxs in splits:
            print(f"\n[CV][{condition_name}] fold {fold}: train={train_idxs}, test={test_idxs}")
            fold_dir = os.path.join(condition_output, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            train_dfs = subset_dfs_by_indices(all_condition_dfs, train_idxs)
            test_dfs = subset_dfs_by_indices(all_condition_dfs, test_idxs)

            train_real_df, train_traj_df, speed_profile, deadzone_speed, train_worm = prepare_condition_subset(
                condition_name, train_dfs, train_idxs, label=f"fold{fold}_train"
            )
            test_real_df, test_traj_df, _, _, test_worm = prepare_condition_subset(
                condition_name, test_dfs, test_idxs, label=f"fold{fold}_test"
            )

            best_params, param_df, _ = fit_models_on_dataset(
                condition_name, train_real_df, train_worm, speed_profile, deadzone_speed,
                fold_dir, cv_fit_config()
            )
            param_df.insert(0, "fold", fold)
            param_df["train_idxs"] = ",".join(map(str, train_idxs))
            param_df["test_idxs"] = ",".join(map(str, test_idxs))
            all_param_rows.extend(param_df.to_dict("records"))

            # Repeated train and held-out evaluation with identical fitted params.
            n_train_worms = max(4, train_traj_df["ID"].nunique())
            n_test_worms = max(4, test_traj_df["ID"].nunique())

            train_eval = evaluate_best_params_on_target(
                condition_name, best_params,
                fit_real_df=train_real_df, fit_worm_summary=train_worm,
                target_real_df=train_real_df, target_worm_summary=train_worm,
                speed_profile=speed_profile, deadzone_speed=deadzone_speed,
                num_worms=n_train_worms, repeats=EVAL_REPEATS,
                seed_base=710000 + fold * 1000, split_label="train"
            )
            test_eval = evaluate_best_params_on_target(
                condition_name, best_params,
                fit_real_df=train_real_df, fit_worm_summary=train_worm,
                target_real_df=test_real_df, target_worm_summary=test_worm,
                speed_profile=speed_profile, deadzone_speed=deadzone_speed,
                num_worms=n_test_worms, repeats=EVAL_REPEATS,
                seed_base=810000 + fold * 1000, split_label="test"
            )

            fold_eval = pd.concat([train_eval, test_eval], ignore_index=True)
            fold_eval.insert(0, "fold", fold)
            fold_eval["train_idxs"] = ",".join(map(str, train_idxs))
            fold_eval["test_idxs"] = ",".join(map(str, test_idxs))
            fold_eval.to_csv(os.path.join(fold_dir, f"{condition_name}_fold{fold}_eval.csv"), index=False)
            all_cv_rows.extend(fold_eval.to_dict("records"))

            # Save one held-out trajectory set per model for diagnostics.
            for m in MODEL_NAMES:
                sim_test_xy = simulate_population(
                    best_params[m], m, n_test_worms, MAX_TIMESTAMP, DT_REAL,
                    speed_profile, deadzone_speed, n_jobs=N_JOBS,
                    seed_base=910000 + fold * 1000, return_xy=True
                )
                sim_test_xy.to_csv(os.path.join(fold_dir, f"{condition_name}_fold{fold}_{m}_test_trajectories.csv"), index=False)

        condition_cv = pd.DataFrame([r for r in all_cv_rows if r.get("condition") == condition_name])
        condition_cv.to_csv(os.path.join(condition_output, f"{condition_name}_CV_results.csv"), index=False)
        summarize_cv_results(condition_cv).to_csv(os.path.join(condition_output, f"{condition_name}_CV_summary.csv"), index=False)

    cv_df = pd.DataFrame(all_cv_rows)
    param_df = pd.DataFrame(all_param_rows)
    if not cv_df.empty:
        cv_df.to_csv(os.path.join(cv_root, "CV_ALL_FOLDS_results.csv"), index=False)
        cv_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "CV_ALL_FOLDS_results.csv"), index=False)
        cv_summary = summarize_cv_results(cv_df)
        cv_summary.to_csv(os.path.join(cv_root, "CV_MODEL_SUMMARY.csv"), index=False)
        cv_summary.to_csv(os.path.join(BASE_OUTPUT_DIR, "CV_MODEL_SUMMARY.csv"), index=False)
        group_scores = compute_group_pattern_scores(cv_df)
        group_scores.to_csv(os.path.join(cv_root, "CV_GROUP_PATTERN_SCORES.csv"), index=False)
        group_scores.to_csv(os.path.join(BASE_OUTPUT_DIR, "CV_GROUP_PATTERN_SCORES.csv"), index=False)
    if not param_df.empty:
        param_df.to_csv(os.path.join(cv_root, "CV_FITTED_PARAMS.csv"), index=False)
        param_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "CV_FITTED_PARAMS.csv"), index=False)
    return cv_df


# =========================
# MAIN
# =========================
def main():
    pkl_path = "condition_dict.pkl"
    if not os.path.exists(pkl_path):
        print(f"[ERROR] missing {pkl_path}")
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        condition_dict = pickle.load(f)

    if RUN_FINAL_FIT:
        run_final_fit(condition_dict)
    if RUN_CROSS_VALIDATION:
        run_cross_validation(condition_dict)

    print(f"\n[INFO] Done. Results saved in: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
