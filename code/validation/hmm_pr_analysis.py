# -*- coding: utf-8 -*-
"""Two-state HMM for real PR-like states.

Observations are binary high-angular-velocity frames (|agl_velocity| >= q90).
The two hidden states are non-PR and PR-like. Emission probabilities are fixed
at 0.05 (non-PR) and 0.70 (PR-like); only the initial distribution and the
transition matrix are estimated. This avoids treating every high-angular frame
as an independent PR event and avoids an arbitrary gap-merging rule.

The HMM state is still a kinematic inference, not a ground-truth PR/Run label.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
OUT = Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\审阅版\figures\value_state_analysis")
OUT.mkdir(parents=True, exist_ok=True)
CONDS = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
LABEL = {"3g-1gNa-1h": "1 h", "3g-1gNa-4h": "4 h", "1gNa-end(8x)": "8 h"}
MAX_TS = 14400
DF_SELECTION = list(range(10, 20))
STRIDE = 5
DT = 0.05 * STRIDE
N_TRAIN = 20
MAX_ITER = 12
P0, P1 = 0.05, 0.70

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def collect(dfs):
    agl = []
    for j, df in enumerate(dfs):
        if j not in DF_SELECTION:
            continue
        if "Timestamp" in df:
            df = df[df["Timestamp"] <= MAX_TS]
        for _, g in df.groupby("ID", sort=False):
            v = np.abs(g["agl_velocity"].dropna().to_numpy(dtype=float))[::STRIDE]
            if len(v) >= 20 and np.isfinite(v).all():
                agl.append(v)
    return agl


def make_obs(agl, q):
    pooled = np.concatenate(agl)
    th = float(np.percentile(pooled, q))
    obs = [(o >= th).astype(np.int8) for o in agl]
    return obs, th


def fb(y, pi, A, p0=P0, p1=P1):
    T = len(y)
    b = np.where(y[:, None] == 1,
                 np.array([p0, p1])[None, :],
                 np.array([1 - p0, 1 - p1])[None, :])
    alpha = np.zeros((T, 2)); beta = np.zeros((T, 2)); c = np.zeros(T)
    alpha[0] = pi * b[0]
    c[0] = alpha[0].sum()
    alpha[0] /= max(c[0], 1e-300)
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * b[t]
        c[t] = alpha[t].sum()
        alpha[t] /= max(c[t], 1e-300)
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (b[t + 1] * beta[t + 1])
        beta[t] /= max(c[t + 1], 1e-300)
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    xi_sum = np.zeros((2, 2))
    for t in range(T - 1):
        xi = alpha[t][:, None] * A * (b[t + 1] * beta[t + 1])[None, :]
        xi /= max(xi.sum(), 1e-300)
        xi_sum += xi
    return gamma, xi_sum


def fit_hmm(obs, seed=20261116, p0=P0, p1=P1):
    rng = np.random.default_rng(seed)
    train = [obs[i] for i in rng.choice(len(obs), size=min(N_TRAIN, len(obs)), replace=False)]
    pi = np.array([0.9, 0.1])
    A = np.array([[0.98, 0.02], [0.10, 0.90]])
    for _ in range(MAX_ITER):
        g0 = np.zeros(2)
        xi_sum = np.zeros((2, 2))
        for y in train:
            g, xi = fb(y, pi, A, p0, p1)
            g0 += g[0]
            xi_sum += xi
        pi = (g0 + 1e-6) / (g0.sum() + 2e-6)
        A = (xi_sum + 1e-6) / (xi_sum.sum(axis=1, keepdims=True) + 2e-6)
    return pi, A


def viterbi(y, pi, A, p0=P0, p1=P1):
    T = len(y)
    b = np.where(y[:, None] == 1,
                 np.array([p0, p1])[None, :],
                 np.array([1 - p0, 1 - p1])[None, :])
    logb = np.log(np.maximum(b, 1e-300))
    logA = np.log(np.maximum(A, 1e-300))
    delta = np.zeros((T, 2)); psi = np.zeros((T, 2), dtype=np.int8)
    delta[0] = np.log(np.maximum(pi, 1e-300)) + logb[0]
    for t in range(1, T):
        tmp = delta[t - 1][:, None] + logA
        psi[t] = np.argmax(tmp, axis=0)
        delta[t] = tmp.max(axis=0) + logb[t]
    states = np.zeros(T, dtype=np.int8)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def run_stats(mask):
    p = np.r_[False, mask, False]
    starts = np.where(~p[:-1] & p[1:])[0]
    ends = np.where(p[:-1] & ~p[1:])[0]
    lens = ends - starts
    total_min = max(len(mask) * DT / 60.0, 1e-9)
    return {
        "fraction": float(mask.mean()),
        "events_per_min": len(lens) / total_min,
        "duration_s": float(np.mean(lens) * DT) if len(lens) else 0.0,
    }


def summarize(agl, q=90, p0=P0, p1=P1):
    obs, th = make_obs(agl, q)
    pi, A = fit_hmm(obs, p0=p0, p1=p1)
    rows = []
    for y in obs:
        s = viterbi(y, pi, A, p0, p1)
        st = run_stats(s == 1)
        rows.append(st)
    d = pd.DataFrame(rows)
    rec = {
        "threshold_q": q, "threshold_value": th, "p0": p0, "p1": p1,
        "n": len(d),
        "hmm_pr_fraction": d["fraction"].mean(),
        "hmm_pr_fraction_sem": d["fraction"].std(ddof=1) / np.sqrt(len(d)),
        "hmm_pr_events_per_min": d["events_per_min"].mean(),
        "hmm_pr_events_per_min_sem": d["events_per_min"].std(ddof=1) / np.sqrt(len(d)),
        "hmm_pr_duration_s": d["duration_s"].mean(),
        "hmm_pr_duration_s_sem": d["duration_s"].std(ddof=1) / np.sqrt(len(d)),
        "pi_pr": float(pi[1]),
        "A_00": float(A[0, 0]), "A_01": float(A[0, 1]),
        "A_10": float(A[1, 0]), "A_11": float(A[1, 1]),
        "dur_state1_s": float(DT / max(1 - A[1, 1], 1e-9)),
        "dur_state0_s": float(DT / max(1 - A[0, 0], 1e-9)),
    }
    return rec, d


def main():
    with open(ROOT / "data" / "condition_dict.pkl", "rb") as f:
        cd = pickle.load(f)
    all_rows = []
    sens_rows = []
    for cond in CONDS:
        print("=" * 50, cond)
        agl = collect(cd[cond])
        print("worms", len(agl))
        rec, d = summarize(agl, q=90, p0=P0, p1=P1)
        rec["condition"] = cond
        all_rows.append(rec)
        d["condition"] = cond
        d.to_csv(OUT / f"hmm_pr_per_worm_{cond.replace('(', '').replace(')', '')}.csv", index=False)
        # sensitivity: stricter threshold
        r, _ = summarize(agl, q=95, p0=P0, p1=P1)
        r["condition"] = cond
        sens_rows.append(r)
        print(pd.DataFrame([rec]).to_string(index=False))
    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT / "hmm_pr_summary.csv", index=False)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "hmm_pr_sensitivity.csv", index=False)
    print(summary.to_string(index=False))

    model = pd.read_csv(OUT / "value_state_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    x = np.arange(len(CONDS))
    w = 0.26
    metrics = [
        ("hmm_pr_fraction", "state_pr_fraction", "PR 时间占比"),
        ("hmm_pr_events_per_min", "state_pr_events_per_min", "PR 事件率（次/分钟）"),
        ("hmm_pr_duration_s", "state_pr_duration_s", "PR 持续时间（s）"),
    ]
    for ax, (hcol, mcol, ylab) in zip(axes, metrics):
        real = [summary[summary.condition == c][hcol].iloc[0] for c in CONDS]
        full = [model[(model.condition == c) & (model.model == "KRW_switch_full")][mcol].iloc[0] for c in CONDS]
        noval = [model[(model.condition == c) & (model.model == "KRW_switch_no_value")][mcol].iloc[0] for c in CONDS]
        real_e = [summary[summary.condition == c][hcol + "_sem"].iloc[0] for c in CONDS]
        full_e = [model[(model.condition == c) & (model.model == "KRW_switch_full")][mcol + "_sem"].iloc[0] for c in CONDS]
        noval_e = [model[(model.condition == c) & (model.model == "KRW_switch_no_value")][mcol + "_sem"].iloc[0] for c in CONDS]
        ax.bar(x - w, real, w, yerr=real_e, capsize=2, label="真实 HMM", color="#4D4D4D", edgecolor="white", linewidth=0.3)
        ax.bar(x, full, w, yerr=full_e, capsize=2, label="KRW 全模型", color="#6A3D9A", edgecolor="white", linewidth=0.3)
        ax.bar(x + w, noval, w, yerr=noval_e, capsize=2, label="KRW 无价值", color="#D9C2E8", edgecolor="white", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[c] for c in CONDS])
        ax.set_ylabel(ylab)
        ax.set_title(ylab, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hmm_pr_model.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_hmm_pr_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "fig_hmm_pr_model.pdf")


if __name__ == "__main__":
    main()
