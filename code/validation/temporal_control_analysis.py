# -*- coding: utf-8 -*-
"""Direct validation of switch and a continuous AR(1) continuity control.

Outputs go to the review folder, not into the thesis.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

import DistributionOverlapDecision as dod
import hmm_pr_analysis as hmm
from kinematic_utils import omega_metrics, speed_profile_from_cache, acf

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
REVIEW = Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\周老师第5点_时间相关性_待审")
DATA = REVIEW / "data"; FIG = REVIEW / "figures"; TAB = REVIEW / "tables"
for d in [DATA, FIG, TAB]:
    d.mkdir(parents=True, exist_ok=True)

CONDS = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
FNAME = {"3g-1gNa-1h": "3g-1gNa-1h", "3g-1gNa-4h": "3g-1gNa-4h", "1gNa-end(8x)": "1gNa-end8x"}
LABEL = {"3g-1gNa-1h": "1 h", "3g-1gNa-4h": "4 h", "1gNa-end(8x)": "8 h"}
N_SIM = 50
MAX_LAG = 10

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def state_stats_full(s, dt=0.05):
    s = np.asarray(s, dtype=int)
    n0 = np.sum(s == 0); n1 = np.sum(s == 1)
    n01 = np.sum((s[:-1] == 0) & (s[1:] == 1))
    n10 = np.sum((s[:-1] == 1) & (s[1:] == 0))
    p = np.r_[False, s == 1, False]
    starts = np.where(~p[:-1] & p[1:])[0]
    ends = np.where(p[:-1] & ~p[1:])[0]
    lens = ends - starts
    p01 = n01 / max(n0, 1); p10 = n10 / max(n1, 1)
    return {
        "fraction": float(np.mean(s == 1)),
        "events_per_min": len(lens) / max(len(s) * dt / 60.0, 1e-9),
        "duration_s": float(np.mean(lens) * dt) if len(lens) else 0.0,
        "A01": 1.0 - (1.0 - p01) ** 5,
        "A10": 1.0 - (1.0 - p10) ** 5,
    }


def state_acf(s, max_lag=MAX_LAG):
    s = np.asarray(s, dtype=float)
    p = float(np.mean(s))
    den = p * (1.0 - p)
    if den <= 0:
        return np.full(max_lag + 1, np.nan)
    out = [1.0]
    for k in range(1, max_lag + 1):
        out.append(float(np.mean((s[:-k] - p) * (s[k:] - p)) / den))
    return np.array(out)


def fit_decode_hmm(agl_list, q=90):
    obs, th = hmm.make_obs(agl_list, q)
    pi, A = hmm.fit_hmm(obs)
    states = [hmm.viterbi(y, pi, A, hmm.P0, hmm.P1) for y in obs]
    stats = [hmm.run_stats(s == 1) for s in states]
    d = pd.DataFrame(stats)
    rec = {
        "threshold_value": th,
        "hmm_pr_fraction": d["fraction"].mean(),
        "hmm_pr_events_per_min": d["events_per_min"].mean(),
        "hmm_pr_duration_s": d["duration_s"].mean(),
        "A01": float(A[0, 1]), "A10": float(A[1, 0]),
        "pi_pr": float(pi[1]),
    }
    return rec, states


def sim_stats(sim):
    rows = []
    for _, g in sim.groupby("ID"):
        st = state_stats_full(g["state"].to_numpy())
        st["state_acf1"] = state_acf(g["state"].to_numpy()[::5])[1]
        om = omega_metrics(g["x"], g["y"])
        st.update({"omega_mean": om["mean"], "omega_std": om["std"], "omega_acf1": om["acf"][1]})
        rows.append(st)
    return pd.DataFrame(rows)


def main():
    params = pd.read_csv(ROOT / "Fitting" / "SingleConc_resultsV13_CV_features" / "ALL_CONDITIONS_results.csv")
    grid = pd.read_csv(ROOT / "Fitting" / "ar1_fit_grid.csv")
    hmm_sum = pd.read_csv(Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\审阅版\figures\value_state_analysis\hmm_pr_summary.csv"))

    summary = []
    state_acf_curves = {}
    kin_acf_curves = {}
    per_worm_all = []

    for cond in CONDS:
        print("=" * 40, cond)
        with open(ROOT / "Fitting" / "hmm_seq_cache" / (FNAME[cond] + ".pkl"), "rb") as f:
            seq = pickle.load(f)
        with open(ROOT / "Fitting" / "xy_seq_cache" / (FNAME[cond] + ".pkl"), "rb") as f:
            xy = pickle.load(f)
        profile = speed_profile_from_cache(seq)

        # real HMM states and real kinematic ACF
        hr = hmm_sum[hmm_sum.condition == cond].iloc[0]
        pi = np.array([1 - hr.pi_pr, hr.pi_pr])
        A = np.array([[hr.A_00, hr.A_01], [hr.A_10, hr.A_11]])
        real_states = []
        for o in seq:
            y = (np.abs(o["agl"])[::5] >= hr.threshold_value).astype(np.int8)
            real_states.append(hmm.viterbi(y, pi, A, hmm.P0, hmm.P1))
        real_sacf = np.nanmean([state_acf(s) for s in real_states], axis=0)
        real_om = [omega_metrics(o["x"], o["y"]) for o in xy]
        real_kacf = np.nanmean([o["acf"] for o in real_om], axis=0)
        summary.append({
            "condition": cond, "model": "Real HMM", "n": len(real_states),
            "pr_fraction": hr.hmm_pr_fraction,
            "events_per_min": hr.hmm_pr_events_per_min,
            "duration_s": hr.hmm_pr_duration_s,
            "A01": hr.A_01, "A10": hr.A_10,
            "state_acf1": real_sacf[1],
            "omega_mean": np.mean([o["mean"] for o in real_om]),
            "omega_std": np.mean([o["std"] for o in real_om]),
            "omega_acf1": real_kacf[1],
        })
        state_acf_curves[(cond, "Real HMM")] = real_sacf
        kin_acf_curves[(cond, "Real HMM")] = real_kacf

        p = params[(params.condition == cond) & (params.model == "KRW_switch")].iloc[0].to_dict()

        # switch simulation
        sim_sw = dod.simulate_population(p, "KRW_switch", N_SIM, dod.MAX_TIMESTAMP, dod.DT_REAL,
                                         profile, None, n_jobs=1, seed_base=20261116, return_xy=True)
        sw = sim_stats(sim_sw)
        sw["condition"] = cond; sw["model"] = "KRW-switch"
        per_worm_all.append(sw)
        summary.append({
            "condition": cond, "model": "KRW-switch", "n": len(sw),
            "pr_fraction": sw.fraction.mean(),
            "events_per_min": sw.events_per_min.mean(),
            "duration_s": sw.duration_s.mean(),
            "A01": sw.A01.mean(), "A10": sw.A10.mean(),
            "state_acf1": sw.state_acf1.mean(),
            "omega_mean": sw.omega_mean.mean(),
            "omega_std": sw.omega_std.mean(),
            "omega_acf1": sw.omega_acf1.mean(),
        })
        # switch kinematic ACF per worm
        sw_acfs = [omega_metrics(g["x"], g["y"])["acf"] for _, g in sim_sw.groupby("ID")]
        state_acf_curves[(cond, "KRW-switch")] = np.nanmean(
            [state_acf(g["state"].to_numpy()[::5]) for _, g in sim_sw.groupby("ID")], axis=0)
        kin_acf_curves[(cond, "KRW-switch")] = np.nanmean(sw_acfs, axis=0)
        del sim_sw

        # AR1 control
        best = grid[grid.condition == cond].sort_values("loss").iloc[0]
        pp = {**p, "rho_ar": float(best.rho_ar), "sigma_ar": float(best.sigma_ar)}
        sim_ar = dod.simulate_population(pp, "KRW_AR1", N_SIM, dod.MAX_TIMESTAMP, dod.DT_REAL,
                                         profile, None, n_jobs=1, seed_base=20261116, return_xy=True)
        agl_ar = [g["abs_agl_velocity"].to_numpy()[::5] for _, g in sim_ar.groupby("ID")]
        rec_ar, states_ar = fit_decode_hmm(agl_ar, q=90)
        ar = pd.DataFrame([hmm.run_stats(s == 1) for s in states_ar])
        ar["state_acf1"] = [state_acf(s)[1] for s in states_ar]
        ar["omega_mean"] = [omega_metrics(g["x"], g["y"])["mean"] for _, g in sim_ar.groupby("ID")]
        ar["omega_std"] = [omega_metrics(g["x"], g["y"])["std"] for _, g in sim_ar.groupby("ID")]
        ar["omega_acf1"] = [omega_metrics(g["x"], g["y"])["acf"][1] for _, g in sim_ar.groupby("ID")]
        ar["condition"] = cond; ar["model"] = "KRW-AR1"
        per_worm_all.append(ar)
        summary.append({
            "condition": cond, "model": "KRW-AR1", "n": len(ar),
            "pr_fraction": ar.fraction.mean(),
            "events_per_min": ar.events_per_min.mean(),
            "duration_s": ar.duration_s.mean(),
            "A01": rec_ar["A01"], "A10": rec_ar["A10"],
            "state_acf1": ar.state_acf1.mean(),
            "omega_mean": ar.omega_mean.mean(),
            "omega_std": ar.omega_std.mean(),
            "omega_acf1": ar.omega_acf1.mean(),
            "threshold_value": rec_ar["threshold_value"],
        })
        state_acf_curves[(cond, "KRW-AR1")] = np.nanmean([state_acf(s) for s in states_ar], axis=0)
        ar_acfs = [omega_metrics(g["x"], g["y"])["acf"] for _, g in sim_ar.groupby("ID")]
        kin_acf_curves[(cond, "KRW-AR1")] = np.nanmean(ar_acfs, axis=0)
        del sim_ar

    s = pd.DataFrame(summary)
    s.to_csv(DATA / "temporal_validation_summary.csv", index=False)
    pd.concat(per_worm_all, ignore_index=True).to_csv(DATA / "temporal_validation_per_worm.csv", index=False)
    print(s.to_string(index=False))

    # Figure 1: PR stats
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    x = np.arange(len(CONDS)); w = 0.26
    models = ["Real HMM", "KRW-switch", "KRW-AR1"]
    colors = {"Real HMM": "#4D4D4D", "KRW-switch": "#6A3D9A", "KRW-AR1": "#2C7FB8"}
    metrics = [("pr_fraction", "PR 时间占比"), ("events_per_min", "PR 事件率（次/分钟）"), ("duration_s", "PR 持续时间（s）")]
    for ax, (col, ylab) in zip(axes, metrics):
        for i, m in enumerate(models):
            vals = [s[(s.condition == c) & (s.model == m)][col].iloc[0] for c in CONDS]
            ax.bar(x + (i - 1) * w, vals, w, label=m, color=colors[m], edgecolor="white", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in CONDS])
        ax.set_ylabel(ylab); ax.set_title(ylab, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_pr_direct_validation.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_pr_direct_validation.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Figure 2: transition probabilities
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, col, ylab in zip(axes, ["A01", "A10"], ["$P(\\mathrm{nonPR}\\to\\mathrm{PR})$", "$P(\\mathrm{PR}\\to\\mathrm{nonPR})$"]):
        for i, m in enumerate(models):
            vals = [s[(s.condition == c) & (s.model == m)][col].iloc[0] for c in CONDS]
            ax.bar(x + (i - 1) * w, vals, w, label=m, color=colors[m], edgecolor="white", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in CONDS]); ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_transition_direct.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_transition_direct.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Figure 3: state ACF
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
    lags = np.arange(MAX_LAG + 1) * 0.25
    for ax, cond in zip(axes, CONDS):
        for m in models:
            ax.plot(lags, state_acf_curves[(cond, m)], marker="o", ms=2.6, lw=1.4, label=m, color=colors[m])
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(LABEL[cond], fontsize=9); ax.set_xlabel("滞后（s）")
    axes[0].set_ylabel("状态自相关")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_state_acf.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_state_acf.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Figure 4: kinematic ACF
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
    for ax, cond in zip(axes, CONDS):
        for m in models:
            ax.plot(lags, kin_acf_curves[(cond, m)][:MAX_LAG + 1], marker="o", ms=2.6, lw=1.4, label=m, color=colors[m])
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(LABEL[cond], fontsize=9); ax.set_xlabel("滞后（s）")
    axes[0].set_ylabel("|角速度| 自相关")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_kinematic_acf.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_kinematic_acf.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("saved", REVIEW)


if __name__ == "__main__":
    main()
