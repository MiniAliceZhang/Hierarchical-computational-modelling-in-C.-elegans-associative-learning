# -*- coding: utf-8 -*-
"""Logistic regression of PR entry/exit on normalized distance change and reward.

Real states come from the q90 HMM. The selected target of a real worm is
inferred from the sign of the smoothed CTX (positive = toward low salt).
Model states come from KRW-switch simulations and use the model's own target.
Both regressions are at 0.25 s resolution with worm-clustered standard errors.
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import DistributionOverlapDecision as dod
import hmm_pr_analysis as hmm
from kinematic_utils import speed_profile_from_cache

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
OUT = Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\周老师第5点_时间相关性_待审\状态转换回归")
DATA = OUT / "data"; FIG = OUT / "figures"; TAB = OUT / "tables"
for d in [DATA, FIG, TAB]:
    d.mkdir(parents=True, exist_ok=True)

CONDS = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
FNAME = {"3g-1gNa-1h": "3g-1gNa-1h", "3g-1gNa-4h": "3g-1gNa-4h", "1gNa-end(8x)": "1gNa-end8x"}
LABEL = {"3g-1gNa-1h": "1 h", "3g-1gNa-4h": "4 h", "1gNa-end(8x)": "8 h"}
N_SIM = 50
STRIDE = 5
GRAD = (dod.maxNaCl - dod.minNaCl) / dod.WIDTH_REAL

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def fill_target(ctx):
    s = pd.Series(ctx).rolling(5, center=True, min_periods=1).mean().to_numpy()
    target = np.where(s > 0, 1, np.where(s < 0, 0, np.nan))
    target = pd.Series(target).ffill().bfill().fillna(1).to_numpy()
    return target.astype(int)


def dr_signal(x, speed, target):
    sample = np.where(target == 1, dod.minNaCl, dod.maxNaCl)
    cur = dod.calc_concentration(x)
    nxt = dod.calc_concentration(np.r_[x[1:], x[-1]])
    delta = np.abs(sample - cur) - np.abs(sample - nxt)
    vpx = np.maximum(speed * dod.PX_PER_CM, 1e-9)
    return -delta / (vpx * GRAD)


def fit_logit(dr, reward, y, groups):
    dr = np.asarray(dr, float); reward = np.asarray(reward, float); y = np.asarray(y, int)
    ok = np.isfinite(dr) & np.isfinite(reward) & np.isfinite(y)
    dr, reward, y, groups = dr[ok], reward[ok], y[ok], np.asarray(groups)[ok]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    # winsorize extreme normalized dr values
    lo, hi = np.percentile(dr, [1, 99])
    dr = np.clip(dr, lo, hi)
    zdr = (dr - dr.mean()) / (dr.std() + 1e-9)
    zr = (reward - reward.mean()) / (reward.std() + 1e-9)
    X = sm.add_constant(np.column_stack([zdr, zr, zdr * zr]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups}, maxiter=200)
        return res
    except Exception:
        try:
            res = sm.Logit(y, X).fit_regularized(disp=0, maxiter=200)
            return res
        except Exception:
            return None


def add_coef(rows, cond, model, res, kind):
    if res is None:
        return
    names = ["const", "dr", "reward", "interaction"]
    for i, name in enumerate(names):
        rows.append({
            "condition": cond, "model": model, "kind": kind, "term": name,
            "coef": float(res.params[i]),
            "se": float(res.bse[i]) if hasattr(res, "bse") else np.nan,
            "p": float(res.pvalues[i]) if hasattr(res, "pvalues") else np.nan,
            "n": int(res.nobs),
        })


def main():
    params = pd.read_csv(ROOT / "Fitting" / "SingleConc_resultsV13_CV_features" / "ALL_CONDITIONS_results.csv")
    hmm_sum = pd.read_csv(Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\审阅版\figures\value_state_analysis\hmm_pr_summary.csv"))
    rows = []
    for cond in CONDS:
        print("=" * 40, cond)
        with open(ROOT / "Fitting" / "regression_cache" / (FNAME[cond] + ".pkl"), "rb") as f:
            real = pickle.load(f)
        with open(ROOT / "Fitting" / "hmm_seq_cache" / (FNAME[cond] + ".pkl"), "rb") as f:
            seq = pickle.load(f)
        hr = hmm_sum[hmm_sum.condition == cond].iloc[0]
        pi = np.array([1 - hr.pi_pr, hr.pi_pr])
        A = np.array([[hr.A_00, hr.A_01], [hr.A_10, hr.A_11]])
        p = params[(params.condition == cond) & (params.model == "KRW_switch")].iloc[0].to_dict()
        r_low, r_high = dod.reward_transform(float(p["w1"])), dod.reward_transform(float(p["w2"]))

        # real entry/exit
        dr_e, rew_e, ye, ge = [], [], [], []
        dr_x, rew_x, yx, gx = [], [], [], []
        for o in real:
            x = o["x"][::STRIDE]; sp = o["speed"][::STRIDE]; ag = np.abs(o["agl"])[::STRIDE]; ctx = o["ctx"][::STRIDE]
            if len(x) < 10:
                continue
            st = hmm.viterbi((ag >= hr.threshold_value).astype(np.int8), pi, A, hmm.P0, hmm.P1)
            target = fill_target(ctx)
            dr = dr_signal(x, sp, target); rew = np.where(target == 1, r_low, r_high)
            gid = (o["df_idx"], o["worm_id"])
            for t in range(len(st) - 1):
                if st[t] == 0:
                    dr_e.append(dr[t]); rew_e.append(rew[t]); ye.append(int(st[t + 1] == 1)); ge.append(gid)
                elif st[t] == 1:
                    dr_x.append(dr[t]); rew_x.append(rew[t]); yx.append(int(st[t + 1] == 0)); gx.append(gid)
        add_coef(rows, cond, "Real HMM", fit_logit(dr_e, rew_e, ye, ge), "entry")
        add_coef(rows, cond, "Real HMM", fit_logit(dr_x, rew_x, yx, gx), "exit")

        # model entry/exit
        profile = speed_profile_from_cache(seq)
        sim = dod.simulate_population(p, "KRW_switch", N_SIM, dod.MAX_TIMESTAMP, dod.DT_REAL,
                                      profile, None, n_jobs=1, seed_base=20261116, return_xy=True)
        dr_e, rew_e, ye, ge = [], [], [], []
        dr_x, rew_x, yx, gx = [], [], [], []
        for wid, g in sim.groupby("ID"):
            x = g["x"].to_numpy(); sp = g["speed"].to_numpy(); st = g["state"].to_numpy(int)
            rew = g["selected_reward"].to_numpy(); sd = g["select_dist"].to_numpy()
            idx = np.arange(0, len(st) - STRIDE, STRIDE)
            xt = x[idx]; sp_t = sp[idx]; rew_t = rew[idx]; st_t = st[idx]; st_n = st[idx + STRIDE]; sd_t = sd[idx]
            sample = np.where(sd_t == 0, dod.minNaCl, dod.maxNaCl)
            cur = dod.calc_concentration(xt); nxt = dod.calc_concentration(x[idx + STRIDE])
            delta = np.abs(sample - cur) - np.abs(sample - nxt)
            vpx = np.maximum(sp_t * dod.PX_PER_CM, 1e-9)
            dr = -delta / (vpx * GRAD)
            for i in range(len(st_t)):
                if st_t[i] == 0:
                    dr_e.append(dr[i]); rew_e.append(rew_t[i]); ye.append(int(st_n[i] == 1)); ge.append(wid)
                elif st_t[i] == 1:
                    dr_x.append(dr[i]); rew_x.append(rew_t[i]); yx.append(int(st_n[i] == 0)); gx.append(wid)
        add_coef(rows, cond, "KRW-switch", fit_logit(dr_e, rew_e, ye, ge), "entry")
        add_coef(rows, cond, "KRW-switch", fit_logit(dr_x, rew_x, yx, gx), "exit")
        del sim

    d = pd.DataFrame(rows)
    d.to_csv(DATA / "transition_regression.csv", index=False)
    print(d.round(3).to_string(index=False))

    # coefficient figure: entry
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True)
    terms = [("dr", "$\\Delta R$"), ("reward", "reward"), ("interaction", "$\\Delta R\\times$reward")]
    x = np.arange(len(CONDS)); w = 0.34
    colors = {"Real HMM": "#4D4D4D", "KRW-switch": "#6A3D9A"}
    for ax, (term, lab) in zip(axes, terms):
        for i, model in enumerate(["Real HMM", "KRW-switch"]):
            sub = d[(d.kind == "entry") & (d.model == model) & (d.term == term)]
            vals = [sub[sub.condition == c].coef.iloc[0] if len(sub[sub.condition == c]) else np.nan for c in CONDS]
            errs = [sub[sub.condition == c].se.iloc[0] if len(sub[sub.condition == c]) else np.nan for c in CONDS]
            ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=2, label=model,
                   color=colors[model], edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in CONDS])
        ax.set_ylabel("标准化 logit 系数"); ax.set_title(lab, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_transition_coefficients.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_transition_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
