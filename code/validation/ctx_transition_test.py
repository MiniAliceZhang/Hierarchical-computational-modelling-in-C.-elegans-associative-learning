# -*- coding: utf-8 -*-
"""Exploratory entry/exit regression on observable CTX and speed."""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import DistributionOverlapDecision as dod
import hmm_pr_analysis as hmm
from kinematic_utils import speed_profile_from_cache

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
CONDS = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
FNAME = {"3g-1gNa-1h": "3g-1gNa-1h", "3g-1gNa-4h": "3g-1gNa-4h", "1gNa-end(8x)": "1gNa-end8x"}
N_SIM = 50
STRIDE = 5


def fit(ctx, speed, y, groups):
    ok = np.isfinite(ctx) & np.isfinite(speed) & np.isfinite(y)
    ctx, speed, y, groups = ctx[ok], speed[ok], y[ok], np.asarray(groups)[ok]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    zc = (ctx - ctx.mean()) / (ctx.std() + 1e-9)
    zs = (speed - speed.mean()) / (speed.std() + 1e-9)
    X = sm.add_constant(np.column_stack([zc, zs]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups}, maxiter=200)
    except Exception:
        return None


def main():
    hmm_sum = pd.read_csv(Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\审阅版\figures\value_state_analysis\hmm_pr_summary.csv"))
    params = pd.read_csv(ROOT / "Fitting" / "SingleConc_resultsV13_CV_features" / "ALL_CONDITIONS_results.csv")
    rows = []
    for cond in CONDS:
        with open(ROOT / "Fitting" / "regression_cache" / (FNAME[cond] + ".pkl"), "rb") as f:
            real = pickle.load(f)
        with open(ROOT / "Fitting" / "hmm_seq_cache" / (FNAME[cond] + ".pkl"), "rb") as f:
            seq = pickle.load(f)
        hr = hmm_sum[hmm_sum.condition == cond].iloc[0]
        pi = np.array([1 - hr.pi_pr, hr.pi_pr]); A = np.array([[hr.A_00, hr.A_01], [hr.A_10, hr.A_11]])
        for name, objs, state_getter, ctx_getter, speed_getter in [
            ("Real HMM", real, None, lambda o: o["ctx"][::STRIDE], lambda o: o["speed"][::STRIDE]),
        ]:
            ce, se, ye, ge = [], [], [], []; cx, sx, yx, gx = [], [], [], []
            for o in objs:
                ag = np.abs(o["agl"])[::STRIDE]
                if len(ag) < 10:
                    continue
                st = hmm.viterbi((ag >= hr.threshold_value).astype(np.int8), pi, A, hmm.P0, hmm.P1)
                c = ctx_getter(o); s = speed_getter(o); gid = (o["df_idx"], o["worm_id"])
                for t in range(len(st) - 1):
                    if st[t] == 0:
                        ce.append(c[t]); se.append(s[t]); ye.append(int(st[t + 1] == 1)); ge.append(gid)
                    elif st[t] == 1:
                        cx.append(c[t]); sx.append(s[t]); yx.append(int(st[t + 1] == 0)); gx.append(gid)
            for kind, xx, yy, gg in [("entry", ce, ye, ge), ("exit", cx, yx, gx)]:
                r = fit(np.asarray(xx), np.asarray(se if kind == "entry" else sx), np.asarray(yy), gg)
                if r is not None:
                    rows.append({"condition": cond, "model": name, "kind": kind,
                                 "b_ctx": r.params[1], "se_ctx": r.bse[1], "p_ctx": r.pvalues[1],
                                 "b_speed": r.params[2], "se_speed": r.bse[2], "p_speed": r.pvalues[2], "n": int(r.nobs)})

        p = params[(params.condition == cond) & (params.model == "KRW_switch")].iloc[0].to_dict()
        profile = speed_profile_from_cache(seq)
        sim = dod.simulate_population(p, "KRW_switch", N_SIM, dod.MAX_TIMESTAMP, dod.DT_REAL,
                                      profile, None, n_jobs=1, seed_base=20261116, return_xy=True)
        ce, se, ye, ge = [], [], [], []; cx, sx, yx, gx = [], [], [], []
        for wid, g in sim.groupby("ID"):
            st = g["state"].to_numpy(int); c = g["CTX"].to_numpy(); s = g["speed"].to_numpy()
            idx = np.arange(0, len(st) - STRIDE, STRIDE)
            st_t = st[idx]; st_n = st[idx + STRIDE]; c_t = c[idx]; s_t = s[idx]
            for i in range(len(st_t)):
                if st_t[i] == 0:
                    ce.append(c_t[i]); se.append(s_t[i]); ye.append(int(st_n[i] == 1)); ge.append(wid)
                elif st_t[i] == 1:
                    cx.append(c_t[i]); sx.append(s_t[i]); yx.append(int(st_n[i] == 0)); gx.append(wid)
        for kind, xx, ss, yy, gg in [("entry", ce, se, ye, ge), ("exit", cx, sx, yx, gx)]:
            r = fit(np.asarray(xx), np.asarray(ss), np.asarray(yy), gg)
            if r is not None:
                rows.append({"condition": cond, "model": "KRW-switch", "kind": kind,
                             "b_ctx": r.params[1], "se_ctx": r.bse[1], "p_ctx": r.pvalues[1],
                             "b_speed": r.params[2], "se_speed": r.bse[2], "p_speed": r.pvalues[2], "n": int(r.nobs)})
        del sim
    d = pd.DataFrame(rows)
    print(d.round(3).to_string(index=False))
    d.to_csv(ROOT / "Fitting" / "ctx_transition_test.csv", index=False)


if __name__ == "__main__":
    main()
