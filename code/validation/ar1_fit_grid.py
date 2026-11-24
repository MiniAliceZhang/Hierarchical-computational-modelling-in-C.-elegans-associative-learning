# -*- coding: utf-8 -*-
"""Fit the two AR(1) control parameters to the real angular-velocity ACF."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import DistributionOverlapDecision as dod
from kinematic_utils import omega_metrics, speed_profile_from_cache

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
XY = ROOT / "Fitting" / "xy_seq_cache"
SEQ = ROOT / "Fitting" / "hmm_seq_cache"
PARAMS = ROOT / "Fitting" / "SingleConc_resultsV13_CV_features" / "ALL_CONDITIONS_results.csv"
OUT = ROOT / "Fitting" / "ar1_fit_grid.csv"
CONDS = [("3g-1gNa-1h", "3g-1gNa-1h"), ("3g-1gNa-4h", "3g-1gNa-4h"), ("1gNa-end(8x)", "1gNa-end8x")]
RHOS = [0.9, 0.95, 0.98]
SIGMAS = [1.5, 2.5]
N_FIT = 2


def main():
    params = pd.read_csv(PARAMS)
    rows = []
    for cond, fname in CONDS:
        with open(XY / (fname + ".pkl"), "rb") as f:
            xy = pickle.load(f)
        with open(SEQ / (fname + ".pkl"), "rb") as f:
            seq = pickle.load(f)
        profile = speed_profile_from_cache(seq)
        real = [omega_metrics(o["x"], o["y"]) for o in xy]
        real_mean = np.mean([r["mean"] for r in real])
        real_std = np.mean([r["std"] for r in real])
        real_acf = np.nanmean([r["acf"] for r in real], axis=0)
        p = params[(params.condition == cond) & (params.model == "KRW_switch")].iloc[0].to_dict()
        for rho in RHOS:
            for sigma in SIGMAS:
                pp = {**p, "rho_ar": rho, "sigma_ar": sigma}
                sim = dod.simulate_population(pp, "KRW_AR1", N_FIT, dod.MAX_TIMESTAMP, dod.DT_REAL,
                                              profile, None, n_jobs=1, seed_base=20261116, return_xy=True)
                sims = [omega_metrics(g["x"], g["y"]) for _, g in sim.groupby("ID")]
                sm = np.mean([r["mean"] for r in sims])
                ss = np.mean([r["std"] for r in sims])
                sa = np.nanmean([r["acf"] for r in sims], axis=0)
                acf_loss = float(np.nanmean((sa[:11] - real_acf[:11]) ** 2))
                mean_loss = ((sm - real_mean) / max(real_mean, 1e-9)) ** 2
                std_loss = ((ss - real_std) / max(real_std, 1e-9)) ** 2
                rows.append({
                    "condition": cond, "rho_ar": rho, "sigma_ar": sigma,
                    "real_mean": real_mean, "real_std": real_std,
                    "sim_mean": sm, "sim_std": ss,
                    "acf_loss": acf_loss, "mean_loss": mean_loss, "std_loss": std_loss,
                    "loss": acf_loss + 0.5 * mean_loss + 0.5 * std_loss,
                })
                print(cond, rho, sigma, round(rows[-1]["loss"], 4), round(sm, 2), round(ss, 2), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    for cond, _ in CONDS:
        print("\n", cond)
        print(out[out.condition == cond].sort_values("loss").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
