# -*- coding: utf-8 -*-
"""Post-hoc calibration of PR duration and entry in KRW-switch.

Value and motor parameters are held fixed. Per condition, min_pir_steps is
chosen to match the HMM-inferred PR duration, then a0 is shifted to match the
HMM-inferred PR fraction. This is a sensitivity analysis, not a refit.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import DistributionOverlapDecision as dod
from kinematic_utils import speed_profile_from_cache

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
OUT = Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\周老师第5点_时间相关性_待审\状态统计校准")
DATA = OUT / "data"; FIG = OUT / "figures"; TAB = OUT / "tables"
for d in [DATA, FIG, TAB]:
    d.mkdir(parents=True, exist_ok=True)

CONDS = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
FNAME = {"3g-1gNa-1h": "3g-1gNa-1h", "3g-1gNa-4h": "3g-1gNa-4h", "1gNa-end(8x)": "1gNa-end8x"}
LABEL = {"3g-1gNa-1h": "1 h", "3g-1gNa-4h": "4 h", "1gNa-end(8x)": "8 h"}
N_CAL = 8
N_FINAL = 50
MIN_GRID = [10, 12, 15, 18, 22, 26, 30, 35]
A0_OFFSETS = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0]

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def stats_and_features(sim):
    rows = []
    for _, g in sim.groupby("ID"):
        s = g["state"].to_numpy(int)
        p = np.r_[False, s == 1, False]
        st = np.where(~p[:-1] & p[1:])[0]
        en = np.where(p[:-1] & ~p[1:])[0]
        lens = en - st
        rows.append({
            "fraction": float(np.mean(s == 1)),
            "events_per_min": len(lens) / max(len(s) * 0.05 / 60.0, 1e-9),
            "duration_s": float(np.mean(lens) * 0.05) if len(lens) else 0.0,
            "mean_CTX": float(g["CTX"].mean()),
            "mean_abs_agl": float(g["abs_agl_velocity"].mean()),
            "net_displacement": float(g["net_displacement"].iloc[0]),
            "mean_dist_to_midline": float(g["mean_dist_to_midline"].iloc[0]),
            "state0_agl": float(g.loc[g.state == 0, "abs_agl_velocity"].mean()) if (g.state == 0).any() else np.nan,
            "state1_agl": float(g.loc[g.state == 1, "abs_agl_velocity"].mean()) if (g.state == 1).any() else np.nan,
        })
    d = pd.DataFrame(rows)
    return d


def simulate(p, profile, n):
    return dod.simulate_population(p, "KRW_switch", n, dod.MAX_TIMESTAMP, dod.DT_REAL,
                                   profile, None, n_jobs=1, seed_base=20261116, return_xy=True)


def main():
    params = pd.read_csv(ROOT / "Fitting" / "SingleConc_resultsV13_CV_features" / "ALL_CONDITIONS_results.csv")
    hmm = pd.read_csv(Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\审阅版\figures\value_state_analysis\hmm_pr_summary.csv"))
    grid_rows = []
    best_rows = []
    final_rows = []
    for cond in CONDS:
        print("=" * 40, cond)
        with open(ROOT / "Fitting" / "hmm_seq_cache" / (FNAME[cond] + ".pkl"), "rb") as f:
            seq = pickle.load(f)
        profile = speed_profile_from_cache(seq)
        p0 = params[(params.condition == cond) & (params.model == "KRW_switch")].iloc[0].to_dict()
        hr = hmm[hmm.condition == cond].iloc[0]
        target_frac = float(hr.hmm_pr_fraction)
        target_dur = float(hr.hmm_pr_duration_s)

        # Stage 1: duration via min_pir_steps
        dur_grid = []
        for ms in MIN_GRID:
            p = {**p0, "min_pir_steps": ms}
            sim = simulate(p, profile, N_CAL)
            d = stats_and_features(sim)
            dur = d.duration_s.mean()
            dur_grid.append({"condition": cond, "stage": "duration", "min_pir_steps": ms,
                             "a0": p0["a0"], "fraction": d.fraction.mean(), "duration_s": dur,
                             "events_per_min": d.events_per_min.mean()})
            print("  ms", ms, "dur", round(dur, 3), "frac", round(d.fraction.mean(), 3))
            del sim
        dur_df = pd.DataFrame(dur_grid)
        grid_rows.append(dur_df)
        best_ms = float(dur_df.iloc[(dur_df.duration_s - target_dur).abs().argmin()].min_pir_steps)

        # Stage 2: fraction via a0 shift at fixed duration
        a0_grid = []
        for off in A0_OFFSETS:
            p = {**p0, "min_pir_steps": int(best_ms), "a0": float(p0["a0"] + off)}
            sim = simulate(p, profile, N_CAL)
            d = stats_and_features(sim)
            a0_grid.append({"condition": cond, "stage": "fraction", "min_pir_steps": int(best_ms),
                            "a0": p["a0"], "fraction": d.fraction.mean(),
                            "duration_s": d.duration_s.mean(), "events_per_min": d.events_per_min.mean()})
            print("  a0", round(p["a0"], 3), "frac", round(d.fraction.mean(), 3), "dur", round(d.duration_s.mean(), 3))
            del sim
        a_df = pd.DataFrame(a0_grid)
        grid_rows.append(a_df)
        best = a_df.iloc[(a_df.fraction - target_frac).abs().argmin()]
        best_rows.append(best.to_dict())

        # final simulation
        p = {**p0, "min_pir_steps": int(best.min_pir_steps), "a0": float(best.a0)}
        sim = simulate(p, profile, N_FINAL)
        d = stats_and_features(sim)
        rec = {"condition": cond, "model": "KRW-switch calibrated", "n": len(d),
               "pr_fraction": d.fraction.mean(), "duration_s": d.duration_s.mean(),
               "events_per_min": d.events_per_min.mean(),
               "CTX": d.mean_CTX.mean(), "abs_agl": d.mean_abs_agl.mean(),
               "net_displacement": d.net_displacement.mean(),
               "midline_distance": d.mean_dist_to_midline.mean(),
               "state0_agl": d.state0_agl.mean(), "state1_agl": d.state1_agl.mean(),
               "min_pir_steps": int(best.min_pir_steps), "a0": float(best.a0)}
        final_rows.append(rec)
        print("FINAL", rec)
        del sim

    grid = pd.concat(grid_rows, ignore_index=True)
    grid.to_csv(DATA / "calibration_grid.csv", index=False)
    pd.DataFrame(best_rows).to_csv(DATA / "calibration_best_per_condition.csv", index=False)
    final = pd.DataFrame(final_rows)

    # compare with real and original switch
    orig_stats = pd.read_csv(Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\审阅版\figures\value_state_analysis\value_state_summary.csv"))
    orig_features = pd.read_csv(ROOT / "Fitting" / "SingleConc_resultsV13_analysis" / "PER_WORM_FEATURE_TABLE.csv")
    comp = []
    for cond in CONDS:
        hr = hmm[hmm.condition == cond].iloc[0]
        rf = orig_features[(orig_features.condition == cond) & (orig_features.model == "Real Data")]
        comp.append({"condition": cond, "model": "Real HMM", "n": int(hr.n),
                     "pr_fraction": hr.hmm_pr_fraction, "duration_s": hr.hmm_pr_duration_s,
                     "events_per_min": hr.hmm_pr_events_per_min,
                     "CTX": rf.mean_CTX.mean(), "abs_agl": rf.mean_abs_agl_velocity.mean(),
                     "net_displacement": rf.net_displacement.mean(),
                     "midline_distance": rf.mean_dist_to_midline.mean(),
                     "state0_agl": np.nan, "state1_agl": np.nan,
                     "min_pir_steps": np.nan, "a0": np.nan})
        os = orig_stats[(orig_stats.condition == cond) & (orig_stats.model == "KRW_switch_full")].iloc[0]
        of = orig_features[(orig_features.condition == cond) & (orig_features.model == "KRW_switch")]
        comp.append({"condition": cond, "model": "KRW-switch original", "n": int(os.n),
                     "pr_fraction": os.state_pr_fraction, "duration_s": os.state_pr_duration_s,
                     "events_per_min": os.state_pr_events_per_min,
                     "CTX": of.mean_CTX.mean(), "abs_agl": of.mean_abs_agl_velocity.mean(),
                     "net_displacement": of.net_displacement.mean(),
                     "midline_distance": of.mean_dist_to_midline.mean(),
                     "state0_agl": np.nan, "state1_agl": np.nan,
                     "min_pir_steps": 35, "a0": np.nan})
        comp.append(final[final.condition == cond].iloc[0].to_dict())
    comparison = pd.DataFrame(comp)
    comparison.to_csv(DATA / "calibration_comparison.csv", index=False)
    print(comparison.round(3).to_string(index=False))

    # figure: 2 x 3
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2))
    x = np.arange(3); w = 0.26
    models = ["Real HMM", "KRW-switch original", "KRW-switch calibrated"]
    colors = {"Real HMM": "#4D4D4D", "KRW-switch original": "#CAB2D6", "KRW-switch calibrated": "#6A3D9A"}
    panels = [("pr_fraction", "PR 时间占比", axes[0, 0]),
              ("duration_s", "PR 持续时间（s）", axes[0, 1]),
              ("events_per_min", "PR 事件率（次/分钟）", axes[0, 2]),
              ("CTX", "平均 CTX", axes[1, 0]),
              ("midline_distance", "离中线距离（像素）", axes[1, 1]),
              ("net_displacement", "净位移（像素）", axes[1, 2])]
    for col, lab, ax in panels:
        for i, m in enumerate(models):
            vals = [comparison[(comparison.condition == c) & (comparison.model == m)][col].iloc[0] for c in CONDS]
            ax.bar(x + (i - 1) * w, vals, w, color=colors[m], edgecolor="white", linewidth=0.3, label=m)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in CONDS])
        ax.set_title(lab, fontsize=9); ax.set_ylabel(lab)
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_calibration_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_calibration_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
