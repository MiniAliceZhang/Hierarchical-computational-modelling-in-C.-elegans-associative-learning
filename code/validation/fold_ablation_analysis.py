# -*- coding: utf-8 -*-
"""Fold-wise fixed-parameter ablation for KRW-switch."""
import gc
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

import DistributionOverlapDecision as dod
import value_state_link_analysis as vs

OUT = Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\审阅版\figures\value_state_analysis")
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(r"E:\ZHY\code\AssociativeModel")
CONDS = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
LABEL = {"3g-1gNa-1h": "1 h", "3g-1gNa-4h": "4 h", "1gNa-end(8x)": "8 h"}
N_WORMS = 20

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def speed_profile_for(dfs):
    parts = []
    for df in dfs:
        if "speed" in df.columns and "Timestamp" in df.columns:
            sub = df[["speed", "Timestamp"]].dropna()
            if len(sub) > 200000:
                sub = sub.iloc[:: max(1, len(sub) // 200000)]
            parts.append(sub.astype({"speed": "float32", "Timestamp": "float32"}))
    sp = pd.concat(parts, ignore_index=True)
    nt = sp["Timestamp"].to_numpy(float)
    nt = nt / max(nt.max(), 1e-9)
    bins = np.linspace(0, 1, 51)
    centers = (bins[:-1] + bins[1:]) / 2
    sv = sp["speed"].to_numpy(float)
    means = []
    for i in range(50):
        m = (nt >= bins[i]) & (nt < bins[i + 1])
        means.append(float(np.nanmean(sv[m])) if m.sum() else np.nan)
    means = np.array(means)
    ok = ~np.isnan(means)
    if ok.any():
        means = np.interp(centers, centers[ok], means[ok])
    return (centers, means)


def main():
    with open(ROOT / "data" / "condition_dict.pkl", "rb") as f:
        cd = pickle.load(f)
    cv = pd.read_csv(ROOT / "Fitting/SingleConc_resultsV13_CV_features/CV_FITTED_PARAMS.csv")
    rows = []
    for cond in CONDS:
        profile = speed_profile_for(cd[cond])
        for fold in range(5):
            base = cv[(cv.condition == cond) & (cv.model == "KRW_switch") & (cv.fold == fold)]
            if base.empty:
                continue
            p0 = base.iloc[0].to_dict()
            variants = {
                "full": dict(p0),
                "no_interaction": {**p0, "a_interaction": 0.0},
                "no_value": {**p0, "a_reward": 0.0, "a_interaction": 0.0},
            }
            for name, p in variants.items():
                print(cond, "fold", fold, name)
                sim = dod.simulate_population(
                    p, "KRW_switch", N_WORMS, dod.MAX_TIMESTAMP, dod.DT_REAL,
                    profile, None, n_jobs=1, seed_base=20261116, return_xy=True
                )
                st = vs.state_stats(sim)
                st["condition"] = cond
                st["fold"] = fold
                st["variant"] = name
                rows.append(st)
                del sim
                gc.collect()
        del profile
        gc.collect()
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(OUT / "fold_ablation_per_worm.csv", index=False)
    summ = []
    for (cond, fold, var), g in all_df.groupby(["condition", "fold", "variant"]):
        summ.append({
            "condition": cond, "fold": fold, "variant": var,
            "state_pr_fraction": g["state_pr_fraction"].mean(),
            "state_pr_events_per_min": g["state_pr_events_per_min"].mean(),
            "state_pr_duration_s": g["state_pr_duration_s"].mean(),
        })
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "fold_ablation_summary.csv", index=False)
    print(summ.groupby(["condition", "variant"])["state_pr_fraction"].agg(["mean", "min", "max"]))
    # figure: per-condition mean with fold range
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(CONDS))
    w = 0.25
    colors = {"full": "#6A3D9A", "no_interaction": "#CAB2D6", "no_value": "#BDBDBD"}
    labels = {"full": "全模型", "no_interaction": "去掉交互项", "no_value": "去掉价值项"}
    for i, var in enumerate(["full", "no_interaction", "no_value"]):
        means, lows, highs = [], [], []
        for cond in CONDS:
            g = summ[(summ.condition == cond) & (summ.variant == var)]["state_pr_fraction"]
            means.append(g.mean())
            lows.append(g.mean() - g.min())
            highs.append(g.max() - g.mean())
        ax.bar(x + (i - 1) * w, means, w, yerr=[lows, highs], capsize=3,
               label=labels[var], color=colors[var], edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c] for c in CONDS])
    ax.set_ylabel("模型状态中 PR 时间占比")
    ax.set_title("5 折参数集上的价值项消融", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fold_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_fold_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
