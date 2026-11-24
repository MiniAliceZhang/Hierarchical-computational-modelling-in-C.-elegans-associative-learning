# -*- coding: utf-8 -*-
"""Real-data kinematic PR proxy vs model state PR fraction."""
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
FRAME_RATE = 20
MAX_TS = 14400
MERGE_GAP = int(3.8 * FRAME_RATE)

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def merge_events(mask, gap):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return mask
    out = mask.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= gap:
            out[a:b + 1] = True
    return out


def summarize_real():
    with open(ROOT / "data" / "condition_dict.pkl", "rb") as f:
        cd = pickle.load(f)
    rows = []
    for cond in CONDS:
        # pooled threshold from the first 12 min
        vals = []
        for df in cd[cond]:
            if "Timestamp" in df:
                df = df[df["Timestamp"] <= MAX_TS]
            vals.append(np.abs(df["agl_velocity"].dropna().to_numpy()))
        vals = np.concatenate(vals)
        thresholds = {q: float(np.percentile(vals, q)) for q in [90, 95]}
        for q, th in thresholds.items():
            for j, df in enumerate(cd[cond]):
                if "Timestamp" in df:
                    df = df[df["Timestamp"] <= MAX_TS]
                for wid, g in df.groupby("ID"):
                    v = np.abs(g["agl_velocity"].to_numpy())
                    mask = v >= th
                    mask = merge_events(mask, MERGE_GAP)
                    frac = float(mask.mean())
                    # event count and mean duration
                    padded = np.r_[False, mask, False]
                    starts = np.where(~padded[:-1] & padded[1:])[0]
                    ends = np.where(padded[:-1] & ~padded[1:])[0]
                    lens = ends - starts
                    total_min = max(len(mask) / FRAME_RATE / 60.0, 1e-9)
                    rows.append({
                        "condition": cond, "threshold_q": q, "threshold_value": th,
                        "worm_id": wid, "df_idx": j,
                        "real_pr_fraction": frac,
                        "real_pr_events_per_min": len(lens) / total_min,
                        "real_pr_duration_s": float(np.mean(lens) / FRAME_RATE) if len(lens) else 0.0,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "real_pr_proxy_per_worm.csv", index=False)
    summ = []
    for (cond, q), g in df.groupby(["condition", "threshold_q"]):
        rec = {"condition": cond, "threshold_q": q, "threshold_value": g["threshold_value"].iloc[0], "n": len(g)}
        for v in ["real_pr_fraction", "real_pr_events_per_min", "real_pr_duration_s"]:
            rec[v] = float(g[v].mean())
            rec[v + "_sem"] = float(g[v].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0
        summ.append(rec)
    summary = pd.DataFrame(summ)
    summary.to_csv(OUT / "real_pr_proxy_summary.csv", index=False)
    return summary


def main():
    real = summarize_real()
    model = pd.read_csv(OUT / "value_state_summary.csv")
    model = model[model["model"].isin(["KRW_switch_full", "KRW_switch_no_value", "RW_switch_full"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    conds = CONDS
    x = np.arange(len(conds))
    width = 0.18
    series = []
    for q in [90, 95]:
        series.append((f"真实 PR 代理（q{q}）", [real[(real.condition == c) & (real.threshold_q == q)]["real_pr_fraction"].iloc[0] for c in conds],
                       [real[(real.condition == c) & (real.threshold_q == q)]["real_pr_fraction_sem"].iloc[0] for c in conds],
                       "#4D4D4D" if q == 90 else "#969696"))
    for m, lab, col in [("KRW_switch_full", "KRW-switch 全模型", "#6A3D9A"),
                        ("KRW_switch_no_value", "KRW-switch 无价值项", "#CAB2D6"),
                        ("RW_switch_full", "RW-switch 全模型", "#1F78B4")]:
        vals = [model[(model.condition == c) & (model.model == m)]["state_pr_fraction"].iloc[0] for c in conds]
        errs = [model[(model.condition == c) & (model.model == m)]["state_pr_fraction_sem"].iloc[0] for c in conds]
        series.append((lab, vals, errs, col))
    for i, (lab, vals, errs, col) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, yerr=errs, capsize=2,
               label=lab, color=col, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c] for c in conds])
    ax.set_ylabel("PR 时间占比")
    ax.set_title("真实 PR 代理与模型状态 PR 时间占比", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pr_real_vs_model.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_pr_real_vs_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(real.to_string(index=False))
    print("saved", OUT)


if __name__ == "__main__":
    main()
