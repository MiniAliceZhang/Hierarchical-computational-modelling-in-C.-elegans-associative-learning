# -*- coding: utf-8 -*-
"""
Value representation -> behavioral state transition analysis.

Produces:
1. Parameter ablation of the value terms in the PR logit:
   full / no interaction / no value.
2. Kinematic PR proxy statistics for real and simulated trajectories.
3. Figures and CSV tables for the thesis.

The kinematic PR proxy is a high-angle turn event: heading change > 90 deg
over a 1 s window (20 frames at 20 Hz). It is not a ground-truth PR label;
it is used because the raw tracking files do not contain PR/Run annotations.
"""
import gc
import json
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

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
CONDITIONS = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
COND_LABEL = {"3g-1gNa-1h": "1 h", "3g-1gNa-4h": "4 h", "1gNa-end(8x)": "8 h"}
MODELS = ["KRW_switch", "RW_switch"]
OUT = Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\审阅版\figures\value_state_analysis")
OUT.mkdir(parents=True, exist_ok=True)

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

WINDOW = 20          # 1 s at 20 Hz
ANGLE_THRESH = 90.0  # degrees
DT = dod.DT_REAL


def run_stats(mask):
    """Return (fraction, n_events, mean_duration_steps)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.nan, 0, np.nan
    frac = float(mask.mean())
    # run lengths
    padded = np.r_[False, mask, False]
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0]
    lengths = ends - starts
    if len(lengths) == 0:
        return frac, 0, np.nan
    return frac, int(len(lengths)), float(np.mean(lengths))


def proxy_stats(df, window=WINDOW, thresh=ANGLE_THRESH, dt=DT, id_col="ID"):
    rows = []
    for wid, g in df.groupby(id_col, sort=False):
        g = g.dropna(subset=["x", "y"])
        x = g["x"].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)
        if len(x) <= window + 3:
            continue
        dx = x[window:] - x[:-window]
        dy = y[window:] - y[:-window]
        amp = np.hypot(dx, dy)
        heading = np.unwrap(np.arctan2(dy, dx))
        dth = np.diff(heading)
        valid = amp[1:] > 1.0
        mask = (np.abs(np.degrees(dth)) > thresh) & valid
        frac, nev, dur_steps = run_stats(mask)
        total_s = max(len(mask) * window * dt, 1e-9)
        rows.append({
            "worm_id": wid,
            "proxy_pr_fraction": frac,
            "proxy_pr_events": nev,
            "proxy_pr_events_per_min": nev / (total_s / 60.0),
            "proxy_pr_duration_s": dur_steps * window * dt if np.isfinite(dur_steps) else np.nan,
            "n_windows": len(mask),
        })
    return pd.DataFrame(rows)


def state_stats(df, id_col="ID", dt=DT):
    rows = []
    for wid, g in df.groupby(id_col, sort=False):
        s = g["state"].to_numpy(dtype=int)
        frac, nev, dur_steps = run_stats(s == 1)
        total_s = max(len(s) * dt, 1e-9)
        rows.append({
            "worm_id": wid,
            "state_pr_fraction": frac,
            "state_pr_events": nev,
            "state_pr_events_per_min": nev / (total_s / 60.0),
            "state_pr_duration_s": dur_steps * dt if np.isfinite(dur_steps) else np.nan,
            "n_frames": len(s),
        })
    return pd.DataFrame(rows)


def agg(df, value):
    x = df[value].dropna()
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def plot_grouped(summary, value, ylabel, path, models, title=None):
    conds = CONDITIONS
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    n = len(models)
    width = 0.8 / n
    xpos = np.arange(len(conds))
    colors = {"Real": "#4D4D4D", "KRW_switch_full": "#6A3D9A", "KRW_switch_no_interaction": "#CAB2D6",
              "KRW_switch_no_value": "#BDBDBD", "RW_switch_full": "#1F78B4",
              "RW_switch_no_interaction": "#A6CEE3", "RW_switch_no_value": "#D9D9D9"}
    for i, m in enumerate(models):
        vals = [summary[(summary["condition"] == c) & (summary["model"] == m)][value].iloc[0]
                if ((summary["condition"] == c) & (summary["model"] == m)).any() else np.nan
                for c in conds]
        errs = [summary[(summary["condition"] == c) & (summary["model"] == m)][value + "_sem"].iloc[0]
                if ((summary["condition"] == c) & (summary["model"] == m)).any() else np.nan
                for c in conds]
        ax.bar(xpos + (i - (n - 1) / 2) * width, vals, width, yerr=errs, capsize=3,
               label=m, color=colors.get(m, None), edgecolor="white", linewidth=0.3)
    ax.set_xticks(xpos)
    ax.set_xticklabels([COND_LABEL[c] for c in conds])
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, ncol=1)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    with open(ROOT / "data" / "condition_dict.pkl", "rb") as f:
        condition_dict = pickle.load(f)

    params = pd.read_csv(ROOT / "Fitting/SingleConc_resultsV13_CV_features/ALL_CONDITIONS_results.csv")
    rows = []
    real_rows = []

    for cond in CONDITIONS:
        print("=" * 60)
        print(cond)
        dfs = condition_dict[cond]
        # Process each dataframe separately to avoid concatenating the full 90-min table.
        speed_parts = []
        n_real = 0
        for j, df in enumerate(dfs):
            use = df[["X", "Y", "ID"]].rename(columns={"X": "x", "Y": "y"})
            rp = proxy_stats(use)
            rp["condition"] = cond
            rp["model"] = "Real"
            rp["df_idx"] = j
            real_rows.append(rp)
            n_real += len(rp)
            if "speed" in df.columns and "Timestamp" in df.columns:
                sub = df[["speed", "Timestamp"]].dropna()
                if len(sub) > 200000:
                    sub = sub.iloc[:: max(1, len(sub) // 200000)]
                speed_parts.append(sub.astype({"speed": "float32", "Timestamp": "float32"}))
        # Average normalized speed profile from a sample of all real frames.
        sp = pd.concat(speed_parts, ignore_index=True)
        nt = sp["Timestamp"].to_numpy(dtype=float)
        nt = nt / max(nt.max(), 1e-9)
        bins = np.linspace(0, 1, 51)
        centers = (bins[:-1] + bins[1:]) / 2
        means = []
        sv = sp["speed"].to_numpy(dtype=float)
        for i in range(50):
            m = (nt >= bins[i]) & (nt < bins[i + 1])
            means.append(float(np.nanmean(sv[m])) if m.sum() else np.nan)
        means = np.array(means, dtype=float)
        if np.isnan(means).any():
            ok = ~np.isnan(means)
            means = np.interp(centers, centers[ok], means[ok])
        speed_profile = (centers, means)
        deadzone_speed = None
        print("real worms", n_real, "proxy frac", pd.concat(real_rows)["proxy_pr_fraction"].mean())

        n_sim = min(60, max(1, n_real))
        for model in MODELS:
            row = params[(params["condition"] == cond) & (params["model"] == model)]
            if row.empty:
                continue
            p_full = row.iloc[0].to_dict()
            variants = {
                f"{model}_full": dict(p_full),
                f"{model}_no_interaction": {**p_full, "a_interaction": 0.0},
                f"{model}_no_value": {**p_full, "a_reward": 0.0, "a_interaction": 0.0},
            }
            for name, p_use in variants.items():
                print("simulate", cond, name)
                sim = dod.simulate_population(
                    p_use, model, n_sim, dod.MAX_TIMESTAMP, dod.DT_REAL,
                    speed_profile, deadzone_speed, n_jobs=1, seed_base=20261116, return_xy=True
                )
                ss = state_stats(sim)
                ps = proxy_stats(sim)
                m = ss.merge(ps, on="worm_id", how="inner")
                m["condition"] = cond
                m["model"] = name
                rows.append(m)
                print(name, "state frac", m["state_pr_fraction"].mean(),
                      "proxy frac", m["proxy_pr_fraction"].mean())
                del sim
                gc.collect()
        del dfs, sp, speed_parts
        gc.collect()

    all_rows = pd.concat(rows, ignore_index=True)
    real_all = pd.concat(real_rows, ignore_index=True)
    all_rows.to_csv(OUT / "per_worm_value_state_stats.csv", index=False)
    real_all.to_csv(OUT / "per_worm_real_pr_proxy.csv", index=False)

    # Summary
    summ = []
    for (cond, model), g in all_rows.groupby(["condition", "model"]):
        rec = {"condition": cond, "model": model, "n": len(g)}
        for v in ["state_pr_fraction", "state_pr_events_per_min", "state_pr_duration_s",
                  "proxy_pr_fraction", "proxy_pr_events_per_min", "proxy_pr_duration_s"]:
            mean, sem = agg(g, v)
            rec[v] = mean
            rec[v + "_sem"] = sem
        summ.append(rec)
    for (cond, model), g in real_all.groupby(["condition", "model"]):
        rec = {"condition": cond, "model": model, "n": len(g)}
        for v in ["proxy_pr_fraction", "proxy_pr_events_per_min", "proxy_pr_duration_s"]:
            mean, sem = agg(g, v)
            rec[v] = mean
            rec[v + "_sem"] = sem
        for v in ["state_pr_fraction", "state_pr_events_per_min", "state_pr_duration_s"]:
            rec[v] = np.nan
            rec[v + "_sem"] = np.nan
        summ.append(rec)
    summary = pd.DataFrame(summ)
    summary.to_csv(OUT / "value_state_summary.csv", index=False)

    # Figure 1: explicit state PR fraction + proxy fraction
    plot_grouped(
        summary, "state_pr_fraction", "模型状态中 PR 时间占比", OUT / "fig_value_state_ablation_state.pdf",
        ["KRW_switch_full", "KRW_switch_no_interaction", "KRW_switch_no_value"],
        "KRW-switch 中移除价值项后的状态统计"
    )
    plot_grouped(
        summary, "proxy_pr_fraction", "高角速度转向事件时间占比", OUT / "fig_value_state_ablation_proxy.pdf",
        ["Real", "KRW_switch_full", "KRW_switch_no_value"],
        "真实轨迹与 KRW-switch 的 PR 代理统计"
    )

    # Figure 2: event rate and duration
    plot_grouped(
        summary, "proxy_pr_events_per_min", "高角速度转向事件（次/分钟）", OUT / "fig_pr_event_rate.pdf",
        ["Real", "KRW_switch_full"], "真实与模拟轨迹的 PR 代理事件率"
    )
    plot_grouped(
        summary, "proxy_pr_duration_s", "高角速度转向事件持续时间（s）", OUT / "fig_pr_duration.pdf",
        ["Real", "KRW_switch_full"], "真实与模拟轨迹的 PR 代理持续时间"
    )

    # Figure 3: coupling parameters
    p2 = params[params["model"].isin(MODELS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    x = np.arange(len(CONDITIONS))
    width = 0.36
    for ax, col, ylab in zip(axes, ["a_reward", "a_interaction"], ["$a_{reward}$", "$a_{interaction}$"]):
        for i, model in enumerate(MODELS):
            vals = []
            for c in CONDITIONS:
                r = p2[(p2["condition"] == c) & (p2["model"] == model)]
                vals.append(float(r[col].iloc[0]) if len(r) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width, label=model,
                   color=["#6A3D9A", "#1F78B4"][i], edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_coupling_parameters.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_coupling_parameters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    key = {}
    for _, r in summary.iterrows():
        key[f"{r['condition']}|{r['model']}"] = {
            k: (None if pd.isna(r[k]) else float(r[k]))
            for k in summary.columns if k not in ["condition", "model"]
        }
    (OUT / "key_numbers.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", OUT)


if __name__ == "__main__":
    main()
