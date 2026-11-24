# -*- coding: utf-8 -*-
"""State-conditional speed and angular-velocity statistics.

Compares real HMM PR/non-PR states with KRW-switch state 1/0. Speed is
time-bin residualized within each dataset to remove the shared speed profile.
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
import hmm_pr_analysis as hmm
from kinematic_utils import speed_profile_from_cache

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
OUT = Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\周老师第5点_时间相关性_待审\状态条件运动学")
DATA = OUT / "data"; FIG = OUT / "figures"; TAB = OUT / "tables"
for d in [DATA, FIG, TAB]:
    d.mkdir(parents=True, exist_ok=True)

CONDS = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
FNAME = {"3g-1gNa-1h": "3g-1gNa-1h", "3g-1gNa-4h": "3g-1gNa-4h", "1gNa-end(8x)": "1gNa-end8x"}
LABEL = {"3g-1gNa-1h": "1 h", "3g-1gNa-4h": "4 h", "1gNa-end(8x)": "8 h"}
N_SIM = 50
BIN = 4  # 1 s at the 0.25 s analysis resolution

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def residualize_speed(worms):
    """Subtract the condition-wide mean speed in each 1 s time bin."""
    sums = {}
    counts = {}
    for w in worms:
        s = w["speed"]
        for i, v in enumerate(s):
            b = i // BIN
            if np.isfinite(v):
                sums[b] = sums.get(b, 0.0) + float(v)
                counts[b] = counts.get(b, 0) + 1
    means = {b: sums[b] / counts[b] for b in sums}
    out = []
    for w in worms:
        s = np.asarray(w["speed"], float)
        b = np.arange(len(s)) // BIN
        mu = np.array([means.get(int(x), np.nan) for x in b])
        out.append(s - mu)
    return out


def per_worm_state_means(worms, states, speed_res, agl_abs):
    rows = []
    for w, st, sp, ag in zip(worms, states, speed_res, agl_abs):
        st = np.asarray(st, int)
        sp = np.asarray(sp, float)
        ag = np.asarray(ag, float)
        m0 = st == 0
        m1 = st == 1
        if m0.sum() < 5 or m1.sum() < 5:
            continue
        raw = np.asarray(w["speed"], float)
        rows.append({
            "worm": w["worm_id"],
            "speed0": np.nanmean(sp[m0]), "speed1": np.nanmean(sp[m1]),
            "rawspeed0": np.nanmean(raw[m0]), "rawspeed1": np.nanmean(raw[m1]),
            "agl0": np.nanmean(ag[m0]), "agl1": np.nanmean(ag[m1]),
            "n0": int(m0.sum()), "n1": int(m1.sum()),
        })
    return pd.DataFrame(rows)


def summarize(df, cond, model):
    d = df.copy()
    d["dspeed"] = d.speed1 - d.speed0
    d["dagl"] = d.agl1 - d.agl0
    rec = {"condition": cond, "model": model, "n": len(d)}
    d["drawspeed"] = d.rawspeed1 - d.rawspeed0
    for k in ["speed0", "speed1", "dspeed", "rawspeed0", "rawspeed1", "drawspeed", "agl0", "agl1", "dagl"]:
        rec[k] = float(d[k].mean())
        rec[k + "_sem"] = float(d[k].std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    rec["speed_pct"] = 100.0 * rec["dspeed"] / abs(rec["speed0"]) if abs(rec["speed0"]) > 1e-9 else np.nan
    rec["agl_pct"] = 100.0 * rec["dagl"] / abs(rec["agl0"]) if abs(rec["agl0"]) > 1e-9 else np.nan
    return rec


def main():
    params = pd.read_csv(ROOT / "Fitting" / "SingleConc_resultsV13_CV_features" / "ALL_CONDITIONS_results.csv")
    hmm_sum = pd.read_csv(Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\审阅版\figures\value_state_analysis\hmm_pr_summary.csv"))
    summary = []
    per_worm_all = []
    for cond in CONDS:
        print("=" * 40, cond)
        with open(ROOT / "Fitting" / "hmm_seq_cache" / (FNAME[cond] + ".pkl"), "rb") as f:
            seq = pickle.load(f)

        # real HMM states at 0.25 s
        hr = hmm_sum[hmm_sum.condition == cond].iloc[0]
        pi = np.array([1 - hr.pi_pr, hr.pi_pr])
        A = np.array([[hr.A_00, hr.A_01], [hr.A_10, hr.A_11]])
        real_worms = []
        real_states = []
        real_agl = []
        for o in seq:
            sp = o["speed"][::5]
            ag = np.abs(o["agl"])[::5]
            y = (ag >= hr.threshold_value).astype(np.int8)
            st = hmm.viterbi(y, pi, A, hmm.P0, hmm.P1)
            real_worms.append({"worm_id": o["worm_id"], "speed": sp})
            real_states.append(st)
            real_agl.append(ag)
        real_res = residualize_speed(real_worms)
        rw = per_worm_state_means(real_worms, real_states, real_res, real_agl)
        summary.append(summarize(rw, cond, "Real HMM"))
        rw["condition"] = cond; rw["model"] = "Real HMM"
        per_worm_all.append(rw)

        # switch simulation
        p = params[(params.condition == cond) & (params.model == "KRW_switch")].iloc[0].to_dict()
        profile = speed_profile_from_cache(seq)
        sim = dod.simulate_population(p, "KRW_switch", N_SIM, dod.MAX_TIMESTAMP, dod.DT_REAL,
                                      profile, None, n_jobs=1, seed_base=20261116, return_xy=True)
        sw_worms = []
        sw_states = []
        sw_agl = []
        for wid, g in sim.groupby("ID"):
            sw_worms.append({"worm_id": wid, "speed": g["speed"].to_numpy()[::5]})
            sw_states.append(g["state"].to_numpy()[::5].astype(int))
            sw_agl.append(g["abs_agl_velocity"].to_numpy()[::5])
        sw_res = residualize_speed(sw_worms)
        sw = per_worm_state_means(sw_worms, sw_states, sw_res, sw_agl)
        summary.append(summarize(sw, cond, "KRW-switch"))
        sw["condition"] = cond; sw["model"] = "KRW-switch"
        per_worm_all.append(sw)
        del sim

    s = pd.DataFrame(summary)
    s.to_csv(DATA / "state_conditional_summary.csv", index=False)
    pd.concat(per_worm_all, ignore_index=True).to_csv(DATA / "state_conditional_per_worm.csv", index=False)
    print(s.round(3).to_string(index=False))

    # figure: 2 x 3 panels
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), sharex="col")
    colors = {"Real HMM": ["#BDBDBD", "#4D4D4D"], "KRW-switch": ["#CAB2D6", "#6A3D9A"]}
    labels = ["非 PR", "PR", "非 PR", "PR"]
    for j, cond in enumerate(CONDS):
        for i, (metric, ylab) in enumerate([("speed", "速度残差"), ("agl", "|角速度|")]):
            ax = axes[i, j]
            for k, model in enumerate(["Real HMM", "KRW-switch"]):
                r = s[(s.condition == cond) & (s.model == model)].iloc[0]
                vals = [r[metric + "0"], r[metric + "1"]]
                errs = [r[metric + "0_sem"], r[metric + "1_sem"]]
                ax.bar(np.arange(2) + 2 * k, vals, 0.7, yerr=errs, capsize=2,
                       color=colors[model], edgecolor="white", linewidth=0.3)
            ax.axhline(0, color="gray", lw=0.6)
            ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(labels, fontsize=7)
            ax.text(0.5, 0.96, "真实 HMM", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=7)
            ax.text(2.5, 0.96, "KRW-switch", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=7)
            if i == 1:
                ax.set_xlabel(LABEL[cond])
            if j == 0:
                ax.set_ylabel(ylab)
    fig.tight_layout()
    fig.savefig(FIG / "fig_state_conditional.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_state_conditional.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
