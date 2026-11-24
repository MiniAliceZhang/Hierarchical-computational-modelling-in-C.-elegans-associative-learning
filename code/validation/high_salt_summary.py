# -*- coding: utf-8 -*-
"""Descriptive CTX summary for the high-salt groups used to justify scope."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
OUT = Path(r"E:\ZHY\other\博资开题考\转硕\硕士论文模板\00_最终论文\周老师第5点_时间相关性_待审\高盐组说明")
OUT.mkdir(parents=True, exist_ok=True)
DF_SELECTION = list(range(10, 20))
MAX_TS = 14400
LOW = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
HIGH = ["3g-5gNa-1h", "3g-5gNa-4h", "5gNa-end(8x)"]


def main():
    with open(ROOT / "data" / "condition_dict.pkl", "rb") as f:
        cd = pickle.load(f)
    print("keys", list(cd.keys()))
    rows = []
    for cond in LOW + HIGH:
        if cond not in cd:
            print("missing", cond)
            continue
        print(cond, "n_dfs", len(cd[cond]))
        per = []
        for j, df in enumerate(cd[cond]):
            if cond in LOW and j not in DF_SELECTION:
                continue
            if "Timestamp" in df:
                df = df[df["Timestamp"] <= MAX_TS]
            for wid, g in df.groupby("ID", sort=False):
                c = g["CTX"].dropna().to_numpy()
                if len(c) < 50:
                    continue
                per.append({
                    "df_idx": j, "worm_id": str(wid), "mean_CTX": float(np.mean(c)),
                    "sd_CTX": float(np.std(c, ddof=1)) if len(c) > 1 else 0.0,
                    "n": len(c),
                })
        d = pd.DataFrame(per)
        if d.empty:
            continue
        x = d.mean_CTX.to_numpy()
        se = x.std(ddof=1) / np.sqrt(len(x))
        p = wilcoxon(x).pvalue if len(x) > 5 else np.nan
        rows.append({
            "condition": cond, "group": "low" if cond in LOW else "high",
            "n": len(d), "mean_CTX": float(np.mean(x)), "sd_CTX": float(np.std(x, ddof=1)),
            "sem_CTX": float(se), "ci_low": float(np.mean(x) - 1.96 * se),
            "ci_high": float(np.mean(x) + 1.96 * se),
            "positive_fraction": float(np.mean(x > 0)), "wilcoxon_p": float(p),
        })
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    out.to_csv(OUT / "high_salt_ctx_summary.csv", index=False)


if __name__ == "__main__":
    main()
