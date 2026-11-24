# -*- coding: utf-8 -*-
"""Cache real x/y/speed/agl/CTX for transition-probability regression."""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
CACHE = ROOT / "Fitting" / "regression_cache"
CACHE.mkdir(exist_ok=True)
CONDS = ["3g-1gNa-1h", "3g-1gNa-4h", "1gNa-end(8x)"]
DF_SELECTION = list(range(10, 20))
MAX_TS = 14400


def main():
    with open(ROOT / "data" / "condition_dict.pkl", "rb") as f:
        cd = pickle.load(f)
    for cond in CONDS:
        seqs = []
        for j, df in enumerate(cd[cond]):
            if j not in DF_SELECTION:
                continue
            if "Timestamp" in df:
                df = df[df["Timestamp"] <= MAX_TS]
            xcol = "X" if "X" in df.columns else "x"
            ycol = "Y" if "Y" in df.columns else "y"
            for wid, g in df.groupby("ID", sort=False):
                g = g[[xcol, ycol, "speed", "agl_velocity", "CTX"]].dropna()
                if len(g) < 50:
                    continue
                seqs.append({
                    "df_idx": j,
                    "worm_id": str(wid),
                    "x": g[xcol].to_numpy(dtype=np.float32),
                    "y": g[ycol].to_numpy(dtype=np.float32),
                    "speed": g["speed"].to_numpy(dtype=np.float32),
                    "agl": g["agl_velocity"].to_numpy(dtype=np.float32),
                    "ctx": g["CTX"].to_numpy(dtype=np.float32),
                })
        out = CACHE / (cond.replace("(", "").replace(")", "") + ".pkl")
        with open(out, "wb") as f:
            pickle.dump(seqs, f)
        print(cond, "worms", len(seqs), "->", out)


if __name__ == "__main__":
    main()
