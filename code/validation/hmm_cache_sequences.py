# -*- coding: utf-8 -*-
"""Cache real tracking sequences used by the main fit for fast HMM iteration."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\ZHY\code\AssociativeModel")
CACHE = ROOT / "Fitting" / "hmm_seq_cache"
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
            for wid, g in df.groupby("ID", sort=False):
                g = g.dropna(subset=["speed", "agl_velocity"])
                if len(g) < 50:
                    continue
                seqs.append({
                    "df_idx": j,
                    "worm_id": str(wid),
                    "speed": g["speed"].to_numpy(dtype=np.float32),
                    "agl": g["agl_velocity"].to_numpy(dtype=np.float32),
                })
        out = CACHE / (cond.replace("(", "").replace(")", "") + ".pkl")
        with open(out, "wb") as f:
            pickle.dump(seqs, f)
        print(cond, "worms", len(seqs), "->", out)


if __name__ == "__main__":
    main()
