"""
build_features.py — dataset_clean.csv -> rep_features.csv

Collapses each repetition (about 50 rows) into one row holding five
statistics per channel. The channel list comes from features.py, so this
script stays correct when the channel count changes.

    python build_features.py
    python build_features.py --dataset dataset_clean.csv --out rep_features.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import CHANNELS, extract_features, feature_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_clean.csv")
    ap.add_argument("--out", default="rep_features.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{args.dataset} has no column(s): {missing}\n"
            "The dataset was recorded on firmware that did not emit these "
            "channels. Re-record it, or check features.CHANNELS.")

    rows, keys, short = [], [], []
    for (lab, rep), g in df.groupby(["label", "rep"], sort=True):
        w = g.sort_values("sample")[CHANNELS].values
        if len(w) < 40:
            short.append((lab, rep, len(w)))
            continue
        rows.append(extract_features(w))
        keys.append((lab, rep))

    out = pd.DataFrame(np.stack(rows), columns=feature_names(CHANNELS))
    out.insert(0, "rep", [k[1] for k in keys])
    out.insert(0, "label", [k[0] for k in keys])
    out.to_csv(args.out, index=False)

    if short:
        print(f"skipped {len(short)} repetitions with under 40 samples: "
              + ", ".join(f"{l}:{r}({n})" for l, r, n in short[:8])
              + (" ..." if len(short) > 8 else ""))
    print(f"{args.out}: {out.shape[0]} repetitions x {out.shape[1] - 2} "
          f"features, {out.label.nunique()} classes, "
          f"{len(CHANNELS)} channels")

    sp = Path("split_keys.csv")
    if sp.exists():
        s = pd.read_csv(sp)
        have = set(zip(out.label, out.rep))
        miss = [(r.label, r.rep) for r in s.itertuples()
                if (r.label, r.rep) not in have]
        extra = sorted({l for l, r in have} - set(s.label))
        if miss:
            print(f"  note: {len(miss)} rows in split_keys.csv have no "
                  "repetition in the feature file")
        if extra:
            print(f"  note: classes with no split entry (they default to "
                  f"train): {' '.join(extra)}")


if __name__ == "__main__":
    main()
