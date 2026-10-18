"""
analyze_words.py — what does the word dataset actually contain?

Run this BEFORE train_words.py. It answers the questions that decide which
model to train and whether the recordings are usable at all:

  1  Is each class separable, and by how much?
  2  Does time order matter? (the test that says whether a sequence model
     is justified, or whether summary statistics are enough)
  3  Which channels carry the signal - is the left glove pulling its weight,
     or is the right hand doing all the work?
  4  How many repetitions do you actually need?
  5  Are any two words being confused, and which?

Usage:
    python analyze_words.py                       # words.csv
    python analyze_words.py --dataset words.csv --channels merged
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features import (MERGED_CHANNELS, WORD_CHANNELS, WORD_WINDOW,
                      extract_features, fix_length)

warnings.filterwarnings("ignore")
SEED = 42


def model():
    return Pipeline([("s", StandardScaler()),
                     ("c", LDA(solver="lsqr", shrinkage="auto"))])


def cv(X, y, folds=5):
    n = min(folds, int(pd.Series(y).value_counts().min()))
    if n < 2:
        return float("nan"), float("nan")
    s = cross_val_score(model(), X, y,
                        cv=StratifiedKFold(n, shuffle=True, random_state=SEED))
    return s.mean(), s.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="words.csv")
    ap.add_argument("--channels", choices=["word", "merged"], default="word")
    args = ap.parse_args()

    chans = WORD_CHANNELS if args.channels == "word" else MERGED_CHANNELS
    df = pd.read_csv(args.dataset)
    missing = [c for c in chans if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.dataset} is missing: {missing}")
    dead = [c for c in chans if float(df[c].std()) < 1e-9]
    if dead:
        print(f"channels that never move (dropped): {' '.join(dead)}")
        chans = [c for c in chans if c not in dead]

    seqs, y = [], []
    lengths = []
    for (lab, rep), g in df.groupby(["label", "rep"], sort=True):
        g = g.sort_values("sample")
        lengths.append(len(g))
        seqs.append(fix_length(g[chans].values, max(lengths[0], WORD_WINDOW)))
        y.append(str(lab).upper())
    y = np.array(y)
    L = seqs[0].shape[0]
    print(f"\n{len(seqs)} repetitions, {len(set(y))} classes, "
          f"{len(chans)} channels, {L} samples per rep "
          f"({L / 50:.1f} s at 50 Hz)")
    counts = pd.Series(y).value_counts().sort_index()
    print("  reps per class: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    if counts.min() < 15:
        print(f"  WARNING: {counts.idxmin()} has only {counts.min()} reps - "
              "record more before trusting any number below")
    if len(set(lengths)) > 1:
        print(f"  note: repetition lengths vary {min(lengths)}-{max(lengths)}")

    le = LabelEncoder(); Y = le.fit_transform(y)
    centre = (L - WORD_WINDOW) // 2
    win = [s[centre:centre + WORD_WINDOW] for s in seqs]

    X_stats = np.stack([extract_features(w) for w in win])
    X_mean = np.stack([w.mean(axis=0) for w in win])
    rng = np.random.default_rng(SEED)
    X_one = np.stack([w[rng.integers(0, len(w))] for w in win])
    X_flat = np.stack([w.ravel() for w in win])

    print("\n1. DOES TIME ORDER MATTER?  (5-fold CV, LDA throughout)")
    for X, name in [(X_stats, "5 stats per channel - order thrown away"),
                    (X_mean, "mean only - one number per channel"),
                    (X_one, "ONE random timestep - no window at all"),
                    (X_flat, "every timestep, order preserved")]:
        m, s = cv(X, Y)
        print(f"   {name:44s} {X.shape[1]:4d} feat   {m:.4f} ±{s:.4f}")
    print("""
   If 'one random timestep' is close to the others, these words are static
   poses and a sequence model has nothing to learn. If 'order preserved'
   is clearly best, the motion IS the signal and a BiLSTM is justified.""")

    print("\n2. WHICH HAND CARRIES THE SIGNAL?")
    for tag, name in [("r_", "right glove only"), ("l_", "left glove only")]:
        idx = [i for i, c in enumerate(chans) if c.startswith(tag)]
        if not idx:
            continue
        cols = np.concatenate([np.arange(i * 5, i * 5 + 5) for i in idx])
        m, s = cv(X_stats[:, cols], Y)
        print(f"   {name:44s} {len(cols):4d} feat   {m:.4f} ±{s:.4f}")
    m, s = cv(X_stats, Y)
    print(f"   {'both gloves':44s} {X_stats.shape[1]:4d} feat   {m:.4f} ±{s:.4f}")
    print("""
   If the right glove alone matches both, the left glove is not adding
   anything and the words are not really two-handed as recorded.""")

    print("\n3. HOW MANY REPETITIONS DO YOU NEED?")
    for n in [3, 5, 10, 15, 20, 30]:
        if n > counts.min():
            break
        keep = np.zeros(len(Y), bool)
        for c in np.unique(Y):
            idx = np.where(Y == c)[0]
            keep[rng.permutation(idx)[:n]] = True
        m, s = cv(X_stats[keep], Y[keep])
        print(f"   {n:2d} reps/class ({keep.sum():3d} total)              "
              f"{m:.4f} ±{s:.4f}")

    print("\n4. WHAT GETS CONFUSED WITH WHAT?")
    folds = min(5, int(counts.min()))
    if folds >= 2:
        pred = np.empty(len(Y), dtype=int)
        for tr, te in StratifiedKFold(folds, shuffle=True,
                                      random_state=SEED).split(X_stats, Y):
            pred[te] = model().fit(X_stats[tr], Y[tr]).predict(X_stats[te])
        cm = confusion_matrix(Y, pred)
        names = list(le.classes_)
        w = max(len(n) for n in names) + 1
        print("   " + " " * w + "".join(f"{n:>9s}" for n in names))
        for i, n in enumerate(names):
            print(f"   {n:<{w}s}" + "".join(f"{v:9d}" for v in cm[i]))
        print(f"\n   overall {accuracy_score(Y, pred):.4f}")
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i != j and cm[i, j] >= 2:
                    print(f"   {a} taken for {b} {cm[i, j]} times - sign them "
                          "more distinctly, or record more of both")


if __name__ == "__main__":
    main()
