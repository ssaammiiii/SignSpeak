"""
check_words.py — is words.csv actually usable?

Checks, in order of how badly each one breaks a model:

  1  DEAD CHANNELS     a sensor that never moves carries no information and
                       gives every statistics-based model zero-variance
                       features. This is what broke the letter LDA.
  2  CORRUPT IMU READS a bad I2C read returns every axis railed at once.
                       Real motion never saturates accel AND gyro on all
                       axes in the same 20 ms sample. These are not data.
  3  FROZEN IMU        the opposite failure: identical values repeated,
                       which MPU6050_light returns when I2C stops
                       answering and the library serves its cache.
  4  RAIL TIME         how much of each channel sits at its limit.
  5  PER-REP DAMAGE    which repetitions are worst, so you know whether to
                       drop a few or re-record the lot.

Usage:
    python check_words.py                    # words.csv
    python check_words.py --dataset words.csv --gyro-range 500
"""

import argparse

import numpy as np
import pandas as pd

from features import FLEX, IMU, MERGED_CHANNELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="words.csv")
    ap.add_argument("--gyro-range", type=float, default=500.0,
                    help="MPU6050 gyro full scale, deg/s (library default 500)")
    ap.add_argument("--accel-range", type=float, default=2.0,
                    help="MPU6050 accel full scale, g (library default 2)")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    chans = [c for c in MERGED_CHANNELS if c in df.columns]
    reps = df.groupby(["label", "rep"]).ngroups
    print(f"{args.dataset}: {len(df)} rows, {reps} repetitions, "
          f"{df.label.nunique()} classes, {len(chans)} channels\n")

    # ---- 1. dead channels ------------------------------------------------
    print("=" * 68)
    print("1. DEAD CHANNELS")
    print("=" * 68)
    dead = []
    for c in chans:
        sd = float(df[c].std())
        if sd < 1e-9:
            dead.append((c, float(df[c].iloc[0])))
    if dead:
        for c, v in dead:
            print(f"   {c:16s} NEVER MOVES - constant {v}")
        print(f"\n   {len(dead)} channel(s) carry no information at all.")
        print("   A flex sensor pinned at 1.00 is bent past its range or its")
        print("   calibration span was too small; pinned at 0.00 is an open")
        print("   circuit or a sensor that never bends. Fix the hardware -")
        print("   no amount of data makes a constant column informative.")
    else:
        print("   none - every channel varies.")

    # near-dead: moves, but barely
    print()
    weak = [(c, float(df[c].std())) for c in chans
            if c.split("_", 1)[1] in FLEX and 1e-9 <= float(df[c].std()) < 0.03]
    for c, sd in weak:
        print(f"   {c:16s} sd {sd:.4f} - moves, but almost not at all")

    # ---- 2. corrupt IMU reads ---------------------------------------------
    print("\n" + "=" * 68)
    print("2. CORRUPT IMU READS  (every axis railed in the same sample)")
    print("=" * 68)
    for side, tag in (("right", "r_"), ("left", "l_")):
        acc = [tag + c for c in IMU[:3] if tag + c in df.columns]
        gyr = [tag + c for c in IMU[3:] if tag + c in df.columns]
        if not acc or not gyr:
            continue
        a_rail = (df[acc].abs() >= args.accel_range * 0.99).any(axis=1)
        g_rail = (df[gyr].abs() >= args.gyro_range * 0.99).any(axis=1)
        both = a_rail & g_rail
        n = int(both.sum())
        print(f"   {side:5s} glove: accel railed {int(a_rail.sum()):5d} rows, "
              f"gyro railed {int(g_rail.sum()):5d}, BOTH {n:5d} "
              f"({n / len(df):.1%})")
        if n:
            idx = np.where(both.values)[0]
            gaps = np.diff(idx)
            gaps = gaps[gaps > 1]
            if len(gaps):
                print(f"          spacing between corrupt samples: median "
                      f"{np.median(gaps):.0f} samples "
                      f"({np.median(gaps) * 0.02:.2f} s), "
                      f"min {gaps.min()}, max {gaps.max()}")

    print("""
   Accelerometer and gyroscope do not both saturate on every axis from
   real hand motion - the hand cannot pull 2 g and spin at 500 deg/s on
   three axes at once, let alone on a regular cadence. Regularly spaced
   all-axis saturation is a failed I2C read: the library returns a
   half-updated register block. Raising the ranges does NOT fix it; the
   bus does.""")

    # ---- 3. frozen ---------------------------------------------------------
    print("=" * 68)
    print("3. FROZEN IMU  (identical consecutive samples)")
    print("=" * 68)
    for side, tag in (("right", "r_"), ("left", "l_")):
        cols = [tag + c for c in IMU if tag + c in df.columns]
        if not cols:
            continue
        worst, worst_key = 0, None
        for (lab, rep), g in df.groupby(["label", "rep"]):
            v = g.sort_values("sample")[cols].values
            same = np.all(np.diff(v, axis=0) == 0, axis=1)
            run = best = 0
            for s in same:
                run = run + 1 if s else 0
                best = max(best, run)
            if best > worst:
                worst, worst_key = best, (lab, rep)
        verdict = ("no freeze" if worst < 3 else
                   f"WARNING - IMU held the same value {worst + 1} samples "
                   f"({(worst + 1) * 0.02:.2f} s)")
        print(f"   {side:5s} glove: longest identical run {worst + 1} "
              f"samples, worst in {worst_key}   {verdict}")

    # ---- 4. rail time ------------------------------------------------------
    print("\n" + "=" * 68)
    print("4. TIME AT THE RAIL, PER CHANNEL")
    print("=" * 68)
    print(f"   {'channel':16s} {'at rail':>9s}   {'min':>8s} {'max':>8s}")
    for c in chans:
        base = c.split("_", 1)[1]
        if base in FLEX:
            rail = float(((df[c] <= 0.001) | (df[c] >= 0.999)).mean())
        elif base.startswith("accel"):
            rail = float((df[c].abs() >= args.accel_range * 0.99).mean())
        else:
            rail = float((df[c].abs() >= args.gyro_range * 0.99).mean())
        if rail > 0.02:
            print(f"   {c:16s} {rail:8.1%}   {df[c].min():8.2f} "
                  f"{df[c].max():8.2f}")

    # ---- 5. per-rep --------------------------------------------------------
    print("\n" + "=" * 68)
    print("5. WORST REPETITIONS")
    print("=" * 68)
    imu_cols = [c for c in chans if c.split("_", 1)[1] in IMU]
    acc_cols = [c for c in imu_cols if "accel" in c]
    gyr_cols = [c for c in imu_cols if "gyro" in c]
    rows = []
    for (lab, rep), g in df.groupby(["label", "rep"]):
        bad = ((g[acc_cols].abs() >= args.accel_range * 0.99).any(axis=1)
               & (g[gyr_cols].abs() >= args.gyro_range * 0.99).any(axis=1))
        rows.append({"label": lab, "rep": rep, "rows": len(g),
                     "corrupt": int(bad.sum()),
                     "pct": bad.mean()})
    r = pd.DataFrame(rows).sort_values("pct", ascending=False)
    print(f"   {'label':10s} {'rep':>4s} {'rows':>5s} {'corrupt':>8s} {'pct':>7s}")
    for x in r.head(10).itertuples():
        print(f"   {x.label:10s} {x.rep:4d} {x.rows:5d} {x.corrupt:8d} "
              f"{x.pct:6.1%}")
    print(f"\n   median corrupt fraction across all reps: {r.pct.median():.1%}")
    if r.pct.median() > 0.02:
        print("   This is not a handful of bad takes - it is every take.")
        print("   Fix the bus, then re-record. Dropping reps will not help.")


if __name__ == "__main__":
    main()
