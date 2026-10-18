"""
diagnose_live.py — find out WHY live predictions disagree with the model.

The model learned a specific numeric range for each sensor channel. If the
glove now produces different ranges, every prediction is wrong no matter
how good the model is. This script measures the gap directly.

It asks you to hold three positions, then compares what the glove sends
against what the training data contains.

Usage:
    python diagnose_live.py --port COM7 --dataset dataset.csv
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

from features import CHANNELS, WINDOW
from live_test import SerialSource, fill_window

FLEX = CHANNELS[:5]


def capture(source, prompt, n=WINDOW):
    print(f"\n  {prompt}")
    try:
        input("  press Enter here when you are holding it: ")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\ncancelled.")
    print("  capturing...", end="", flush=True)
    w = fill_window(source, n, timeout_s=15.0)
    if w is None:
        raise SystemExit(" no data. The stream stopped.")
    print(" done")
    return w


def band(v):
    if v < 0.02:
        return "PINNED LOW "
    if v > 0.98:
        return "PINNED HIGH"
    return "           "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dataset", default="dataset.csv")
    args = ap.parse_args()

    print("loading training data...")
    df = pd.read_csv(args.dataset)

    source = SerialSource(args.port, args.baud)

    flat = capture(source, "Hold your hand FLAT, fingers straight.")
    fist = capture(source, "Make a tight FIST.")
    neutral = capture(source, "Rest your hand in the NEUTRAL position.")

    print("\n" + "=" * 66)
    print("TEST 1 — does the glove span its full range?")
    print("=" * 66)
    print("  Flat should read near 0.00, fist near 1.00 on every channel.")
    print()
    print(f"  {'channel':10s} {'flat':>7s} {'fist':>7s} {'span':>7s}   verdict")
    bad_span = []
    for i, c in enumerate(FLEX):
        f0 = float(flat[:, i].mean())
        f1 = float(fist[:, i].mean())
        span = abs(f1 - f0)
        if span < 0.5:
            verdict = "DEAD — barely responds"
            bad_span.append(c)
        elif span < 0.8:
            verdict = "weak"
            bad_span.append(c)
        else:
            verdict = "ok"
        print(f"  {c:10s} {f0:7.2f} {f1:7.2f} {span:7.2f}   {verdict}")

    print("\n" + "=" * 66)
    print("TEST 2 — is the resting hand where the model expects?")
    print("=" * 66)
    ref = df[df.label == "NEUTRAL"]
    if len(ref) == 0:
        print("  no NEUTRAL rows in the dataset, skipping")
    else:
        print(f"  {'channel':10s} {'live':>7s} {'trained':>8s} {'drift':>7s}   flag")
        drifted = []
        for i, c in enumerate(CHANNELS):
            live = float(neutral[:, i].mean())
            trained = float(ref[c].mean())
            sd = float(ref[c].std())
            d = live - trained
            z = abs(d) / sd if sd > 1e-6 else 0.0
            flag = "OFF" if z > 3 else ("   " if z < 2 else " ? ")
            if z > 3:
                drifted.append(c)
            print(f"  {c:10s} {live:7.2f} {trained:8.2f} {d:+7.2f}   {flag}")

    print("\n" + "=" * 66)
    print("TEST 3 — how much of the live signal is stuck at a rail?")
    print("=" * 66)
    both = np.vstack([flat, fist, neutral])
    print(f"  {'channel':10s} {'at 0.00':>9s} {'at 1.00':>9s}   {'trained clip':>13s}")
    for i, c in enumerate(FLEX):
        lo = float((both[:, i] <= 0.001).mean())
        hi = float((both[:, i] >= 0.999).mean())
        tclip = float(((df[c] <= 0.001) | (df[c] >= 0.999)).mean())
        print(f"  {c:10s} {lo:8.0%} {hi:9.0%}   {tclip:12.0%}")

    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    if bad_span:
        print(f"  Sensors with poor range: {', '.join(bad_span)}")
        print("  These channels cannot distinguish handshapes. Fix the")
        print("  hardware or calibration before touching the dataset.")
    if 'drifted' in dir() and drifted:
        print(f"  Channels far from training values: {', '.join(drifted)}")
        print("  The glove is not producing the numbers the model learned.")
    if not bad_span and not ('drifted' in dir() and drifted):
        print("  Sensors look consistent with the training data.")
        print("  If predictions are still wrong, the problem is likely how")
        print("  the signs are being formed, not the hardware.")

    source.close()


if __name__ == "__main__":
    main()
