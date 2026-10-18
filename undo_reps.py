#!/usr/bin/env python3
"""
undo_reps.py — remove recorded repetitions from dataset.csv.

Useful when you mis-sign a gesture and want to drop it before it
contaminates training.

Examples:
    python undo_reps.py --list                 show the last 10 recordings
    python undo_reps.py --last 2               delete the 2 most recent
    python undo_reps.py --drop A:5 B:12        delete specific label:rep pairs
    python undo_reps.py --drop-label T         delete every rep of T

Nothing is written until you confirm, and a timestamped backup of the
original file is saved first.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def load(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{path} not found")
    df = pd.read_csv(p)
    if "label" not in df.columns or "rep" not in df.columns:
        raise SystemExit("that file does not look like a SignSpeak dataset")
    return df


def rep_order(df):
    """
    Repetitions in file order. The collector appends, so the last group
    in the file is the most recently recorded.
    """
    keys = df[["label", "rep"]].drop_duplicates(keep="last")
    return list(keys.itertuples(index=False, name=None))


def show(df, n=10):
    order = rep_order(df)
    counts = df.groupby(["label", "rep"]).size()
    print(f"\n{len(order)} repetitions, {len(df)} rows total\n")
    print(f"  last {min(n, len(order))} recorded (most recent at the bottom):")
    for lab, rep in order[-n:]:
        print(f"    {lab:8s} rep {rep:<4d} {counts[(lab, rep)]:3d} samples")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.csv")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--last", type=int, help="delete the N most recent reps")
    ap.add_argument("--drop", nargs="+", metavar="LABEL:REP",
                    help="delete specific reps, e.g. A:5 B:12")
    ap.add_argument("--drop-label", nargs="+", metavar="LABEL",
                    help="delete every rep of these labels")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    args = ap.parse_args()

    df = load(args.dataset)

    if args.list or not (args.last or args.drop or args.drop_label):
        show(df, 10 if not args.last else args.last)
        if not (args.last or args.drop or args.drop_label):
            return

    targets = []

    if args.last:
        targets += rep_order(df)[-args.last:]

    if args.drop:
        for item in args.drop:
            if ":" not in item:
                raise SystemExit(f"bad format '{item}', expected LABEL:REP")
            lab, rep = item.split(":", 1)
            targets.append((lab.strip().upper(), int(rep)))

    if args.drop_label:
        for lab in args.drop_label:
            lab = lab.strip().upper()
            for l, r in rep_order(df):
                if l == lab:
                    targets.append((l, r))

    # De-duplicate, keep order.
    seen = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]

    counts = df.groupby(["label", "rep"]).size()
    print("\nabout to delete:")
    total = 0
    missing = []
    for lab, rep in targets:
        if (lab, rep) in counts.index:
            n = counts[(lab, rep)]
            total += n
            print(f"    {lab:8s} rep {rep:<4d} {n:3d} samples")
        else:
            missing.append((lab, rep))
    for lab, rep in missing:
        print(f"    {lab:8s} rep {rep:<4d}  NOT FOUND, skipping")

    if total == 0:
        print("\nnothing to delete.")
        return

    print(f"\n  {len(targets) - len(missing)} repetitions, {total} rows")

    if not args.yes:
        try:
            ans = input("\ntype 'yes' to confirm: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\ncancelled.")
            return
        if ans != "yes":
            print("cancelled.")
            return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{Path(args.dataset).stem}_backup_{stamp}.csv"
    shutil.copy(args.dataset, backup)

    mask = pd.Series(False, index=df.index)
    for lab, rep in targets:
        mask |= (df.label == lab) & (df.rep == rep)
    out = df[~mask]

    out.to_csv(args.dataset, index=False)
    print(f"\n  backup saved to {backup}")
    print(f"  {len(df) - len(out)} rows removed")
    print(f"  {args.dataset} now has {len(out)} rows, "
          f"{len(rep_order(out))} repetitions")


if __name__ == "__main__":
    main()