#!/usr/bin/env python3
"""
record_words.py — record two-hand word signs from BOTH gloves into words.csv.

How the two boards stay together: the host sends the same "LABEL,REP,N"
command to the right board and then the left board within a millisecond.
Firmware 2.0 takes its first sample exactly 500 ms after the command, so
sample 0 on one glove is sample 0 on the other; rows are paired by their
sample index. (Measured alignment: run sync_check.py.)

Usage:
    python record_words.py --ports COM7 COM8            # order does not matter
    python record_words.py --ports COM7 COM8 --samples 100

Type a label once, then press Enter for every repetition:

    DOCTOR          switch to DOCTOR and record one repetition
    <Enter>         record the next repetition of the current label
    FEVER,7         force repetition 7 of FEVER
    PROGRESS        show counts
    QUIT            stop

Record NONE as well: left hand raised, both hands doing anything that is
NOT one of the words (idle movement, rest positions, half-formed shapes).
The live pipeline needs it to know when to stay quiet in word mode.

Delete a bad take with:   python undo_reps.py --dataset words.csv --last 1
"""

import argparse
import csv
import sys
import threading
import time
from pathlib import Path

from features import MERGED_CHANNELS, WORD_RECORD_SAMPLES
from signspeak import config
from signspeak.dual import open_pair

CSV_COLUMNS = ["label", "rep", "sample"] + MERGED_CHANNELS
TARGET_REPS = 45


def next_rep(reps, label):
    used = reps.get(label, set())
    return (max(used) + 1) if used else 1


def existing_reps(path):
    reps = {}
    if not Path(path).exists():
        return reps
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            reps.setdefault(row["label"], set()).add(int(row["rep"]))
    return reps


def progress(reps, labels):
    total = sum(len(v) for v in reps.values())
    print("\n" + "=" * 60)
    print(f"words.csv: {total} repetitions")
    print("=" * 60)
    for lab in sorted(set(labels) | set(reps)):
        n = len(reps.get(lab, set()))
        bar = "#" * int(22 * min(1.0, n / TARGET_REPS))
        print(f"{lab:10s} [{bar:22s}] {n:2d}/{TARGET_REPS}")
    print("=" * 60 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs=2, required=True, metavar=("PORT_A", "PORT_B"))
    ap.add_argument("--out", default="words.csv")
    ap.add_argument("--samples", type=int, default=WORD_RECORD_SAMPLES,
                    help=f"rows per repetition (default {WORD_RECORD_SAMPLES} = 1.5 s)")
    args = ap.parse_args()

    labels = list(config.WORD_LABELS)
    out = Path(args.out)
    if not out.exists():
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
        print(f"created {out}")
    reps = existing_reps(out)
    progress(reps, labels)

    right, left = open_pair(*args.ports)

    print("Type a label once, then press Enter for each repetition.")
    print(f"Labels: {' '.join(labels)}   (any other word is accepted with a warning)")
    print("  LABEL        switch to that label and record one")
    print("  <Enter>      record the next repetition")
    print("  PROGRESS     counts so far")
    print("  QUIT         stop")
    print("\nThe board waits 0.5 s after Enter, then captures "
          f"{args.samples / 50:.1f} s. Start the sign as you press Enter.\n")

    label = None
    eof_count = 0

    try:
        while True:
            prompt = (f"{label} rep {next_rep(reps, label)} > "
                      if label else "label > ")
            try:
                raw = input(prompt)
                eof_count = 0
            except KeyboardInterrupt:
                break
            except EOFError:
                # Windows consoles occasionally hand back EOF after a burst
                # of buffered newlines. One is not a request to quit.
                eof_count += 1
                if eof_count >= 3:
                    print("\n  stdin closed - stopping.")
                    break
                time.sleep(0.2)
                continue

            cmd = raw.strip().upper()
            if cmd in ("QUIT", "EXIT"):
                break
            if cmd in ("PROGRESS", "STATUS"):
                progress(reps, labels)
                continue

            rep = None
            if cmd:
                if "," in cmd:
                    new_label, _, rep_s = cmd.partition(",")
                    new_label = new_label.strip()
                    try:
                        rep = int(rep_s)
                    except ValueError:
                        print("  bad rep number")
                        continue
                else:
                    new_label = cmd
                if not new_label.isalpha():
                    print("  labels are letters only (no spaces): e.g. MYNAME")
                    continue
                if new_label not in labels:
                    print(f"  warning: {new_label} is not in "
                          "config.WORD_LABELS; recording anyway")
                label = new_label
            elif label is None:
                print("  type a label first, e.g. DOCTOR")
                continue

            if rep is None:
                rep = next_rep(reps, label)
            command = f"{label},{rep},{args.samples}"

            # Send to both boards back-to-back, then read both in parallel.
            results = {}

            def grab(link, key):
                results[key] = link.collect(label, rep, args.samples)

            right.send(command)
            left.send(command)
            tr = threading.Thread(target=grab, args=(right, "R"))
            tl = threading.Thread(target=grab, args=(left, "L"))
            tr.start(); tl.start(); tr.join(); tl.join()

            r_rows, r_notes = results["R"]
            l_rows, l_notes = results["L"]
            for n in r_notes:
                if "TIMEOUT" in n or "WARN" in n:
                    print(f"  [RIGHT] {n}")
            for n in l_notes:
                if "TIMEOUT" in n or "WARN" in n:
                    print(f"  [LEFT]  {n}")

            common = sorted(set(r_rows) & set(l_rows))
            if len(common) < args.samples * 0.9:
                print(f"  only {len(common)} paired samples of {args.samples} "
                      f"(right {len(r_rows)}, left {len(l_rows)}) - NOT saved. "
                      "Try again.")
                continue

            with open(out, "a", newline="") as f:
                w = csv.writer(f)
                for i in common:
                    w.writerow([label, rep, i]
                               + [f"{v:.2f}" for v in r_rows[i]]
                               + [f"{v:.2f}" for v in l_rows[i]])
            reps.setdefault(label, set()).add(rep)
            print(f"  saved {label} rep {rep}: {len(common)} paired samples"
                  f"   [{len(reps[label])} total for {label}]")
    finally:
        right.close()
        left.close()
        progress(reps, labels)
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
