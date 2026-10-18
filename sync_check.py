"""
sync_check.py — how well do the two glove streams line up?

Two independent measurements:

  FRAME GAP   For every paired frame, how far apart did the two frames
              arrive. Hundreds of samples, no interpretation needed. This
              is the number to put in the thesis.

  CLAP TEST   Clap the gloves together; each clap is one accelerometer
              spike on both gloves at the same instant, so the difference
              between them is an independent check. Only about five
              samples, and it depends on spike detection working cleanly,
              so it CONFIRMS the frame gap rather than overriding it. If
              the two disagree, believe the frame gap.

Usage:
    python sync_check.py --ports COM6 COM7
    python sync_check.py --ports COM6 COM7 --seconds 15
"""

import argparse
import time

import numpy as np

from signspeak.dual import DualStream, open_pair


def spikes(hist, gap_s=0.4):
    """
    Arrival times of accelerometer spikes, with a threshold set from the
    data rather than a fixed constant, so it adapts to how hard you clap.
    """
    if len(hist) < 50:
        return [], 0.0
    t = np.array([x[0] for x in hist])
    mag = np.array([float(np.linalg.norm(x[1][5:8])) for x in hist])
    base = float(np.median(mag))
    spread = float(np.percentile(mag, 95) - base)
    thresh = base + max(6.0 * spread, 0.8)

    out, last = [], -1e9
    i = 0
    while i < len(mag):
        if mag[i] > thresh and t[i] - last > gap_s:
            # take the peak of this burst, not its leading edge
            j = i
            while j + 1 < len(mag) and t[j + 1] - t[i] < 0.12:
                j += 1
            k = i + int(np.argmax(mag[i:j + 1]))
            out.append(float(t[k]))
            last = t[k]
            i = j + 1
        else:
            i += 1
    return out, thresh


def match(rs, ls, window=0.15):
    """Pair each right spike with the nearest left spike, one to one."""
    used, offs = set(), []
    for t in rs:
        best, bj = None, None
        for j, u in enumerate(ls):
            if j in used:
                continue
            d = u - t
            if abs(d) < window and (best is None or abs(d) < abs(best)):
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            offs.append(best * 1000.0)
    return offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs=2, required=True)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--no-clap", action="store_true",
                    help="skip the clap test, just measure the frame gap")
    args = ap.parse_args()

    right, left = open_pair(*args.ports)
    dual = DualStream(right, left)
    for rd in (dual.r_reader, dual.l_reader):
        rd.hist = rd.hist.__class__(maxlen=6000)
    dual.start()

    if args.no_clap:
        print(f"\nHold still for {args.seconds:.0f} s...")
    else:
        print(f"\nClap the gloves together 5 times, firmly, about a second "
              f"apart, within {args.seconds:.0f} s...")

    t0 = time.time()
    gaps = []
    for t, merged, gap in dual:
        gaps.append(gap * 1000.0)
        if time.time() - t0 > args.seconds:
            break
    r_hist = list(dual.r_reader.hist)
    l_hist = list(dual.l_reader.hist)
    stale = dual.stale
    dual.close()

    gaps = np.array(gaps)
    print("\n" + "=" * 62)
    print("FRAME GAP  (the measurement that matters)")
    print("=" * 62)
    print(f"  right {len(r_hist)} frames at {len(r_hist)/args.seconds:.1f} Hz,"
          f"  left {len(l_hist)} at {len(l_hist)/args.seconds:.1f} Hz")
    print(f"  {len(gaps)} paired, {stale} dropped "
          f"({stale / max(len(gaps) + stale, 1):.1%})")
    print(f"  gap between paired frames: median {np.median(gaps):.1f} ms, "
          f"95th {np.percentile(gaps, 95):.1f} ms, max {gaps.max():.1f} ms")
    ok = np.percentile(gaps, 95) <= 20.0
    print(f"\n  -> {'aligned to within one 20 ms frame' if ok else 'worse than one frame - check both boards run firmware 2.0'}")
    if np.median(gaps) > 12.0:
        print("  -> the median is over half a frame: the two ticks are out of")
        print("     phase. Harmless, but restarting both streams (rerun this)")
        print("     usually re-aligns them.")

    if args.no_clap:
        return

    rs, thr = spikes(r_hist)
    ls, _ = spikes(l_hist)
    print("\n" + "=" * 62)
    print("CLAP TEST  (independent check, small sample)")
    print("=" * 62)
    print(f"  spikes detected: right {len(rs)}, left {len(ls)}")
    if not rs or not ls:
        print("  no claps detected on both gloves - clap harder, or rerun")
        print("  with --no-clap; the frame gap above is the real number.")
        return
    if abs(len(rs) - len(ls)) > 1:
        print("  the two gloves saw different numbers of claps, so some")
        print("  pairings below are between different events and the spread")
        print("  is meaningless. Clap more firmly and keep the gloves still")
        print("  between claps. Trust the frame gap above.")
    offs = match(rs, ls)
    if not offs:
        print("  no clap matched up within 150 ms.")
        return
    offs = np.array(offs)
    print(f"  offset left-right: mean {offs.mean():+.1f} ms, "
          f"sd {offs.std():.1f} ms, worst {offs[np.argmax(abs(offs))]:+.1f} ms "
          f"(n={len(offs)})")
    if offs.std() > 25:
        print("  the spread is larger than the frame gap measurement allows,")
        print("  which means spike detection mismatched some claps - not a")
        print("  clock problem. Report the frame gap.")
    elif abs(offs.mean()) <= 20:
        print("  agrees with the frame gap: the streams are aligned.")


if __name__ == "__main__":
    main()
