"""
live_test.py — see the model predict in real time, with confidence scores.

This is a debug tool, not the final demo pipeline. No TTS, no sentence
building, no Gemini. Just: sign a letter, see what the model thinks.

Three modes:

  guided   Prompts you letter by letter, you press Enter, it captures one
           window and shows the prediction. Tallies a score at the end.
           This is the one you want for "sign every letter and see".

  free     Continuous rolling prediction. Sign whatever, whenever. Good for
           checking stability and watching confidence move as you form a
           handshape.

  replay   No hardware. Feeds recorded rows from dataset.csv through the
           exact same path. Use this to prove the script works before you
           blame the glove.

Examples:
    python live_test.py --mode replay --dataset dataset.csv
    python live_test.py --mode guided --port COM5
    python live_test.py --mode guided --port COM5 --letters ABCDE
    python live_test.py --mode free --port COM5
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from features import CHANNELS, WINDOW
from predict_live import GesturePredictor

N_CH = len(CHANNELS)          # 14 with firmware 2.0
FLEX_IDX = list(range(5))     # first five channels are the flex sensors


# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------

class SerialSource:
    """
    Reads calibrated sensor rows from the ESP32 over USB serial.

    Opening the port resets the ESP32, which drops it back into its
    calibration routine. This class walks through that handshake instead
    of hanging, so calibration happens in the same session as the test.
    """

    def __init__(self, port, baud=115200, timeout=1.0, start_cmd="STREAM",
                 handshake=True):
        try:
            import serial
        except ImportError:
            raise SystemExit("pyserial not installed. Run: uv add pyserial")
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)          # ESP32 reboots when the port opens
        self.ser.reset_input_buffer()
        self._warned = False
        self.start_cmd = start_cmd

        if handshake:
            self._handshake(start_cmd)
        elif start_cmd:
            self._send(start_cmd)

    # -- low level ----------------------------------------------------

    def _send(self, text):
        self.ser.write((text + "\n").encode())
        self.ser.flush()

    def _readline(self):
        raw = self.ser.readline()
        if not raw:
            return None
        try:
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return None

    def _parse(self, line):
        """Returns an (11,) array if the line is sensor data, else None."""
        if not line:
            return None
        vals = []
        for p in line.split(","):
            try:
                vals.append(float(p.strip()))
            except ValueError:
                pass
        if len(vals) < N_CH:
            return None
        if len(vals) > N_CH:
            # STREAM frames carry no prefix, so the sensor data is the
            # leading N_CH values.
            if not self._warned:
                print(f"\n[note] {len(vals)} numbers per line, using the "
                      f"first {N_CH}")
                self._warned = True
            vals = vals[:N_CH]
        return np.array(vals, dtype=np.float64)

    # -- startup ------------------------------------------------------

    def _handshake(self, start_cmd, wait_s=90.0):
        """
        Walk the board from boot to streaming.

        The firmware's stream loop runs `while (Serial.available() == 0)`,
        so ANY extra byte sent after streaming starts will stop it. That
        makes it critical to send start_cmd sparingly and to stop sending
        entirely once the board confirms it is streaming.
        """
        print("connecting to the board...\n")
        deadline = time.time() + wait_s
        last_send = 0.0
        waiting_for_enter = False
        armed = True          # cleared once the board says STREAMING

        def nudge():
            nonlocal last_send
            if armed and not waiting_for_enter and time.time() - last_send > 3.0:
                self._send(start_cmd)
                last_send = time.time()

        while time.time() < deadline:
            line = self._readline()

            if line:
                if self._parse(line) is not None:
                    print("\n  stream is live\n")
                    return
                print(f"  esp32 | {line}")

                if "PRESS_ENTER" in line:
                    waiting_for_enter = True
                    try:
                        input("\n  >>> do that, then press Enter HERE: ")
                    except (EOFError, KeyboardInterrupt):
                        raise SystemExit("\ncancelled.")
                    self._send("")
                    last_send = time.time()
                    waiting_for_enter = False
                    deadline = time.time() + wait_s

                elif "STREAMING" in line:
                    # Board is streaming. Sending anything now would
                    # stop it, so stand down.
                    armed = False

                elif "Calibration complete" in line or "READY" in line:
                    waiting_for_enter = False
                    nudge()
                continue

            nudge()

        print(f"\n*** No sensor data after {wait_s:.0f}s.")
        print("*** Check: is the Arduino Serial Monitor closed?")
        print("*** Check: is the sketch with the STREAM command uploaded?")
        print("*** Try:   python live_test.py --mode raw --port <PORT>")
        raise SystemExit(1)

    # -- streaming ----------------------------------------------------

    def read(self):
        """Returns one (11,) array, or None if the line was not usable."""
        return self._parse(self._readline())

    def close(self):
        try:
            self._send("x")   # any byte stops the firmware stream
            time.sleep(0.2)
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass


class ReplaySource:
    """Replays recorded repetitions so the script can be tested offline."""

    def __init__(self, dataset_path, rate_hz=200.0):
        import pandas as pd
        df = pd.read_csv(dataset_path)
        self.groups = []
        for (lab, rep), g in df.groupby(["label", "rep"], sort=True):
            self.groups.append((lab, g.sort_values("sample")[CHANNELS].values))
        self.rate = rate_hz
        self.gi = 0
        self.ri = 0

    def current_label(self):
        return self.groups[self.gi % len(self.groups)][0]

    def seek_label(self, label):
        """Jump to the first recorded repetition of a given letter."""
        for i, (lab, _) in enumerate(self.groups):
            if lab == label:
                self.gi, self.ri = i, 0
                return True
        return False

    def read(self):
        lab, arr = self.groups[self.gi % len(self.groups)]
        row = arr[self.ri]
        self.ri += 1
        if self.ri >= len(arr):
            self.ri = 0
            self.gi += 1
        if self.rate:
            time.sleep(1.0 / self.rate)
        return row.astype(np.float64)

    def close(self):
        pass


# ----------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------

def bar(value, width=20):
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "█" * filled + "░" * (width - filled)


def sanity_check(window, threshold_msg=True):
    """
    Flex channels must arrive already calibrated to 0.0-1.0, the same as the
    training data. Raw millivolts here means the model sees nonsense.
    """
    flex = window[:, FLEX_IDX]
    lo, hi = float(flex.min()), float(flex.max())
    if hi > 1.5 or lo < -0.5:
        print(f"\n*** WARNING: flex values range {lo:.1f} to {hi:.1f} ***")
        print("*** Expected 0.0 to 1.0. The firmware is sending uncalibrated")
        print("*** readings, or the channel order does not match CHANNELS.")
        print("*** Predictions below are meaningless until this is fixed.\n")
        return False
    return True


def top_k(predictor, window, k=3):
    """Returns [(label, prob), ...] sorted high to low."""
    from features import extract_features, fix_length
    w = fix_length(window, WINDOW)[:, :len(predictor.channels)]
    if predictor.kind == "deep":
        x = ((w - predictor.mu) / predictor.sd)[None, ...].astype(np.float32)
        probs = predictor.model.predict(x, verbose=0)[0]
    else:
        x = extract_features(w)[None, :]
        probs = predictor.model.predict_proba(x)[0]
    order = np.argsort(probs)[::-1][:k]
    labels = predictor.le.inverse_transform(order)
    return list(zip(labels, probs[order]))


def fill_window(source, n=WINDOW, timeout_s=10.0):
    """Collects n valid rows. Returns None on timeout."""
    buf = []
    t0 = time.time()
    while len(buf) < n:
        if time.time() - t0 > timeout_s:
            return None
        row = source.read()
        if row is not None:
            buf.append(row)
    return np.stack(buf)


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------

def run_guided(source, predictor, letters, threshold, logfile):
    print("\n" + "=" * 60)
    print("GUIDED TEST — press Enter, hold the sign for one second")
    print("Enter with nothing = capture   |   s = skip   |   q = quit")
    print("=" * 60 + "\n")

    results = []
    checked = False

    for i, letter in enumerate(letters, 1):
        while True:
            try:
                cmd = input(f"[{i}/{len(letters)}]  Sign  {letter}  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return results
            if cmd == "q":
                return results
            if cmd == "s":
                print("   skipped\n")
                break

            if isinstance(source, ReplaySource):
                source.seek_label(letter)

            print("   capturing...", end="", flush=True)
            window = fill_window(source, WINDOW)
            if window is None:
                print(" no data. Check the port and that firmware is streaming.")
                continue

            if not checked:
                sanity_check(window)
                checked = True

            ranked = top_k(predictor, window, k=3)
            pred, conf = ranked[0]
            ok = pred == letter
            accepted = conf >= threshold

            mark = "OK " if ok else "MISS"
            gate = "accepted" if accepted else f"below {threshold:.2f}"
            print(f"\r   {mark}  predicted {pred}   {conf:.3f}  "
                  f"{bar(conf)}  {gate}")
            if not ok or not accepted:
                alts = "   ".join(f"{l}:{p:.3f}" for l, p in ranked[1:])
                print(f"        next best   {alts}")
            print()

            results.append({"expected": letter, "predicted": pred,
                            "confidence": round(float(conf), 4),
                            "correct": ok, "accepted": accepted})
            break

    return results


def run_raw(port, baud, seconds=20):
    """Dump whatever the board sends. Use this when nothing else works."""
    import serial
    print(f"opening {port} at {baud}, listening for {seconds}s")
    print("(the board resets on connect, so calibration prompts are normal)\n")
    ser = serial.Serial(port, baud, timeout=1.0)
    time.sleep(2.0)
    t0 = time.time()
    n = 0
    try:
        while time.time() - t0 < seconds:
            raw = ser.readline()
            if not raw:
                continue
            n += 1
            print(f"  {raw.decode('utf-8', errors='ignore').rstrip()}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
    if n == 0:
        print("\n*** Nothing received at all.")
        print("*** Wrong COM port, Serial Monitor still open, or baud mismatch.")
    else:
        print(f"\n{n} lines received.")


def run_free(source, predictor, threshold, stride=10):
    print("\n" + "=" * 60)
    print("FREE RUN — rolling prediction. Ctrl+C to stop.")
    print("=" * 60 + "\n")

    buf = deque(maxlen=WINDOW)
    checked = False
    last_stable = None
    stable_count = 0
    since = 0
    last_row_at = time.time()
    stalled = False

    try:
        while True:
            row = source.read()
            if row is None:
                if not stalled and time.time() - last_row_at > 4.0:
                    print("\n\n*** Stream stopped sending data.")
                    print("*** The firmware halts its stream when any byte")
                    print("*** arrives on serial. Something wrote to the port.")
                    print("*** Restart the script; if it repeats, re-upload.\n")
                    stalled = True
                continue
            last_row_at = time.time()
            stalled = False
            buf.append(row)
            if len(buf) < WINDOW:
                print(f"\r  filling window {len(buf)}/{WINDOW}", end="", flush=True)
                continue

            since += 1
            if since < stride:
                continue
            since = 0

            window = np.stack(buf)
            if not checked:
                print()
                sanity_check(window)
                checked = True

            ranked = top_k(predictor, window, k=2)
            pred, conf = ranked[0]
            second = f"{ranked[1][0]}:{ranked[1][1]:.2f}" if len(ranked) > 1 else ""
            gate = "ACCEPT" if conf >= threshold else "  --  "

            print(f"\r  {pred:>7s}   {conf:.3f}  {bar(conf)}  {gate}   "
                  f"2nd {second}      ", end="", flush=True)

            # Log a confirmed letter once it has been held steadily.
            if conf >= threshold and pred == last_stable:
                stable_count += 1
                if stable_count == 3:
                    print(f"\r  >>> {pred}   held at {conf:.3f}"
                          f"{' ' * 40}")
            elif conf >= threshold:
                last_stable, stable_count = pred, 1
            else:
                last_stable, stable_count = None, 0

    except KeyboardInterrupt:
        print("\n\nstopped.")


# ----------------------------------------------------------------------

def summarise(results, threshold, logfile):
    if not results:
        print("nothing captured.")
        return
    import pandas as pd
    df = pd.DataFrame(results)
    n = len(df)
    correct = int(df.correct.sum())
    accepted = int(df.accepted.sum())

    print("\n" + "=" * 60)
    print(f"  correct        {correct}/{n}   ({correct / n:.1%})")
    print(f"  above {threshold:.2f}     {accepted}/{n}   ({accepted / n:.1%})")
    print(f"  mean confidence   {df.confidence.mean():.3f}")
    print("=" * 60)

    misses = df[~df.correct]
    if len(misses):
        print("\n  misses:")
        for r in misses.itertuples():
            print(f"    {r.expected} -> {r.predicted}   {r.confidence:.3f}")

    weak = df[df.correct & ~df.accepted]
    if len(weak):
        print("\n  correct but below threshold:")
        for r in weak.itertuples():
            print(f"    {r.expected}   {r.confidence:.3f}")

    df.to_csv(logfile, index=False)
    print(f"\n  saved to {logfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["guided", "free", "replay", "raw"],
                    default="guided")
    ap.add_argument("--port", help="e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dataset", default="dataset.csv",
                    help="used by replay mode")
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--letters", default=None,
                    help="which letters to test, e.g. ABCDE. "
                         "Default: every class the model knows.")
    ap.add_argument("--log", default="live_test_log.csv")
    ap.add_argument("--start-cmd", default="STREAM",
                    help="command sent to the ESP32 to begin streaming. "
                         "Use '' if your firmware streams on boot.")
    args = ap.parse_args()

    if args.mode == "raw":
        if not args.port:
            raise SystemExit("--port is required for raw mode")
        run_raw(args.port, args.baud)
        return

    predictor = GesturePredictor(args.results, model=args.model,
                                 threshold=args.threshold)
    known = list(predictor.le.classes_)
    print(f"model: {args.model}   knows {len(known)} classes")
    print(f"       {' '.join(known)}")

    if args.letters:
        letters = [c for c in args.letters.upper()]
        unknown = [c for c in letters if c not in known]
        if unknown:
            print(f"\n*** these are not in the model: {' '.join(unknown)}")
            print("*** they will always be wrong. Collect and retrain.\n")
    else:
        letters = known

    if args.mode == "replay":
        source = ReplaySource(args.dataset)
        print("source: replay from dataset.csv (no hardware)\n")
        results = run_guided(source, predictor, letters, args.threshold, args.log)
        summarise(results, args.threshold, args.log)
    else:
        if not args.port:
            raise SystemExit("--port is required. Example: --port COM5")
        source = SerialSource(args.port, args.baud,
                              start_cmd=args.start_cmd)
        print(f"source: {args.port} at {args.baud} baud\n")
        try:
            if args.mode == "guided":
                results = run_guided(source, predictor, letters,
                                     args.threshold, args.log)
                summarise(results, args.threshold, args.log)
            else:
                run_free(source, predictor, args.threshold)
        finally:
            source.close()


if __name__ == "__main__":
    main()