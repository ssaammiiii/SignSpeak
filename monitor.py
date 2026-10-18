"""
monitor.py — watch the flex sensors live, one bar per finger.

Two modes:

  raw     Raw millivolts, straight off the ADC. This is what you want after
          swapping a sensor or rewiring. It shows each sensor's true
          response and the span between flat and bent, which is the number
          that actually matters. Needs the RAW command in firmware.

  norm    The calibrated 0.0-1.0 values the model actually sees. Use this
          to confirm calibration is sane before collecting data.

While it runs it tracks the min and max each sensor has reached, so you can
bend each finger in turn and read off the span without recording anything.

Usage:
    python monitor.py --port COM7                  # raw millivolts
    python monitor.py --port COM7 --mode norm      # calibrated 0-1

Press Ctrl+C to stop. A summary of each sensor's observed range prints
when you exit.
"""

import argparse
import time

import numpy as np

FINGERS = ["thumb ", "index ", "middle", "ring  ", "pinky "]


def bar(value, lo, hi, width=28):
    """Draws a bar positioned between lo and hi."""
    if hi - lo < 1e-9:
        return "░" * width
    frac = (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def open_port(port, baud, cmd):
    try:
        import serial
    except ImportError:
        raise SystemExit("pyserial not installed. Run: uv add pyserial")

    ser = serial.Serial(port, baud, timeout=1.0)
    time.sleep(2.0)                 # the board reboots when the port opens
    ser.reset_input_buffer()

    print("connecting...\n")
    deadline = time.time() + 90.0
    last_send = 0.0
    waiting = False
    armed = True

    def send(text):
        ser.write((text + "\n").encode())
        ser.flush()

    def parse(line, n):
        vals = []
        for p in line.split(","):
            try:
                vals.append(float(p.strip()))
            except ValueError:
                pass
        return vals if len(vals) >= n else None

    while time.time() < deadline:
        raw = ser.readline()
        line = raw.decode("utf-8", errors="ignore").strip() if raw else ""

        if line:
            if parse(line, 5) is not None:
                print("  stream is live\n")
                return ser
            print(f"  esp32 | {line}")

            if "PRESS_ENTER" in line:
                waiting = True
                try:
                    input("\n  >>> do that, then press Enter HERE: ")
                except (EOFError, KeyboardInterrupt):
                    raise SystemExit("\ncancelled.")
                send("")
                last_send = time.time()
                waiting = False
                deadline = time.time() + 90.0
            elif "RAWSTREAM" in line or "STREAMING" in line:
                armed = False       # any further byte would stop the stream
            elif "Calibration complete" in line or "READY" in line:
                waiting = False
                if armed and time.time() - last_send > 3.0:
                    send(cmd)
                    last_send = time.time()
            continue

        if armed and not waiting and time.time() - last_send > 3.0:
            send(cmd)
            last_send = time.time()

    raise SystemExit("\n*** No data. Is the Serial Monitor closed, and is "
                     "the firmware with the RAW command uploaded?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mode", choices=["raw", "norm"], default="raw")
    args = ap.parse_args()

    cmd = "RAW" if args.mode == "raw" else "STREAM"
    unit = "mV" if args.mode == "raw" else "  "
    ser = open_port(args.port, args.baud, cmd)

    seen_min = np.full(5, np.inf)
    seen_max = np.full(5, -np.inf)
    n = 0

    if args.mode == "raw":
        print("Bend each finger fully, one at a time, and watch the span.")
        print("A healthy sensor should move several hundred mV.\n")
    else:
        print("Flat hand should read near 0.00, fist near 1.00.\n")

    print("\n" * 8)   # reserve screen space

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()

            vals = []
            for p in line.split(","):
                try:
                    vals.append(float(p.strip()))
                except ValueError:
                    pass
            if len(vals) < 5:
                continue
            v = np.array(vals[:5], dtype=float)

            n += 1
            seen_min = np.minimum(seen_min, v)
            seen_max = np.maximum(seen_max, v)

            # Redraw in place.
            print("\033[8A", end="")
            print(f"  {'finger':7s} {'value':>8s} {'min':>8s} {'max':>8s} "
                  f"{'span':>8s}   {'live':<28s}")
            for i in range(5):
                lo, hi = seen_min[i], seen_max[i]
                span = hi - lo
                if args.mode == "raw":
                    flag = "  weak" if n > 100 and span < 150 else ""
                else:
                    lo, hi = 0.0, 1.0
                    span = seen_max[i] - seen_min[i]
                    flag = "  weak" if n > 100 and span < 0.5 else ""
                print(f"  {FINGERS[i]}  {v[i]:8.1f} {seen_min[i]:8.1f} "
                      f"{seen_max[i]:8.1f} {seen_max[i] - seen_min[i]:8.1f}   "
                      f"{bar(v[i], lo, hi)}{flag}")
            print(f"  {n} samples{unit:>10s}          Ctrl+C to stop      ")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ser.write(b"x\n")
            ser.flush()
            time.sleep(0.2)
            ser.close()
        except Exception:
            pass

    print("\n" + "=" * 58)
    print("  observed range per sensor")
    print("=" * 58)
    print(f"  {'finger':8s} {'min':>9s} {'max':>9s} {'span':>9s}   verdict")
    for i in range(5):
        span = seen_max[i] - seen_min[i]
        if args.mode == "raw":
            verdict = ("DEAD" if span < 50 else
                       "weak" if span < 150 else
                       "ok")
        else:
            verdict = ("DEAD" if span < 0.2 else
                       "weak" if span < 0.5 else
                       "ok")
        print(f"  {FINGERS[i]} {seen_min[i]:9.1f} {seen_max[i]:9.1f} "
              f"{span:9.1f}   {verdict}")
    print()
    if args.mode == "raw":
        print("  A sensor reading a flat line never changed — check the")
        print("  solder joint, the divider resistor, and the GPIO pin.")


if __name__ == "__main__":
    main()
