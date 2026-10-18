"""
which_glove.py — ask every connected board which glove it is.

COM numbers can move between sessions, so run this first and use what it
prints. It opens each CP210x/CH340 port, resets the board, and reads the
boot banner.

    python which_glove.py
    python which_glove.py --ports COM6 COM7
"""

import argparse
import time


def banner(port, baud=115200, wait_s=6.0):
    import serial
    try:
        ser = serial.Serial(port, baud, timeout=0.5)
    except Exception as e:
        return None, None, f"could not open: {e}"
    try:
        ser.dtr = False; ser.rts = True; time.sleep(0.1)   # pulse reset
        ser.rts = False
        t0 = time.time()
        lines = []
        while time.time() - t0 < wait_s:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            lines.append(line)
            if line.startswith("GLOVE:"):
                side = ch = None
                for part in line.split():
                    k, _, v = part.partition(":")
                    if k == "GLOVE":
                        side = v.upper()
                    elif k == "CHANNELS":
                        ch = v
                return side, ch, line
        hint = lines[-1] if lines else "silent"
        return None, None, f"no GLOVE: banner (last line: {hint})"
    finally:
        ser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="*", default=None)
    args = ap.parse_args()

    ports = args.ports
    if not ports:
        import serial.tools.list_ports
        ports = [p.device for p in serial.tools.list_ports.comports()
                 if any(k in p.description for k in
                        ("CP210", "CH340", "FTDI", "USB Serial"))]
    if not ports:
        raise SystemExit("no USB-serial boards found. Check the cables.")

    print(f"checking {len(ports)} port(s)... (close the Serial Monitor first)\n")
    found = {}
    for p in ports:
        side, ch, note = banner(p)
        if side:
            found.setdefault(side, []).append(p)
            print(f"  {p:8s} GLOVE:{side}  {ch} channels")
        else:
            print(f"  {p:8s} ?         {note}")

    print()
    for side in ("RIGHT", "LEFT"):
        ps = found.get(side, [])
        if len(ps) > 1:
            print(f"  ** two boards both say {side}: {' '.join(ps)} - "
                  "reflash one with the other GLOVE_SIDE")
    if "RIGHT" in found and "LEFT" in found:
        r, l = found["RIGHT"][0], found["LEFT"][0]
        print(f"  uv run python -m signspeak.main --port {r} "
              f"--left-port {l} --model lda --word-model lda")
        print(f"  uv run python sync_check.py --ports {r} {l}")
        print(f"  uv run python record_words.py --ports {r} {l}")
    elif "RIGHT" in found:
        print("  only the right glove answered. The left board is either not "
              "flashed with firmware 2.0, or not connected.")


if __name__ == "__main__":
    main()
