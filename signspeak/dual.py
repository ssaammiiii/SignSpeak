"""
dual.py — two-glove acquisition, frame pairing and mode arbitration.

Physical set-up: one ESP32 per glove, one USB cable each. The host opens
both ports, identifies which is which from the firmware banner
("GLOVE:RIGHT" / "GLOVE:LEFT"), walks each board through its calibration
in turn, then sends STREAM to both back-to-back. Firmware 2.0 restarts its
50 Hz tick on the STREAM command, so the two streams start in phase.

Pairing (right-driven, nearest frame in either direction):
    every frame from either glove is time-stamped on arrival by its reader
    thread. For each RIGHT frame the LEFT frame closest in time is taken,
    looking both before and after it (waiting up to 25 ms for a later one).
    Matching in both directions bounds the pairing error at half a sample
    period, 10 ms, in normal operation. If the chosen left frame is more
    than PAIR_MAX_AGE_S (45 ms) away the partner is treated as missing and
    the pair is discarded. There is no timing master: the boards never
    communicate, and alignment is done entirely on host arrival times.

Mode arbitration:
    left glove at rest  -> LETTER mode: right-glove channels -> letter model
    left glove active   -> WORD mode:   22 flex + IMU channels  -> word model
    "Rest" is measured against a template captured at start-up (left hand
    relaxed at the side): flex deviation, gravity-vector deviation and gyro
    magnitude all below thresholds (config.LEFT_REST_*).

Public entry points:
    GloveLink          one glove over serial, side-aware handshake
    open_pair()        open two ports, sort out which is which, calibrate both
    DualStream         paired 24-channel frames from two links
    two_glove_source() (symbol, confidence) generator for pipeline.run()
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Iterator, Optional, Tuple

import numpy as np

try:
    from . import config
    from .glove import Classifier
except ImportError:          # running as a script from the package folder
    import config
    from glove import Classifier

from features import (LEFT_CHANNELS, MERGED_CHANNELS, RIGHT_CHANNELS,
                      WORD_CHANNELS, WINDOW)

Gesture = Tuple[str, float]

N_RIGHT = len(RIGHT_CHANNELS)   # 13
N_LEFT = len(LEFT_CHANNELS)     # 11
N_MERGED = len(MERGED_CHANNELS) # 24


# --------------------------------------------------------------------------
# One glove
# --------------------------------------------------------------------------

class GloveLink:
    """One ESP32 glove over USB serial, firmware 2.0 protocol."""

    def __init__(self, port: str, baud: int = 115200, verbose: bool = True,
                 tag: Optional[str] = None):
        try:
            import serial
        except ImportError:
            raise SystemExit("pyserial not installed. Run: uv add pyserial")
        self.port = port
        self.verbose = verbose
        self.tag = tag or port
        self.side: Optional[str] = None      # "RIGHT" / "LEFT" from the banner
        self.n_ch: Optional[int] = None      # channels per frame, from the banner
        self.ser = serial.Serial(port, baud, timeout=1.0)
        # Pulse the reset line so the board restarts NOW, under our control.
        # Without this we depend on the port-open auto-reset, which some
        # CP210x boards and cables do not do - and then the banner and the
        # calibration prompt are printed before we are listening, and the
        # handshake waits for a READY that already went past.
        try:
            self.ser.dtr = False
            self.ser.rts = True
            time.sleep(0.12)
            self.ser.rts = False
        except Exception:
            pass
        time.sleep(0.4)
        self.ser.reset_input_buffer()

    # -- low level --------------------------------------------------------

    def send(self, text: str) -> None:
        self.ser.write((text + "\n").encode())
        self.ser.flush()

    def readline(self) -> str:
        raw = self.ser.readline()
        return raw.decode("utf-8", errors="ignore").strip() if raw else ""

    def _note(self, line: str) -> None:
        """Pick identity out of a banner line, e.g. 'GLOVE:RIGHT CHANNELS:13 FW:2.0'."""
        if line.startswith("GLOVE:"):
            for part in line.split():
                k, _, v = part.partition(":")
                if k == "GLOVE":
                    self.side = v.upper()
                elif k == "CHANNELS" and v.isdigit():
                    self.n_ch = int(v)
            self.tag = self.side or self.tag

    def parse(self, line: str) -> Optional[np.ndarray]:
        """Numeric frame -> array, else None. Uses the first n_ch values."""
        if not line or not (line[0].isdigit() or line[0] in "-."):
            return None
        vals = []
        for p in line.split(","):
            try:
                vals.append(float(p))
            except ValueError:
                return None
        n = self.n_ch or len(vals)
        if len(vals) < n:
            return None
        return np.asarray(vals[:n], dtype=np.float64)

    # -- start-up -------------------------------------------------------------

    def handshake(self, wait_s: float = 120.0) -> None:
        """Boot -> calibration prompts -> READY. Does NOT start streaming."""
        deadline = time.time() + wait_s
        quiet = 0.0
        nudged = False
        while time.time() < deadline:
            line = self.readline()
            if not line:
                quiet += 1.0
                if quiet >= 3.0 and not nudged:
                    # Silent. Either the board is already past its banner,
                    # or it is mid-stream from a previous run. Stop any
                    # stream, then ask it who it is.
                    if self.verbose:
                        print(f"  [{self.tag}] no output yet - nudging the "
                              f"board on {self.port}")
                    self.send("x")
                    time.sleep(0.3)
                    self.ser.reset_input_buffer()
                    self.send("ID")
                    nudged = True
                elif quiet >= 8.0:
                    raise SystemExit(
                        f"\n[{self.tag}] {self.port} is not talking.\n"
                        "  - is this the right COM port? run: "
                        "uv run python which_glove.py\n"
                        "  - is the Arduino Serial Monitor still open?\n"
                        "  - is firmware 2.0 flashed on this board?")
                continue
            quiet = 0.0
            self._note(line)
            if "PRESS_ENTER" in line:
                prompt = line.split("PRESS_ENTER", 1)[1].lstrip(": ").strip()
                try:
                    input(f"\n  [{self.tag}] >>> {prompt} ")
                except (EOFError, KeyboardInterrupt):
                    raise SystemExit("\ncancelled.")
                self.send("")
                deadline = time.time() + wait_s
                continue
            if self.verbose:
                print(f"  [{self.tag}] {line}")
            if "READY" in line:
                break
        else:
            raise SystemExit(f"[{self.tag}] no READY from {self.port}. "
                             "Serial Monitor open? Firmware 2.0 flashed?")

        if self.side is None:
            # Banner was missed (board did not reset on open). Ask.
            self.send("ID")
            t0 = time.time()
            while time.time() - t0 < 3.0 and self.side is None:
                self._note(self.readline())
        if self.side is None:
            raise SystemExit(f"[{self.tag}] board on {self.port} did not "
                             "identify itself. Flash firmware 2.0.")
        want = N_RIGHT if self.side == "RIGHT" else N_LEFT
        if self.n_ch != want:
            raise SystemExit(
                f"[{self.side}] firmware streams {self.n_ch} channels but "
                f"features.py expects {want}. Match TOUCH_SCAN_THUMB_INDEX "
                "in the sketch with TOUCH in features.py.")

    def start_stream(self, wait_s: float = 5.0) -> None:
        self.ser.reset_input_buffer()
        self.send("STREAM")
        t0 = time.time()
        while time.time() - t0 < wait_s:
            line = self.readline()
            if "STREAMING" in line or self.parse(line) is not None:
                return
        raise SystemExit(f"[{self.tag}] STREAM did not start.")

    # -- streaming --------------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        return self.parse(self.readline())

    def stop(self) -> None:
        try:
            self.send("x")            # any byte stops the firmware stream
            time.sleep(0.2)
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        try:
            self.ser.close()
        except Exception:
            pass

    # -- dataset collection (LABEL,REP,N) ----------------------------------------

    def collect(self, label: str, rep: int, n: int, timeout_s: float = 8.0):
        """
        Send LABEL,REP,N and read the rows back until END.
        Returns {sample_index: np.ndarray(n_ch)} and a list of status lines.
        The caller sends this on both gloves; see record_words.py.
        """
        rows, notes = {}, []
        deadline = time.time() + timeout_s + n * 0.02
        while time.time() < deadline:
            line = self.readline()
            if not line:
                continue
            if line == "END":
                return rows, notes
            parts = line.split(",")
            if len(parts) >= 3 and parts[0] == label:
                try:
                    idx = int(parts[2])
                    vals = np.asarray([float(p) for p in parts[3:]], dtype=np.float64)
                except ValueError:
                    notes.append(line)
                    continue
                if self.n_ch and len(vals) >= self.n_ch:
                    rows[idx] = vals[:self.n_ch]
                continue
            notes.append(line)
        notes.append("TIMEOUT waiting for END")
        return rows, notes


def open_pair(port_a: str, port_b: str, verbose: bool = True):
    """
    Open two ports, calibrate both, return (right, left) in the correct
    order no matter which port was given first.
    """
    a = GloveLink(port_a, verbose=verbose)
    b = GloveLink(port_b, verbose=verbose)
    if verbose:
        print("\n=== calibrating the first board ===")
    a.handshake()
    if verbose:
        print("\n=== calibrating the second board ===")
    b.handshake()
    if a.side == b.side:
        raise SystemExit(f"both boards report GLOVE:{a.side}. "
                         "Flash one of them with the other GLOVE_SIDE.")
    right, left = (a, b) if a.side == "RIGHT" else (b, a)
    if verbose:
        print(f"\n  right glove on {right.port}  ({right.n_ch} ch)")
        print(f"  left  glove on {left.port}  ({left.n_ch} ch)\n")
    return right, left


# --------------------------------------------------------------------------
# Threads and pairing
# --------------------------------------------------------------------------

class _Reader(threading.Thread):
    """Reads one link continuously; keeps a time-stamped history of frames."""

    def __init__(self, link: GloveLink, history: int = 200,
                 queue_frames: bool = False):
        super().__init__(daemon=True)
        self.link = link
        self.hist: deque = deque(maxlen=history)   # (t_arrival, frame)
        self.q: Optional[queue.Queue] = queue.Queue() if queue_frames else None
        self.lock = threading.Lock()
        self.count = 0
        self.last_t = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            row = self.link.read()
            if row is None:
                continue
            t = time.monotonic()
            with self.lock:
                self.hist.append((t, row))
                self.count += 1
                self.last_t = t
            if self.q is not None:
                self.q.put((t, row))

    def stop(self) -> None:
        self._stop.set()

    def nearest(self, t: float, wait_s: float = 0.025):
        """
        Closest frame to t in EITHER direction.

        Taking only the most recent earlier frame costs up to a full frame
        period whenever the two boards' ticks are slightly out of phase:
        a left frame landing 2 ms after the right one is discarded in
        favour of one 18 ms older. Waiting briefly for the frame that
        straddles t and then picking whichever side is closer removes that
        systematic lag, at the cost of up to wait_s of latency.
        """
        deadline = time.monotonic() + wait_s
        while True:
            with self.lock:
                hist = list(self.hist)
            before = after = None
            for tl, row in reversed(hist):
                if tl <= t:
                    before = (tl, row)
                    break
                after = (tl, row)
            if after is not None or time.monotonic() > deadline:
                break
            time.sleep(0.002)
        if before is None:
            return after if after is not None else (None, None)
        if after is None:
            return before
        return before if (t - before[0]) <= (after[0] - t) else after


class DualStream:
    """
    Iterates (t, merged_frame, gap_s): merged_frame is right + left in
    MERGED_CHANNELS order, gap_s is how far apart the two frames arrived.
    """

    def __init__(self, right: GloveLink, left: GloveLink,
                 max_age_s: Optional[float] = None, verbose: bool = True):
        self.right, self.left = right, left
        self.max_age = max_age_s if max_age_s is not None else config.PAIR_MAX_AGE_S
        self.verbose = verbose
        self.pairs = 0
        self.stale = 0
        self.r_reader = _Reader(right, queue_frames=True)
        self.l_reader = _Reader(left)

    def start(self) -> None:
        # Back-to-back so both boards restart their tick on STREAM together.
        self.right.start_stream()
        self.left.start_stream()
        self.r_reader.start()
        self.l_reader.start()
        if self.verbose:
            print("[dual] both streams live")

    def __iter__(self):
        quiet = 0.0
        while True:
            try:
                t, r_row = self.r_reader.q.get(timeout=1.0)
            except queue.Empty:
                quiet += 1.0
                if self.verbose and quiet >= 4.0:
                    print("\n[dual] no frames from the RIGHT glove for 4 s. "
                          "Something wrote to its port, or the cable dropped.")
                    quiet = 0.0
                continue
            quiet = 0.0
            tl, l_row = self.l_reader.nearest(t)
            if l_row is None or abs(t - tl) > self.max_age:
                self.stale += 1
                if self.verbose and self.stale in (50, 500, 5000):
                    print(f"\n[dual] {self.stale} right frames had no fresh "
                          "left partner - is the LEFT glove still streaming?")
                continue
            self.pairs += 1
            yield t, np.concatenate([r_row, l_row]), abs(t - tl)

    def rates(self) -> Tuple[int, int, int, int]:
        return self.r_reader.count, self.l_reader.count, self.pairs, self.stale

    def close(self) -> None:
        self.r_reader.stop()
        self.l_reader.stop()
        self.right.close()
        self.left.close()


# --------------------------------------------------------------------------
# Left-hand rest detection (the mode switch)
# --------------------------------------------------------------------------

class RestTemplate:
    def __init__(self, flex: np.ndarray, acc: np.ndarray):
        self.flex = flex          # (5,) mean flex at rest
        self.acc = acc            # (3,) mean gravity vector at rest

    @staticmethod
    def capture(dual: "DualStream", n: int = 100, settle_s: float = 4.0,
                verbose: bool = True):
        """
        Learn what the left hand looks like at rest.

        This is the reference the mode switch compares against, so it has to
        be captured in the posture the hand ACTUALLY returns to between
        words - not while it is still in the calibration fist, and not
        mid-way through lowering. Frames are discarded for settle_s first,
        then n frames are averaged, then the spread is checked: a template
        captured during movement is worse than no template at all.
        """
        if verbose:
            print("\n  Let the LEFT hand rest exactly where it will sit "
                  "between words.")
            print("  Same fingers, same arm angle. Capture starts in...",
                  flush=True)

        t0 = time.monotonic()
        shown = None
        frames = []
        for _, merged, _ in dual:
            left = merged[N_RIGHT:]
            waited = time.monotonic() - t0
            if waited < settle_s:
                left_s = int(settle_s - waited) + 1
                if verbose and left_s != shown:
                    print(f"\r  ...{left_s}   ", end="", flush=True)
                    shown = left_s
                continue
            frames.append(left)
            if verbose and len(frames) % 25 == 0:
                print(f"\r  capturing {len(frames)}/{n}   ", end="", flush=True)
            if len(frames) >= n:
                break

        w = np.stack(frames)
        tpl = RestTemplate(w[:, :5].mean(axis=0), w[:, 5:8].mean(axis=0))

        # Was the hand actually still while we measured it?
        flex_jitter = float(w[:, :5].std(axis=0).mean())
        gyro_mag = float(np.linalg.norm(w[:, 8:11], axis=1).mean())
        if verbose:
            print(f"\r  rest template: flex {np.round(tpl.flex, 2)}  "
                  f"acc {np.round(tpl.acc, 2)}")
            if flex_jitter > 0.05 or gyro_mag > 20:
                print(f"  ** the hand was MOVING during capture "
                      f"(flex jitter {flex_jitter:.3f}, gyro {gyro_mag:.0f}). "
                      "Restart and hold it still. **")
            else:
                print(f"  (steady: flex jitter {flex_jitter:.3f}, "
                      f"gyro {gyro_mag:.0f})")
            print()
        return tpl

    def is_active(self, left_window: np.ndarray) -> Tuple[bool, dict]:
        """left_window: (n, 11) in LEFT_CHANNELS order."""
        flex_dev = float(np.abs(left_window[:, :5].mean(axis=0) - self.flex).mean())
        acc_dev = float(np.linalg.norm(left_window[:, 5:8].mean(axis=0) - self.acc))
        gyro = float(np.linalg.norm(left_window[:, 8:11], axis=1).mean())
        active = (flex_dev > config.LEFT_REST_FLEX_DEV
                  or acc_dev > config.LEFT_REST_ACC_DEV
                  or gyro > config.LEFT_REST_GYRO)
        return active, {"flex_dev": flex_dev, "acc_dev": acc_dev, "gyro": gyro}


# --------------------------------------------------------------------------
# Debouncers (one per path so the two never interfere)
# --------------------------------------------------------------------------

def word_motion(frame: np.ndarray, n_right: int = N_RIGHT):
    """
    Motion energy of a merged window, and whether the motion settled.

    Returns (energy, settled, peak):
      energy   mean gyroscope magnitude over both gloves, deg/s
      settled  True if the last fifth of the window is quieter than
               WORD_SETTLE_RATIO of the window's peak
      peak     the loudest instant, for tuning

    Gyroscope rather than accelerometer because gravity sits in the
    accelerometer at all times: a motionless hand still reads ~1 g, so an
    accelerometer threshold would measure orientation as much as movement.
    The gyroscope reads zero when still, whatever the hand's pose.
    """
    r_gyr = frame[:, n_right - 3:n_right]
    l_gyr = frame[:, -3:]
    mag = 0.5 * (np.linalg.norm(r_gyr, axis=1) + np.linalg.norm(l_gyr, axis=1))
    energy = float(mag.mean())
    peak = float(mag.max())
    tail = mag[-max(1, len(mag) // 5):]
    settled = bool(tail.mean() <= config.WORD_SETTLE_RATIO * peak) if peak > 1e-6 else True
    return energy, settled, peak


class _LetterDebounce:
    """Same rule as glove.esp32_source: hold N agreeing frames, latch, release."""

    def __init__(self, hold: int):
        self.hold = hold
        self.reset()

    def reset(self) -> None:
        self.candidate = None
        self.agree = 0
        self.latched = False
        self.cooldown = False

    def feed(self, symbol: str, conf: float, threshold: float) -> Optional[str]:
        if conf < threshold or symbol == "neutral":
            self.reset()
            return None
        if self.cooldown:
            return None
        if symbol == self.candidate:
            self.agree += 1
        else:
            self.candidate, self.agree, self.latched = symbol, 1, False
        if self.agree >= self.hold and not self.latched:
            self.latched = True
            if symbol == "speak":
                self.cooldown = True
            return symbol
        return None


class _WordDebounce:
    """Emit once per sign; re-arm on NONE / low confidence / rest / timeout."""

    def __init__(self, hold: int, refractory_s: float):
        self.hold = hold
        self.refractory = refractory_s
        self.reset()

    def reset(self) -> None:
        self.candidate = None
        self.agree = 0
        self.armed = True
        self.fired_at = 0.0

    def feed(self, symbol: str, conf: float, threshold: float, now: float) -> Optional[str]:
        if symbol == "none" or conf < threshold:
            self.candidate, self.agree = None, 0
            if now - self.fired_at > self.refractory:
                self.armed = True
            return None
        if not self.armed:
            return None
        if symbol == self.candidate:
            self.agree += 1
        else:
            self.candidate, self.agree = symbol, 1
        if self.agree >= self.hold:
            self.armed = False
            self.fired_at = now
            self.candidate, self.agree = None, 0
            return symbol
        return None


# --------------------------------------------------------------------------
# The source
# --------------------------------------------------------------------------

def two_glove_source(
    right_port: str,
    left_port: str,
    results_dir: str = "results",
    letter_model: str = "lda",
    word_results_dir: str = "results_words",
    word_model: str = "lda",
    threshold: Optional[float] = None,
    word_threshold: Optional[float] = None,
    hold_frames: int = 4,
    word_hold: Optional[int] = None,
    stride: int = 8,
    motion_limit: float = 0.06,
    verbose: bool = True,
) -> Iterator[Gesture]:
    """
    Yield (symbol, confidence) from two gloves.

    Letters come out as single lowercase characters plus the control
    symbols "neutral" / "speak". Word signs come out as whole lowercase
    words ("hungry", "my name", "pain"). pipeline.run() handles both.
    """
    threshold = config.CONFIDENCE_THRESHOLD if threshold is None else threshold
    word_threshold = config.WORD_CONFIDENCE if word_threshold is None else word_threshold
    word_hold = config.WORD_HOLD_FRAMES if word_hold is None else word_hold

    letter_clf = Classifier(results_dir, letter_model)
    word_clf = Classifier(word_results_dir, word_model)
    n_letter = len(letter_clf.channels)
    if n_letter > N_RIGHT:
        raise SystemExit("letter model wants more channels than the right glove has")
    # The word model may use a subset of the merged frame (by default the
    # 22 flex+IMU channels, without the right glove's contact pads), so
    # build an index into the merged frame from its saved channel list.
    try:
        word_cols = [MERGED_CHANNELS.index(c) for c in word_clf.channels]
    except ValueError as e:
        raise SystemExit(f"word model names a channel the merged frame does "
                         f"not have: {e}")
    if verbose:
        print(f"[dual] letter model {letter_model} ({n_letter} ch, "
              f"gate {threshold}), word model {word_model} "
              f"({len(word_clf.channels)} ch, gate {word_threshold})")

    right, left = open_pair(right_port, left_port, verbose=verbose)
    dual = DualStream(right, left, verbose=verbose)
    dual.start()
    try:
        rest = RestTemplate.capture(dual, verbose=verbose)

        window: deque = deque(maxlen=WINDOW)
        since = 0
        mode = "letter"
        ldb = _LetterDebounce(hold_frames)
        wdb = _WordDebounce(word_hold, config.WORD_REFRACTORY_S)
        shown = None

        def show(text: str) -> None:
            nonlocal shown
            if verbose and text != shown:
                print(f"\r  {text:<60s}", end="", flush=True)
                shown = text

        for t, merged, age in dual:
            window.append(merged)
            if len(window) < WINDOW:
                continue
            since += 1
            if since < stride:
                continue
            since = 0

            frame = np.stack(window)                    # (50, 25)
            right_w = frame[:, :N_RIGHT]
            left_w = frame[:, N_RIGHT:]

            active, m = rest.is_active(left_w)
            near = (f"L flex{m['flex_dev']:5.2f}/{config.LEFT_REST_FLEX_DEV:.2f} "
                    f"acc{m['acc_dev']:5.2f}/{config.LEFT_REST_ACC_DEV:.2f} "
                    f"gyr{m['gyro']:5.1f}/{config.LEFT_REST_GYRO:.0f}")
            new_mode = "word" if active else "letter"
            if new_mode != mode:
                mode = new_mode
                ldb.reset()
                wdb.reset()
                if verbose:
                    print(f"\n[dual] {mode.upper()} mode  "
                          f"(flex {m['flex_dev']:.2f} acc {m['acc_dev']:.2f} "
                          f"gyro {m['gyro']:.0f})")

            if mode == "word":
                energy, settled, peak = word_motion(frame)
                if energy < config.WORD_MOTION_MIN:
                    # Too still to be a sign. Re-arm so the next real sign
                    # is not swallowed by the refractory period.
                    wdb.feed("none", 0.0, word_threshold, t)
                    show(f"WORD   ...too still  {energy:5.1f}   {near}")
                    continue
                if config.WORD_REQUIRE_SETTLE and not settled:
                    show(f"WORD   ...still moving  {energy:5.1f}   {near}")
                    continue

                symbol, conf = word_clf.predict(frame[:, word_cols])
                show(f"WORD   {symbol:<10s} {conf:4.2f}  motion {energy:5.1f}")
                out = wdb.feed(symbol, conf, word_threshold, t)
                if out is not None:
                    text = config.WORD_TEXT.get(out.upper(), out)
                    if verbose:
                        print(f"\r  >>> {text.upper():<12s} {conf:4.2f}{' ' * 30}")
                    shown = None
                    yield text, conf
                continue

            # ---- letter mode: identical to the single-glove path ----
            if float(right_w[:, :5].std(axis=0).mean()) > motion_limit:
                ldb.reset()
                show("LETTER ...moving")
                continue
            symbol, conf = letter_clf.predict(right_w[:, :n_letter])
            show(f"LETTER {symbol:<10s} {conf:4.2f}")
            out = ldb.feed(symbol, conf, threshold)
            if out is not None:
                if verbose:
                    print(f"\r  >>> {out.upper():<12s} {conf:4.2f}{' ' * 30}")
                shown = None
                yield out, conf
                if out == "speak":
                    window.clear()
    except KeyboardInterrupt:
        if verbose:
            r, l, p, s = dual.rates()
            print(f"\n[dual] stopped. right {r} left {l} frames, "
                  f"{p} pairs, {s} stale")
    finally:
        dual.close()
