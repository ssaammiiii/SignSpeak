"""
Real hardware gesture source: ESP32 glove -> trained model -> (symbol, confidence).

Drop-in replacement for `fake_source()`. Same contract, so `pipeline.run()`
needs no changes:

    from signspeak.glove import esp32_source
    from signspeak import pipeline

    pipeline.run(esp32_source(port="COM7", model="lda"))

Why this module exists rather than a few lines in sources.py: the classifier
runs on a *rolling* window and emits a prediction about six times a second.
Signing one letter therefore produces the same prediction eight or ten times in
a row. The pipeline appends every symbol it accepts, so "HELLO" would arrive as
"HHHHHEEEELLLLLLLLOOOO". This source debounces that down to one symbol per
hand shape before yielding.

Debounce rule: a symbol is emitted once it has been predicted `hold_frames`
times in a row above the confidence threshold. After that the source is
"latched" and will not emit again until the hand leaves that shape — either by
going to NEUTRAL, by dropping below threshold, or by settling on a different
symbol. That maps directly onto how a person actually signs: form, hold,
release, form the next one.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Iterator, Tuple

import numpy as np

try:                     # normal use, inside the package
    from . import config
except ImportError:      # running this file directly
    import config

Gesture = Tuple[str, float]

CHANNELS = [
    "flex1", "flex2", "flex3", "flex4", "flex5",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "touch_index_mid", "touch_thumb_mid",
]
N_CH = len(CHANNELS)

# Control gestures the model knows. Everything else is a letter.
# Note there is no SPACE class: word segmentation is left to the LLM in
# sentence.py, which is prompted to handle an unspaced stream.
CONTROL = {"neutral", "speak"}


# --------------------------------------------------------------------------
# Serial
# --------------------------------------------------------------------------

class _Serial:
    """Opens the port, walks the firmware through calibration, then streams."""

    def __init__(self, port: str, baud: int = 115200, start_cmd: str = "STREAM",
                 verbose: bool = True):
        try:
            import serial
        except ImportError:
            raise SystemExit("pyserial not installed. Run: uv add pyserial")

        self.verbose = verbose
        self.ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(2.0)              # the board reboots when the port opens
        self.ser.reset_input_buffer()
        self._warned = False
        self._handshake(start_cmd)

    def _send(self, text: str) -> None:
        self.ser.write((text + "\n").encode())
        self.ser.flush()

    def _readline(self) -> str:
        raw = self.ser.readline()
        if not raw:
            return ""
        return raw.decode("utf-8", errors="ignore").strip()

    def parse(self, line: str):
        if not line:
            return None
        values = []
        for part in line.split(","):
            try:
                values.append(float(part.strip()))
            except ValueError:
                pass
        if len(values) < N_CH:
            # Old firmware. Say so once instead of timing out silently.
            if len(values) >= 11 and not self._warned and self.verbose:
                print(f"\n*** the glove streams {len(values)} channels but "
                      f"this model needs {N_CH}.")
                print("*** Flash the firmware with the contact pads, or use "
                      "an 11-channel model.\n")
                self._warned = True
            return None
        if len(values) > N_CH:
            if not self._warned and self.verbose:
                print(f"[glove] {len(values)} numbers per line, "
                      f"using the first {N_CH}")
                self._warned = True
            # The FIRST N_CH, not the last. A STREAM frame has no
            # label/rep/sample prefix, so taking the tail would silently
            # drop flex1 and flex2 and shift every channel two places.
            values = values[:N_CH]
        return np.array(values, dtype=np.float64)

    def _handshake(self, start_cmd: str, wait_s: float = 90.0) -> None:
        """
        The firmware's stream loop is `while (Serial.available() == 0)`, so any
        byte sent after streaming begins stops it. Send the start command
        sparingly, and stop entirely once the board confirms it is streaming.
        """
        if self.verbose:
            print("[glove] connecting...")
        deadline = time.time() + wait_s
        last_send = 0.0
        waiting_for_enter = False
        armed = True

        while time.time() < deadline:
            line = self._readline()

            if line:
                if self.parse(line) is not None:
                    if self.verbose:
                        print("[glove] stream is live\n")
                    return
                if self.verbose:
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
                    armed = False
                elif "Calibration complete" in line or "READY" in line:
                    waiting_for_enter = False
                    if armed and time.time() - last_send > 3.0:
                        self._send(start_cmd)
                        last_send = time.time()
                continue

            if armed and not waiting_for_enter and time.time() - last_send > 3.0:
                self._send(start_cmd)
                last_send = time.time()

        raise SystemExit(
            "[glove] no sensor data. Is the Arduino Serial Monitor closed, and "
            "is the firmware with the STREAM command uploaded?"
        )

    def read(self):
        return self.parse(self._readline())

    def close(self) -> None:
        try:
            self._send("x")        # any byte stops the firmware stream
            time.sleep(0.2)
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class Classifier:
    """Wraps a trained model so deep and classical models look the same."""

    def __init__(self, results_dir: str = "results", model: str = "bilstm"):
        import joblib
        from pathlib import Path

        d = Path(results_dir)
        self.name = model
        self.deep = model in ("bilstm", "cnn1d")

        if self.deep:
            from tensorflow import keras
            self.model = keras.models.load_model(d / f"model_{model}.keras")
            scaler = joblib.load(d / "scaler_deep.joblib")
            self.mu, self.sd = scaler["mean"], scaler["std"]
            if list(scaler["channels"]) != CHANNELS:
                raise ValueError(
                    "channel order in scaler_deep.joblib does not match "
                    "CHANNELS in glove.py — retrain or fix the order"
                )
        else:
            self.model = joblib.load(d / f"model_{model}.joblib")
            clf = getattr(self.model, "named_steps", {}).get("clf")
            if hasattr(clf, "n_jobs"):
                clf.n_jobs = 1      # single-window predict; threads only hurt

        self.encoder = joblib.load(d / "label_encoder.joblib")

        # Which channels was this model trained on? train_words.py saves the
        # list; the letter results folder has no such file, so fall back to
        # the scaler (deep) or to the feature count (classical, 5 per
        # channel). This lets one Classifier serve the 13-channel letter
        # model and the 22-channel word model.
        if (d / "channels.joblib").exists():
            self.channels = list(joblib.load(d / "channels.joblib"))
        elif self.deep:
            self.channels = list(joblib.load(d / "scaler_deep.joblib")["channels"])
        else:
            n_feat = int(self.model.n_features_in_)
            if n_feat % 5:
                raise ValueError(f"{n_feat} features is not 5 per channel")
            self.channels = list(CHANNELS[:n_feat // 5])

    def predict(self, window: np.ndarray) -> Gesture:
        """window: (50, N_CH) -> (symbol, confidence). Symbol is lowercased."""
        if self.deep:
            x = ((window - self.mu) / self.sd)[None, ...].astype(np.float32)
            probs = self.model.predict(x, verbose=0)[0]
        else:
            x = _features(window)[None, :]
            probs = self.model.predict_proba(x)[0]

        i = int(np.argmax(probs))
        label = self.encoder.inverse_transform([i])[0]
        return str(label).lower(), float(probs[i])


def _features(window: np.ndarray) -> np.ndarray:
    """Five statistics per channel, in the order training used (ddof=0):
    65 features for the 13-channel letter model, 110 for the word model."""
    # Width comes from the window, not the module constant, so one helper
    # serves the 13-channel letter model and the 22-channel word model.
    n_ch = window.shape[1]
    out = np.empty(n_ch * 5, dtype=np.float64)
    for i in range(n_ch):
        v = window[:, i]
        lo, hi = v.min(), v.max()
        base = i * 5
        out[base:base + 5] = (v.mean(), v.std(), lo, hi, hi - lo)
    return out


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------

def esp32_source(
    port: str,
    results_dir: str = "results",
    model: str = "bilstm",
    baud: int = 115200,
    threshold: float | None = None,
    hold_frames: int = 4,
    stride: int = 8,
    motion_limit: float = 0.06,
    verbose: bool = True,
) -> Iterator[Gesture]:
    """
    Yield one ``(symbol, confidence)`` per hand shape, debounced.

    port        serial port, e.g. "COM7"
    results_dir folder holding model_*.keras / model_*.joblib
    model       "bilstm", "cnn1d", "lda", "logreg", "random_forest", "svm_rbf"
    threshold   confidence gate; defaults to config.CONFIDENCE_THRESHOLD
    hold_frames consecutive agreeing predictions needed to commit a symbol
    stride      predict every N samples (8 at 50 Hz is about 6 predictions/sec)
    motion_limit
                mean flex standard deviation above which the window is treated
                as hand movement rather than a held sign, and skipped. Held
                signs in the training data sit at 0.002-0.022; a window
                straddling two gestures measures around 0.27. 0.06 sits well
                clear of both. Without this gate the classifier is asked to
                label half-formed hand shapes and will confidently return a
                wrong letter.
    """
    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD

    clf = Classifier(results_dir, model)
    if verbose:
        print(f"[glove] model {model}, threshold {threshold}, "
              f"hold {hold_frames} frames")

    link = _Serial(port, baud, verbose=verbose)
    window = deque(maxlen=config.WINDOW_SAMPLES)

    candidate: str | None = None
    agree = 0
    latched = False           # True once a symbol has been emitted
    since_predict = 0
    checked = False
    cooldown = False
    last_shown: str | None = None

    try:
        while True:
            row = link.read()
            if row is None:
                continue

            window.append(row)
            if len(window) < config.WINDOW_SAMPLES:
                if verbose and len(window) % 10 == 0:
                    print(f"\r[glove] filling window "
                          f"{len(window)}/{config.WINDOW_SAMPLES}",
                          end="", flush=True)
                continue

            since_predict += 1
            if since_predict < stride:
                continue
            since_predict = 0

            frame = np.stack(window)

            if not checked:
                _sanity_check(frame, verbose)
                checked = True

            # Skip windows captured mid-movement.
            if float(frame[:, :5].std(axis=0).mean()) > motion_limit:
                candidate, agree, latched = None, 0, False
                cooldown = False
                if verbose and last_shown != "~moving":
                    print("\r  ...moving                 ", end="", flush=True)
                    last_shown = "~moving"
                continue

            symbol, confidence = clf.predict(frame)

            # Below the gate, or resting: release the latch so the next
            # deliberate hand shape can be emitted.
            if confidence < threshold or symbol == "neutral":
                candidate, agree, latched = None, 0, False
                cooldown = False        # hand released; re-arm
                if verbose and symbol != last_shown:
                    print(f"\r  ...{symbol:<10s} {confidence:4.2f}      ",
                          end="", flush=True)
                    last_shown = symbol
                continue

            if cooldown:
                # Still holding the gesture that just fired. Wait for release.
                continue

            if symbol == candidate:
                agree += 1
            else:
                candidate, agree, latched = symbol, 1, False

            if agree >= hold_frames and not latched:
                latched = True
                if verbose:
                    print(f"\r  >>> {symbol.upper():<10s} {confidence:4.2f}"
                          f"{' ' * 12}")
                last_shown = None
                yield symbol, confidence

                if symbol == "speak":
                    # Drop the buffered window so the trailing frames of the
                    # SPEAK gesture are not refilled and read as a second
                    # SPEAK. Require the hand to return to a resting or
                    # low-confidence state before anything is emitted again.
                    window.clear()
                    candidate, agree = None, 0
                    latched = True
                    cooldown = True

    except KeyboardInterrupt:
        if verbose:
            print("\n[glove] stopped.")
    finally:
        link.close()


def _sanity_check(window: np.ndarray, verbose: bool) -> None:
    """Flex channels must arrive calibrated to 0-1, as in training."""
    flex = window[:, :5]
    lo, hi = float(flex.min()), float(flex.max())
    if hi > 1.5 or lo < -0.5:
        print(f"\n*** WARNING: flex values range {lo:.1f} to {hi:.1f}, "
              f"expected 0.0 to 1.0.")
        print("*** The firmware is sending uncalibrated readings, or the "
              "channel order is wrong.")
        print("*** Predictions will be meaningless until this is fixed.\n")
        return

    imu = window[:, 5:]
    if np.all(imu.std(axis=0) < 1e-9):
        print("\n*** WARNING: the IMU is frozen — every reading is identical.")
        print("*** Check the I2C wiring on GPIO 21/22 before trusting "
              "predictions.\n")