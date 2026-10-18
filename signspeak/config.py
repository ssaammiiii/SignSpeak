"""
Central configuration for SignSpeak.

Every tunable parameter lives here, so it can be adjusted in one place without
touching pipeline logic. The Gemini API key is NOT stored here: it is read from
the GEMINI_API_KEY environment variable, loaded from a .env file in the project
root that is kept out of version control.
"""

from pathlib import Path
from dotenv import load_dotenv

# .env sits in the project root, one level above this package.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Letter path (signspeak/glove.py)
# ---------------------------------------------------------------------------

# Letter predictions below this confidence are discarded.
CONFIDENCE_THRESHOLD: float = 0.75

# Sampling and window size. 50 samples at 50 Hz is a 1.0 s window.
SAMPLE_RATE_HZ: int = 50
WINDOW_SAMPLES: int = 50

# ---------------------------------------------------------------------------
# Sentence construction (signspeak/sentence.py)
# ---------------------------------------------------------------------------

# LLM that turns the token stream into a grammatical sentence.
LLM_MODEL: str = "gemini-3.5-flash"
LLM_MAX_TOKENS: int = 200

# ---------------------------------------------------------------------------
# Two-glove settings (signspeak/dual.py)
# ---------------------------------------------------------------------------

# Word signs the word model knows. Labels are what you type into
# record_words.py; WORD_TEXT is what gets spoken. NONE is the "left hand is
# up but nothing is being signed" class and is never spoken.
WORD_LABELS: list[str] = ["DOCTOR", "NOT", "FINE", "GOOD", "DIZZY", "WELL",
                          "FEVER", "SICK", "HOSPITAL", "HELP", "NONE"]
WORD_TEXT: dict[str, str] = {
    "DOCTOR": "doctor", "NOT": "not", "FINE": "fine", "GOOD": "good",
    "DIZZY": "dizzy", "WELL": "well", "FEVER": "fever", "SICK": "sick",
    "HOSPITAL": "hospital", "HELP": "help",
}

# Word predictions below this confidence are ignored.
WORD_CONFIDENCE: float = 0.80
# Agreeing predictions needed before a word is emitted (signs are dynamic,
# so this is lower than the letter hold of 4).
WORD_HOLD_FRAMES: int = 2
# After a word fires, nothing else fires for this long unless the model
# first returns NONE (stops one sign being read twice from its tail).
WORD_REFRACTORY_S: float = 1.5

# Word-mode motion gate. Every word sign involves movement, so a window
# with almost no motion is a pause or a hold, never a completed sign, and a
# window still accelerating at its trailing edge is a sign caught mid-flight
# where the model's confidence means least. This is the word-mode equivalent
# of the letter path's stillness gate, inverted: letters are rejected for
# moving, words are rejected for NOT moving.
#
# Motion energy is the mean gyroscope magnitude across both gloves, deg/s.
# Set WORD_MOTION_MIN to 0 to disable the gate entirely.
WORD_MOTION_MIN: float = 18.0
# Require the sign to have finished inside the window: the last fifth of the
# window must be quieter than WORD_SETTLE_RATIO of the window's peak motion.
# Set False if your signs run right to the edge of the 1 s window.
WORD_REQUIRE_SETTLE: bool = True
WORD_SETTLE_RATIO: float = 0.55

# Frame pairing: a left frame further than this from the right frame it is
# paired with is treated as missing and the pair is dropped.
PAIR_MAX_AGE_S: float = 0.045          # about two frame periods at 50 Hz

# Left-hand rest detection = the mode switch. All three are compared with
# a template captured at start-up with the left hand hanging relaxed.
LEFT_REST_FLEX_DEV: float = 0.15       # mean |flex - rest| over the window (0-1 units)
LEFT_REST_ACC_DEV: float = 0.30        # |mean accel - rest accel| in g
LEFT_REST_GYRO: float = 25.0           # mean gyro magnitude, deg/s
