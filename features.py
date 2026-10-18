"""
features.py — shared channel layout and feature extraction for SignSpeak.

IMPORTANT: this module is used by BOTH training and live inference.
Never change a channel order or the statistic order without retraining,
or the model will receive features in the wrong slots and silently fail.

Two gloves, four channel layouts:

    RIGHT_CHANNELS   what the right glove streams, and what the LETTER
                     model sees.                           13 channels
    LEFT_CHANNELS    what the left glove streams.          11 channels
    MERGED_CHANNELS  right + left, paired frame by frame.  24 channels
    WORD_CHANNELS    what the WORD model sees: the merged
                     frame without the right glove's two
                     contact pads.                         22 channels

`CHANNELS` is kept as an alias of RIGHT_CHANNELS so the existing letter
tools (predict_live, live_test, diagnose_live, train_deep) keep working.

"""

import numpy as np

FLEX = ["flex1", "flex2", "flex3", "flex4", "flex5"]
IMU = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
# Two contact bits. The middle-finger pad is GND; the index pad (GPIO 13)
# and the thumb pad (GPIO 4) each pull their own pin low when they land on
# it. Order here must match the firmware's print order.
# Contact pads are ENABLED (requires ENABLE_TOUCH 1 in the sketch). They
# discriminate A/T (thumb-to-middle) and U/V (index-to-middle), which flex
# + IMU alone cannot separate: with the pads off the letter model scored
# 0.90 and ~90% of its errors were those two pairs; with them on it scores
# 0.99. To disable again, set TOUCH = [] here AND ENABLE_TOUCH 0 in the
# firmware, then retrain the letter model.
TOUCH: list[str] = ["touch_index_mid", "touch_thumb_mid"]

RIGHT_CHANNELS = FLEX + IMU + TOUCH          # 13
LEFT_CHANNELS = FLEX + IMU                   # 11
MERGED_CHANNELS = (["r_" + c for c in RIGHT_CHANNELS]
                   + ["l_" + c for c in LEFT_CHANNELS])   # 24

# What the WORD model trains on: MERGED_CHANNELS without the right glove's
# two contact pads. The pads separate one-handed letters and carry nothing
# about a two-handed sign. Kept as its own list so the pads change the
# letter model without changing the word model.
WORD_CHANNELS = (["r_" + c for c in FLEX + IMU]
                 + ["l_" + c for c in FLEX + IMU])        # 22

CHANNELS = RIGHT_CHANNELS                    # letter-model alias (legacy name)
LEGACY_CHANNELS = FLEX + IMU                 # the 11-channel layout of Dataset 1-3

STATS = ["mean", "std", "min", "max", "rng"]

WINDOW = 50              # samples per letter window (1 s at 50 Hz)
WORD_WINDOW = 50         # samples the word model consumes
WORD_RECORD_SAMPLES = 75 # samples recorded per word repetition (1.5 s);
                         # train_words.py crops WORD_WINDOW out of these


def feature_names(channels):
    return [f"{ch}_{st}" for ch in channels for st in STATS]


FEATURE_NAMES = feature_names(CHANNELS)
N_CHANNELS = len(CHANNELS)          # 13
N_FEATURES = len(FEATURE_NAMES)     # 65


def extract_features(window, n_channels=None):
    """
    Turn one gesture window into a (5 * n_channels) feature vector.

    window : array of shape (n_samples, n_channels), channels in the
             order of whichever channel list the model was trained on.
    returns: array of shape (5 * n_channels,)

    Note: std uses ddof=0 (population std) to match how the training
    feature matrix was built. Do not switch to ddof=1.
    """
    w = np.asarray(window, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D window, got shape {w.shape}")
    if n_channels is not None and w.shape[1] != n_channels:
        raise ValueError(f"expected (n, {n_channels}) window, got {w.shape}")

    n_ch = w.shape[1]
    out = np.empty(n_ch * len(STATS), dtype=np.float64)
    for i in range(n_ch):
        v = w[:, i]
        base = i * len(STATS)
        vmin = v.min()
        vmax = v.max()
        out[base + 0] = v.mean()
        out[base + 1] = v.std()
        out[base + 2] = vmin
        out[base + 3] = vmax
        out[base + 4] = vmax - vmin
    return out


def fix_length(window, target=WINDOW):
    """
    Pad or trim a window to exactly `target` rows.

    Some repetitions in the dataset are 49 samples instead of 50.
    Deep models need a fixed length, so short windows repeat their
    last row and long windows are trimmed.
    """
    w = np.asarray(window, dtype=np.float32)
    n = w.shape[0]
    if n == target:
        return w
    if n > target:
        return w[:target]
    pad = np.repeat(w[-1:], target - n, axis=0)
    return np.vstack([w, pad])
