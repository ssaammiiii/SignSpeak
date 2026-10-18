"""
train_words.py — train the word-sign classifier on the merged two-glove data.

Reads words.csv from record_words.py (label, rep, sample, 22 channels).
Each repetition is WORD_RECORD_SAMPLES rows (1.5 s); the model consumes
WORD_WINDOW rows (1 s). Live, the window slides over the sign, so the
training set is built from several random 1 s crops of every repetition
rather than one fixed crop. Test repetitions are scored on their centre
crop only.

Models (all on one repetition-level split, seed 42):
    lda, logreg, random_forest   on 110 statistical features (features.py)
    bilstm                       on the raw 50 x 22 sequence (needs tensorflow)

Usage:
    python train_words.py                       # words.csv -> results_words/
    python train_words.py --no-deep             # skip the BiLSTM
    python train_words.py --crops 8

Produces in results_words/:
    metrics_words.csv, per_class_<model>.csv, confusion_<model>.png,
    model_<model>.joblib / model_bilstm.keras, scaler_deep.joblib,
    label_encoder.joblib, channels.joblib, split_keys.csv
"""

import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features import (MERGED_CHANNELS, WORD_CHANNELS, WORD_WINDOW,
                      extract_features, fix_length)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_sequences(path, channels):
    df = pd.read_csv(path)
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")

    # A channel that never varies anywhere carries no information and gives
    # every model trained on statistics a set of zero-variance features.
    # That is what broke the letter LDA; drop them here instead.
    dead = [c for c in channels if float(df[c].std()) < 1e-9]
    if dead:
        print(f"dropping {len(dead)} channel(s) that never move: "
              f"{' '.join(dead)}")
        channels = [c for c in channels if c not in dead]

    seqs, labels, keys = [], [], []
    for (lab, rep), g in df.groupby(["label", "rep"], sort=True):
        g = g.sort_values("sample")
        seqs.append(g[channels].values.astype(np.float32))
        labels.append(str(lab).upper())
        keys.append((lab, rep))
    return seqs, np.array(labels), keys, channels


def crops(seq, n_crops, rng, target=WORD_WINDOW):
    """Random 1 s crops of one repetition (plus the centre crop)."""
    n = seq.shape[0]
    if n <= target:
        return [fix_length(seq, target)]
    out = [seq[(n - target) // 2:(n - target) // 2 + target]]
    for _ in range(n_crops):
        s = int(rng.integers(0, n - target + 1))
        out.append(seq[s:s + target])
    return out


def centre(seq, target=WORD_WINDOW):
    n = seq.shape[0]
    if n <= target:
        return fix_length(seq, target)
    s = (n - target) // 2
    return seq[s:s + target]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def plot_confusion(cm, classes, title, path):
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.6 else "#333333")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def latency_ms(fn, x, n=100):
    fn(x)
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(x); t.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(t))


def report(name, y_te, pred, classes, out, lat, train_s, rows, extra=None):
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    row = {"model": name, "test_accuracy": acc, "test_macro_f1": f1,
           "train_seconds": round(train_s, 2), "latency_ms": round(lat, 3)}
    if extra:
        row.update(extra)
    rows.append(row)
    print(f"{name:14s} test={acc:.4f}  f1={f1:.4f}  latency={lat:.2f}ms")
    rep = classification_report(y_te, pred, labels=range(len(classes)),
                                target_names=classes, output_dict=True,
                                zero_division=0)
    pd.DataFrame(rep).T.to_csv(out / f"per_class_{name}.csv")
    cm = confusion_matrix(y_te, pred, labels=range(len(classes)))
    plot_confusion(cm, classes, f"{name} — word signs, test set",
                   out / f"confusion_{name}.png")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="words.csv")
    ap.add_argument("--outdir", default="results_words")
    ap.add_argument("--crops", type=int, default=6,
                    help="random 1 s crops per training repetition")
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--channels", choices=["word", "merged"], default="word",
                    help="'word' = flex+IMU from both gloves (22); "
                         "'merged' = also the right glove's contact pads (24)")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    wanted = WORD_CHANNELS if args.channels == "word" else MERGED_CHANNELS
    seqs, y_raw, keys, channels = load_sequences(args.dataset, wanted)
    N_CH = len(channels)
    print(f"{N_CH} channels")
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    classes = list(le.classes_)
    print(f"{len(seqs)} repetitions, {len(classes)} classes: {' '.join(classes)}")
    counts = pd.Series(y_raw).value_counts()
    if counts.min() < 10:
        print(f"WARNING: fewest reps for a class is {counts.min()} - "
              "results will be noisy. Aim for 45 per class.")

    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                                      random_state=RANDOM_STATE)
    sk = pd.DataFrame(keys, columns=["label", "rep"])
    sk["split"] = "train"; sk.loc[idx_te, "split"] = "test"
    sk.to_csv(out / "split_keys.csv", index=False)

    # Training windows: several crops per rep. Test: centre crop.
    X_tr_seq, y_tr = [], []
    for i in idx_tr:
        for w in crops(seqs[i], args.crops, rng):
            X_tr_seq.append(w); y_tr.append(y[i])
    X_te_seq = [centre(seqs[i]) for i in idx_te]
    y_te = y[idx_te]
    X_tr_seq = np.stack(X_tr_seq); X_te_seq = np.stack(X_te_seq)
    y_tr = np.array(y_tr)
    print(f"train windows {len(X_tr_seq)} (from {len(idx_tr)} reps)  "
          f"test reps {len(idx_te)}")

    joblib.dump(le, out / "label_encoder.joblib")
    joblib.dump(list(channels), out / "channels.joblib")
    rows = []

    # ---- classical ------------------------------------------------------
    F_tr = np.stack([extract_features(w) for w in X_tr_seq])
    F_te = np.stack([extract_features(w) for w in X_te_seq])
    models = {
        "lda": Pipeline([("scale", StandardScaler()),
                         ("clf", LinearDiscriminantAnalysis(solver="lsqr",
                                                            shrinkage="auto"))]),
        "logreg": Pipeline([("scale", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=5000, C=1.0))]),
        "random_forest": Pipeline([("clf", RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))]),
    }
    for name, model in models.items():
        t0 = time.perf_counter(); model.fit(F_tr, y_tr); train_s = time.perf_counter() - t0
        clf = model.named_steps.get("clf")
        if hasattr(clf, "n_jobs"):
            clf.n_jobs = 1
        pred = model.predict(F_te)
        lat = latency_ms(model.predict, F_te[:1])
        report(name, y_te, pred, classes, out, lat, train_s, rows)
        joblib.dump(model, out / f"model_{name}.joblib")

    # ---- BiLSTM ---------------------------------------------------------------
    if not args.no_deep:
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            print("\ntensorflow not installed - skipping bilstm (use --no-deep to silence)")
        else:
            tf.random.set_seed(RANDOM_STATE)
            mu = X_tr_seq.reshape(-1, N_CH).mean(axis=0)
            sd = X_tr_seq.reshape(-1, N_CH).std(axis=0); sd[sd < 1e-8] = 1.0
            joblib.dump({"mean": mu, "std": sd, "channels": list(channels)},
                        out / "scaler_deep.joblib")
            Z_tr = ((X_tr_seq - mu) / sd).astype(np.float32)
            Z_te = ((X_te_seq - mu) / sd).astype(np.float32)
            Z_tr, Z_val, yz_tr, yz_val = train_test_split(
                Z_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=RANDOM_STATE)

            # Smaller than the letter BiLSTM: a few hundred windows, 4 classes.
            model = keras.Sequential([
                keras.Input(shape=(WORD_WINDOW, N_CH)),
                layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
                layers.Dropout(0.2),
                layers.Bidirectional(layers.LSTM(64)),
                layers.Dropout(0.2),
                layers.Dense(64, activation="tanh"),
                layers.Dense(len(classes), activation="softmax"),
            ], name="bilstm_words")
            model.compile(optimizer=keras.optimizers.Adam(1e-3),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                 restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                     patience=7, min_lr=1e-5)]
            t0 = time.perf_counter()
            hist = model.fit(Z_tr, yz_tr, validation_data=(Z_val, yz_val),
                             epochs=args.epochs, batch_size=16, callbacks=cbs,
                             verbose=2)
            train_s = time.perf_counter() - t0
            pred = np.asarray(model(Z_te, training=False)).argmax(axis=1)
            lat = latency_ms(lambda x: model(x, training=False), Z_te[:1])
            report("bilstm", y_te, pred, classes, out, lat, train_s, rows,
                   extra={"epochs_run": len(hist.history["loss"])})
            model.save(out / "model_bilstm.keras")

    m = pd.DataFrame(rows).sort_values(["test_accuracy", "latency_ms"],
                                       ascending=[False, True])
    m.to_csv(out / "metrics_words.csv", index=False)
    print(f"\nresults written to {out.resolve()}")
    print(m.to_string(index=False))
    best = m.iloc[0]["model"]
    print(f"\nuse it with:  uv run python -m signspeak.main --port COM7 "
          f"--left-port COM8 --model lda --word-model {best}")


if __name__ == "__main__":
    main()
