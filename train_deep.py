"""
train_deep.py — trains a 1D CNN and a BiLSTM on the raw 50x13 sequences.

These two models cannot use rep_features.csv. That file has already
collapsed each repetition into summary statistics, which throws away the
time ordering the deep models exist to exploit. So this script reads
dataset.csv directly and rebuilds the sequences.

It reuses the exact train/test split written by train_classical.py, so the
six models in the benchmark are compared on identical data.

Usage:
    python train_classical.py --outdir results     # run this FIRST
    python train_deep.py --dataset dataset.csv --outdir results

Produces, in --outdir:
    metrics_deep.csv           accuracy / macro-F1 / latency per model
    per_class_<model>.csv      precision / recall / F1 per gesture
    confusion_<model>.png      confusion matrix on the held-out test set
    history_<model>.png        training and validation curves
    model_<model>.keras        trained network
    scaler_deep.joblib         per-channel scaler, needed at inference time
    metrics_all.csv            all six models merged, for the report table
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
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from features import CHANNELS, WINDOW, fix_length

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
EPOCHS = 150
BATCH_SIZE = 32
VAL_SIZE = 0.15   # carved out of the training set, for early stopping

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def build_sequences(dataset_path, split_keys_path):
    """Rebuild (n, 50, 11) sequences and align them to the saved split."""
    df = pd.read_csv(dataset_path)
    splits = pd.read_csv(split_keys_path)
    wanted = {(r.label, r.rep): r.split for r in splits.itertuples()}

    X, y, assign = [], [], []
    for (lab, rep), g in df.groupby(["label", "rep"], sort=True):
        key = (lab, rep)
        if key not in wanted:
            continue  # dropped outlier repetition
        g = g.sort_values("sample")
        X.append(fix_length(g[CHANNELS].values, WINDOW))
        y.append(lab)
        assign.append(wanted[key])

    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    assign = np.array(assign)
    print(f"built {X.shape[0]} sequences of shape {X.shape[1:]}")
    if X.shape[0] != len(splits):
        print(f"WARNING: {len(splits)} rows in split file but "
              f"{X.shape[0]} sequences matched")
    return X, y, assign


def scale_sequences(X_tr, X_te, X_val=None):
    """
    Standardise each channel using training-set statistics only.

    Fitting on the test set would leak information and inflate the result.
    """
    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0)
    sd[sd < 1e-8] = 1.0

    def apply(a):
        return None if a is None else ((a - mu) / sd).astype(np.float32)

    return apply(X_tr), apply(X_te), apply(X_val), (mu, sd)


def build_cnn(n_classes):
    m = keras.Sequential([
        keras.Input(shape=(WINDOW, len(CHANNELS))),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="cnn1d")
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def build_bilstm(n_classes):
    # Mirrors the Kanwal et al. architecture: two stacked bidirectional
    # LSTM layers with 10% dropout, then a tanh dense layer.
    m = keras.Sequential([
        keras.Input(shape=(WINDOW, len(CHANNELS))),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.1),
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dropout(0.1),
        layers.Dense(128, activation="tanh"),
        layers.Dense(n_classes, activation="softmax"),
    ], name="bilstm")
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def plot_history(hist, name, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(hist.history["accuracy"], label="train")
    ax[0].plot(hist.history["val_accuracy"], label="validation")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy")
    ax[0].legend(frameon=False)
    ax[1].plot(hist.history["loss"], label="train")
    ax[1].plot(hist.history["val_loss"], label="validation")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss")
    ax[1].legend(frameon=False)
    fig.suptitle(f"{name} training history", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, classes, title, path):
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cmn[i, j] > 0:
                ax.text(j, i, f"{cmn[i, j]:.2f}".lstrip("0"),
                        ha="center", va="center", fontsize=5,
                        color="white" if cmn[i, j] > 0.6 else "#b03a2e")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def measure_latency(model, X, n=100):
    sample = X[:1]
    model.predict(sample, verbose=0)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(sample, verbose=0)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--split", default=None,
                    help="path to split_keys.csv (found automatically if omitted)")
    args = ap.parse_args()

    out = Path(args.outdir)

    # train_classical.py writes split_keys.csv into the results folder; a
    # copy may also sit next to the dataset. Accept either, so the file does
    # not have to be copied around by hand.
    candidates = [Path(args.split)] if args.split else [
        out / "split_keys.csv",
        Path(args.dataset).parent / "split_keys.csv",
        Path("split_keys.csv"),
    ]
    split_file = next((p for p in candidates if p.exists()), None)
    if split_file is None:
        raise SystemExit(
            "split_keys.csv not found. Looked in: "
            + ", ".join(str(p) for p in candidates)
            + "\nRun train_classical.py first."
        )
    print(f"using split from {split_file}")

    X, y_raw, assign = build_sequences(args.dataset, split_file)

    le = joblib.load(out / "label_encoder.joblib") \
        if (out / "label_encoder.joblib").exists() else LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    classes = list(le.classes_)

    tr, te = assign == "train", assign == "test"
    X_tr_all, y_tr_all = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_all, y_tr_all, test_size=VAL_SIZE,
        stratify=y_tr_all, random_state=RANDOM_STATE)

    X_tr, X_te_s, X_val, (mu, sd) = scale_sequences(X_tr, X_te, X_val)
    joblib.dump({"mean": mu, "std": sd, "channels": CHANNELS},
                out / "scaler_deep.joblib")
    print(f"train {len(X_tr)}  val {len(X_val)}  test {len(X_te_s)}  "
          f"classes {len(classes)}")

    builders = {"cnn1d": build_cnn, "bilstm": build_bilstm}
    rows = []

    for name, build in builders.items():
        print(f"\n=== {name} ===")
        model = build(len(classes))
        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=7, min_lr=1e-5),
        ]
        t0 = time.perf_counter()
        hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                         epochs=args.epochs, batch_size=BATCH_SIZE,
                         callbacks=cbs, verbose=2)
        train_s = time.perf_counter() - t0

        pred = model.predict(X_te_s, verbose=0).argmax(axis=1)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        lat = measure_latency(model, X_te_s)

        rows.append({
            "model": name,
            "cv_mean": np.nan,   # deep models use a validation split, not CV
            "cv_std": np.nan,
            "test_accuracy": acc,
            "test_macro_f1": f1,
            "train_seconds": round(train_s, 2),
            "latency_ms": round(lat, 3),
            "epochs_run": len(hist.history["loss"]),
        })
        print(f"{name}: test={acc:.4f}  f1={f1:.4f}  latency={lat:.2f}ms  "
              f"epochs={len(hist.history['loss'])}")

        rep = classification_report(y_te, pred, labels=range(len(classes)),
                                    target_names=classes, output_dict=True,
                                    zero_division=0)
        pd.DataFrame(rep).T.to_csv(out / f"per_class_{name}.csv")
        cm = confusion_matrix(y_te, pred, labels=range(len(classes)))
        plot_confusion(cm, classes, f"{name} — test set confusion matrix",
                       out / f"confusion_{name}.png")
        plot_history(hist, name, out / f"history_{name}.png")
        model.save(out / f"model_{name}.keras")

    deep = pd.DataFrame(rows)
    deep.to_csv(out / "metrics_deep.csv", index=False)

    classical_file = out / "metrics_classical.csv"
    if classical_file.exists():
        allm = pd.concat([pd.read_csv(classical_file), deep],
                         ignore_index=True)
        allm = allm.sort_values("test_accuracy", ascending=False)
        allm.to_csv(out / "metrics_all.csv", index=False)
        print("\n=== full benchmark ===")
        print(allm.to_string(index=False))


if __name__ == "__main__":
    main()