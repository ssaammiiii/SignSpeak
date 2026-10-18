"""
train_classical.py: trains four classifiers (LDA, Logistic Regression,
SVM-RBF, Random Forest) on the per-repetition feature matrix.

Usage:
    python train_classical.py --features rep_features.csv --outdir results

Produces, in --outdir:
    metrics_classical.csv      accuracy / macro-F1 / latency for each model
    cv_scores_classical.csv    per-fold cross-validation scores
    per_class_<model>.csv      precision / recall / F1 per gesture
    confusion_<model>.png      confusion matrix on the held-out test set
    model_<model>.joblib       trained pipeline, ready for live inference
    label_encoder.joblib       label <-> integer mapping
    split_keys.csv             repetition-level train/test split, reused by
                               train_deep.py
    feature_importance.csv     Random Forest feature importances
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
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# Bad repetitions are removed from the dataset before features are built,
# so nothing is dropped here. DROP_REPS remains as a hook for excluding a
# specific (label, rep) pair.
DROP_REPS = []


def load_data(path, drop_outliers=True):
    df = pd.read_csv(path)

    if drop_outliers:
        before = len(df)
        for lab, rep in DROP_REPS:
            df = df[~((df.label == lab) & (df.rep == rep))]
        print(f"dropped {before - len(df)} flagged repetitions")

    y_raw = df["label"].values
    keys = df[["label", "rep"]].reset_index(drop=True)
    X = df.drop(columns=["label", "rep"])
    feature_names = list(X.columns)
    return X.values.astype(np.float64), y_raw, feature_names, keys


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
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def measure_latency(model, X, n=200):
    """Median single-sample prediction time in milliseconds."""
    sample = X[:1]
    model.predict(sample)  # warm up
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="rep_features.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--keep-outliers", action="store_true")
    ap.add_argument("--split", default=None,
                    help="reuse an existing split_keys.csv "
                         "instead of making a new split")
    ap.add_argument("--tune", action="store_true",
                    help="grid-search SVM and Random Forest (slower)")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    X, y_raw, feature_names, keys = load_data(
        args.features, drop_outliers=not args.keep_outliers)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    print(f"{X.shape[0]} repetitions, {X.shape[1]} features, "
          f"{len(classes)} classes")

    # One split, reused by every model and by train_deep.py, so the
    # benchmark comparison is fair.
    idx = np.arange(len(y))
    split_file = Path(args.split) if args.split else (out / "split_keys.csv")
    if split_file.exists():
        # Reuse the existing split, so every model in the benchmark
        # (including the deep ones) sees identical data.
        prev = pd.read_csv(split_file)
        lookup = {(r.label, r.rep): r.split for r in prev.itertuples()}
        assign = np.array([lookup.get((l, r), "train")
                           for l, r in zip(keys.label, keys.rep)])
        idx_tr = idx[assign == "train"]
        idx_te = idx[assign == "test"]
        print(f"reusing split from {split_file}")
    else:
        idx_tr, idx_te = train_test_split(
            idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    if not split_file.exists():
        split_keys = keys.copy()
        split_keys["split"] = "train"
        split_keys.loc[idx_te, "split"] = "test"
        split_keys.to_csv(out / "split_keys.csv", index=False)
    X_tr, X_te, y_tr, y_te = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
    print(f"train {len(idx_tr)}  test {len(idx_te)}")

    models = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=1.0)),
        ]),
        # solver="lsqr" with shrinkage, NOT the default "svd". With 65
        # features and about 36 repetitions per class the pooled
        # within-class covariance is singular: any feature that barely
        # varies inside a class (a contact pad that reads the same for every
        # repetition of a letter, for example) has near-zero variance, and
        # the svd solver collapses. Measured on this split (1,016 train /
        # 254 test): default svd 10.63%, lsqr with Ledoit-Wolf shrinkage
        # 99.21%.
        "lda": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr",
                                               shrinkage="auto")),
        ]),
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=10, gamma="scale", kernel="rbf", probability=True,
                        random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=400,
                                           random_state=RANDOM_STATE,
                                           n_jobs=-1)),
        ]),
    }

    if args.tune:
        grids = {
            "svm_rbf": {"clf__C": [1, 10, 100],
                        "clf__gamma": ["scale", 0.01, 0.05]},
            "random_forest": {"clf__n_estimators": [200, 400, 800],
                              "clf__max_depth": [None, 20, 40]},
        }
        for name, grid in grids.items():
            gs = GridSearchCV(models[name], grid,
                              cv=StratifiedKFold(3, shuffle=True,
                                                 random_state=RANDOM_STATE),
                              n_jobs=-1, scoring="accuracy")
            gs.fit(X_tr, y_tr)
            print(f"{name} best params: {gs.best_params_}  "
                  f"cv={gs.best_score_:.4f}")
            models[name] = gs.best_estimator_

    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, cv_rows = [], []

    for name, model in models.items():
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, n_jobs=-1)
        for i, s in enumerate(scores):
            cv_rows.append({"model": name, "fold": i + 1, "accuracy": s})

        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_s = time.perf_counter() - t0

        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")

        # Live inference predicts one window at a time. Multi-threading
        # only adds overhead at that size, so measure (and ship) with
        # n_jobs=1 or the reported latency will be misleadingly high.
        clf = model.named_steps.get("clf")
        if hasattr(clf, "n_jobs"):
            clf.n_jobs = 1
        lat = measure_latency(model, X_te)

        rows.append({
            "model": name,
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "test_accuracy": acc,
            "test_macro_f1": f1,
            "train_seconds": round(train_s, 2),
            "latency_ms": round(lat, 3),
        })
        print(f"{name:15s} cv={scores.mean():.4f}±{scores.std():.4f}  "
              f"test={acc:.4f}  f1={f1:.4f}  latency={lat:.2f}ms")

        rep = classification_report(y_te, pred, labels=range(len(classes)),
                                    target_names=classes, output_dict=True,
                                    zero_division=0)
        pd.DataFrame(rep).T.to_csv(out / f"per_class_{name}.csv")

        cm = confusion_matrix(y_te, pred, labels=range(len(classes)))
        plot_confusion(cm, classes, f"{name} — test set confusion matrix",
                       out / f"confusion_{name}.png")

        joblib.dump(model, out / f"model_{name}.joblib")

    joblib.dump(le, out / "label_encoder.joblib")
    joblib.dump(feature_names, out / "feature_names.joblib")

    m = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    m.to_csv(out / "metrics_classical.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out / "cv_scores_classical.csv", index=False)

    # Random Forest exposes feature importances, useful for the report.
    rf = models["random_forest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=feature_names)
    imp.sort_values(ascending=False).to_csv(out / "feature_importance.csv")

    print(f"\nresults written to {out.resolve()}")
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()