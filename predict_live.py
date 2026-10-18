"""
predict_live.py — load a trained model and classify one gesture window.

This exists to guarantee that live inference builds features exactly the
way training did. If you ever hand-roll the feature vector somewhere else,
the channel order will drift and accuracy will collapse for reasons that
are very hard to debug.

Use it like this in the real-time loop:

    from predict_live import GesturePredictor

    predictor = GesturePredictor("results", model="logreg")

    window = collect_50_samples_from_esp32()   # (50, 11) in CHANNELS order
    label, confidence = predictor.predict(window)
    if confidence >= 0.85:
        handle(label)
"""

from pathlib import Path

import joblib
import numpy as np

from features import CHANNELS, WINDOW, extract_features, fix_length


class GesturePredictor:
    def __init__(self, results_dir="results", model="logreg",
                 threshold=0.85):
        d = Path(results_dir)
        self.threshold = threshold
        self.name = model

        # Channel count comes from the model, so an 11-channel model from
        # Dataset 3 still runs on the 14-channel firmware (extra channels
        # are ignored) and a 14-channel model uses all of them.
        if model in ("cnn1d", "bilstm"):
            from tensorflow import keras
            self.kind = "deep"
            self.model = keras.models.load_model(d / f"model_{model}.keras")
            sc = joblib.load(d / "scaler_deep.joblib")
            self.mu, self.sd = sc["mean"], sc["std"]
            self.channels = list(sc["channels"])
            if self.channels != CHANNELS[:len(self.channels)]:
                raise ValueError("channel order mismatch between "
                                 "scaler and features.py")
        else:
            self.kind = "classical"
            self.model = joblib.load(d / f"model_{model}.joblib")
            clf = getattr(self.model, "named_steps", {}).get("clf")
            if hasattr(clf, "n_jobs"):
                clf.n_jobs = 1   # single-window prediction, no threading
            n_feat = int(self.model.n_features_in_)
            self.channels = list(CHANNELS[:n_feat // 5])

        self.le = joblib.load(d / "label_encoder.joblib")

    def predict(self, window):
        """
        window : (n, 11) array of raw sensor readings, already calibrated
                 the same way the training data was, in CHANNELS order.

        returns (label, confidence). Confidence is a probability, so it is
        directly comparable against self.threshold.
        """
        w = fix_length(window, WINDOW)[:, :len(self.channels)]

        if self.kind == "deep":
            x = ((w - self.mu) / self.sd)[None, ...].astype(np.float32)
            probs = self.model.predict(x, verbose=0)[0]
        else:
            x = extract_features(w)[None, :]
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(x)[0]
            else:
                idx = int(self.model.predict(x)[0])
                return self.le.inverse_transform([idx])[0], 1.0

        idx = int(np.argmax(probs))
        return self.le.inverse_transform([idx])[0], float(probs[idx])

    def predict_or_none(self, window):
        """Returns None when the model is not confident enough."""
        label, conf = self.predict(window)
        return (label, conf) if conf >= self.threshold else (None, conf)


if __name__ == "__main__":
    # Smoke test using a real repetition from the dataset.
    import pandas as pd
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "logreg"
    df = pd.read_csv("dataset.csv")
    p = GesturePredictor("results", model=model)

    correct = 0
    trials = list(df.groupby(["label", "rep"]))[:40]
    for (lab, rep), g in trials:
        w = g.sort_values("sample")[CHANNELS].values
        pred, conf = p.predict(w)
        correct += (pred == lab)
        print(f"true={lab:8s} pred={pred:8s} conf={conf:.3f}")
    print(f"\n{correct}/{len(trials)} correct on this sample")
