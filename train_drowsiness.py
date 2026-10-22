"""
Train the drowsiness CNN from the project's labelled per-frame features.

Run from the repo root:
    python train_drowsiness.py                 # data prep + baseline + CNN
    python train_drowsiness.py --no-cnn        # data prep + baseline only (no TensorFlow needed)

Input : ayush-labelling/labeled_features.csv   (output of extract_features.py + label_clips.py)
Output: models/cnn_drowsiness.keras            (what live_drowsiness.py loads)
        evaluation_results/train_report.json

What this fixes versus the old pipeline
  1. ONE feature list (8 values, same order as live_drowsiness.py) and ONE output path.
  2. Labels come from behavior_label (alert -> 0, everything else -> 1), so a CSV that stores
     the 'drowsy' column as 1/0 or True/False cannot silently flip the classes.
  3. Impossible EAR values (head-tilt tracking failures) are clipped, not left to poison training.
  4. No-face frames become an all-zero vector with face_detected = 0, exactly what the
     live script feeds the model, so training matches deployment.
  5. Leak-free split: inside every clip the frames are cut into contiguous
     train / validation / test blocks with a gap between them (windows never straddle blocks).
  6. Normalisation is built into the model; class weights; light augmentation.
  7. Honest report: accuracy, precision, recall, F1, confusion matrix, per-clip results.

IMPORTANT LIMIT: all clips come from (apparently) one person, so every number below is
"same person, same camera" performance. It does NOT prove accuracy on new drivers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "ayush-labelling" / "labeled_features.csv"
MODEL_PATH = ROOT / "models" / "cnn_drowsiness.keras"
REPORT_PATH = ROOT / "evaluation_results" / "train_report.json"

FEATURES = ["ear", "mar", "head_pitch", "head_yaw", "head_roll",
            "gaze_vertical", "gaze_horizontal", "face_detected"]   # same order as live script
WINDOW = 30
STRIDE_TRAIN = 3
STRIDE_EVAL = 10
GAP = WINDOW                    # frames dropped between blocks
SPLIT = (0.70, 0.15, 0.15)      # train / val / test, per clip, in time order
EAR_MAX = 0.50                  # EAR above this is a landmark failure, not a real eye
SEED = 42


# --------------------------------------------------------------------- data
def load_frames():
    if not CSV_PATH.is_file():
        raise SystemExit(f"Not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df["face_detected"] = df["face_detected"].astype(str).str.lower().eq("true")
    for c in FEATURES[:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["source_clip", "frame"]).reset_index(drop=True)
    return df


def base_labels(clip_df):
    """1 = drowsy/unsafe, 0 = alert. Uses behavior_label (alert -> 0, anything else -> 1);
    falls back to parsing the 'drowsy' column in any common format (True/1/yes)."""
    if "behavior_label" in clip_df.columns and clip_df["behavior_label"].notna().any():
        beh = clip_df["behavior_label"].astype(str).str.strip().str.lower()
        return (~beh.eq("alert")).astype(int).to_numpy()
    raw = clip_df["drowsy"].astype(str).str.strip().str.lower()
    return raw.isin({"true", "1", "1.0", "yes", "y", "drowsy"}).astype(int).to_numpy()


def relabel(clip_df):
    """Frame-level labels: 1 = drowsy/unsafe behaviour, 0 = alert (clip-based, see base_labels)."""
    return base_labels(clip_df)


def frame_matrix(clip_df):
    X = clip_df[FEATURES[:-1]].to_numpy(dtype=np.float32)
    face = clip_df["face_detected"].to_numpy()
    X = np.nan_to_num(X, nan=0.0)
    X[:, 0] = np.clip(X[:, 0], 0.0, EAR_MAX)          # fix impossible EAR
    X = np.column_stack([X, face.astype(np.float32)])
    X[~face] = 0.0                                     # missing face = zero vector, flag 0
    return X


def make_windows(X, y, frames, lo, hi, stride):
    out_x, out_y = [], []
    for s in range(lo, hi - WINDOW + 1, stride):
        if not np.all(np.diff(frames[s:s + WINDOW]) == 1):
            continue
        out_x.append(X[s:s + WINDOW])
        out_y.append(int(y[s:s + WINDOW].mean() >= 0.5))
    return out_x, out_y


def build_splits(df):
    parts = {k: ([], [], []) for k in ("train", "val", "test")}   # X, y, clip
    for clip, g in df.groupby("source_clip", sort=True):
        g = g.reset_index(drop=True)
        X, y, fr = frame_matrix(g), relabel(g), g["frame"].to_numpy()
        n = len(g)
        a = int(n * SPLIT[0])
        b = int(n * (SPLIT[0] + SPLIT[1]))
        blocks = {"train": (0, a, STRIDE_TRAIN),
                  "val": (a + GAP, b, STRIDE_EVAL),
                  "test": (b + GAP, n, STRIDE_EVAL)}
        for name, (lo, hi, stride) in blocks.items():
            wx, wy = make_windows(X, y, fr, lo, hi, stride)
            parts[name][0].extend(wx)
            parts[name][1].extend(wy)
            parts[name][2].extend([clip] * len(wy))
    out = {}
    for name, (wx, wy, wc) in parts.items():
        out[name] = (np.asarray(wx, np.float32), np.asarray(wy, np.int32), np.asarray(wc))
    return out


def augment(X, rng):
    """Light noise + per-sequence gain, only on training data."""
    Xa = X.copy()
    scale = rng.normal(1.0, 0.03, size=(len(X), 1, X.shape[2])).astype(np.float32)
    noise = rng.normal(0.0, 0.01, size=X.shape).astype(np.float32)
    present = X[:, :, 7:8]                      # do not add noise to missing-face frames
    Xa = (Xa * scale + noise * present)
    Xa[:, :, 7] = X[:, :, 7]
    return Xa


# ------------------------------------------------------------------ metrics
def metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"accuracy": round((tp + tn) / max(len(y_true), 1), 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def per_clip(y, pred, clips):
    return {c: {"n": int((clips == c).sum()),
                "accuracy": round(float((pred[clips == c] == y[clips == c]).mean()), 4)}
            for c in np.unique(clips)}


def summarise(split, name):
    X, y, c = split
    print(f"  {name:5s}: {len(y):4d} windows  alert={int((y == 0).sum())}  drowsy={int((y == 1).sum())}")


# ----------------------------------------------------------------- baseline
def run_baseline(data):
    """Simple logistic regression on window statistics: a sanity check for the CNN."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def stats(X):
        return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)

    Xtr, ytr, _ = data["train"]
    sc = StandardScaler().fit(stats(Xtr))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(stats(Xtr)), ytr)
    res = {}
    for name in ("val", "test"):
        X, y, c = data[name]
        pred = clf.predict(sc.transform(stats(X)))
        res[name] = metrics(y, pred)
    return res


# --------------------------------------------------------------------- CNN
def run_cnn(data, epochs):
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(SEED)
    Xtr, ytr, _ = data["train"]
    rng = np.random.default_rng(SEED)

    norm = keras.layers.Normalization(axis=-1)
    norm.adapt(Xtr.reshape(-1, Xtr.shape[2]))

    model = keras.Sequential([
        keras.Input(shape=(WINDOW, len(FEATURES))),
        norm,
        keras.layers.Conv1D(32, 5, padding="same", activation="relu"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=["accuracy"])

    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    class_weight = {0: len(ytr) / (2 * max(n_neg, 1)), 1: len(ytr) / (2 * max(n_pos, 1))}
    Xva, yva, _ = data["val"]

    best_val, wait, best_weights = float("inf"), 0, None
    for epoch in range(epochs):                       # fresh augmentation every epoch
        model.fit(augment(Xtr, rng), ytr, batch_size=32, epochs=1, verbose=0,
                  class_weight=class_weight, shuffle=True)
        val_loss = float(model.evaluate(Xva, yva, verbose=0)[0])
        if val_loss < best_val - 1e-4:
            best_val, wait, best_weights = val_loss, 0, model.get_weights()
        else:
            wait += 1
            if wait >= 10:
                break
    if best_weights is not None:
        model.set_weights(best_weights)
    print(f"CNN trained for {epoch + 1} epochs (best val loss {best_val:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved {MODEL_PATH}")

    res = {}
    for name in ("val", "test"):
        X, y, c = data[name]
        p = model.predict(X, verbose=0).reshape(-1)
        pred = (p >= 0.5).astype(int)
        res[name] = metrics(y, pred)
        res[name]["per_clip"] = per_clip(y, pred, c)
    return res


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-cnn", action="store_true")
    args = ap.parse_args()

    df = load_frames()
    print("Raw 'drowsy' values in CSV:", sorted(df["drowsy"].astype(str).unique()))
    print("Raw 'behavior_label' values:", sorted(df["behavior_label"].astype(str).unique()))
    print("Frame-level labels after relabelling (per clip):")
    total_pos = total = 0
    for clip, g in df.groupby("source_clip", sort=True):
        y = relabel(g.reset_index(drop=True))
        total_pos += int(y.sum()); total += len(y)
        print(f"  {clip:22s} frames={len(y):4d}  drowsy={int(y.sum()):4d}  alert={int((y == 0).sum()):4d}")
    if total_pos / max(total, 1) < 0.30:
        raise SystemExit("STOP: fewer than 30% of frames are labelled drowsy. Your labels look wrong "
                         "(expected roughly 70% drowsy on the project data). Check the CSV before training.")
    data = build_splits(df)
    print("Windows (30 frames each):")
    for k in ("train", "val", "test"):
        summarise(data[k], k)
    for k in ("train", "val", "test"):
        if len(np.unique(data[k][1])) < 2:
            print(f"WARNING: {k} split has only one class; its metrics are not meaningful.")

    report = {"note": "single-person data; same-person performance only",
              "baseline_logreg": run_baseline(data)}
    print("\nBaseline (logistic regression):")
    for k, v in report["baseline_logreg"].items():
        print(f"  {k}: {v}")

    if not args.no_cnn:
        report["cnn"] = run_cnn(data, args.epochs)
        print("\nCNN:")
        for k, v in report["cnn"].items():
            print(f"  {k}: {v}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()