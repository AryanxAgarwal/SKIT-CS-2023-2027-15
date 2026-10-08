"""
Trains the alert / not-alert temporal classifier on the REAL windowed
dataset (dataset.npz, built by build_dataset.py) and exports it to TFLite.

Design choices worth knowing:
  - Honest validation: windows overlap (30-frame windows, stride 10), so a
    random train/val split would leak near-identical frames into both sets
    and report fake-high accuracy. Instead, each clip is split
    chronologically (first 75% train, last 20% val) with a gap of a few
    windows in between so no frames are shared across the split.
  - Normalization is built INTO the model (Normalization layer adapted on
    training data), so the exported .tflite takes raw feature values --
    no separate scaling step to remember at inference time.
  - Classes are imbalanced (more not-alert than alert), so class weights
    are applied.

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 40
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras


def chronological_split(clip_source, train_frac=0.75, gap=3):
    """Per-clip split by time order. Returns boolean train/val masks."""
    clip_source = np.asarray(clip_source)
    train_mask = np.zeros(len(clip_source), dtype=bool)
    val_mask = np.zeros(len(clip_source), dtype=bool)
    for clip in np.unique(clip_source):
        idx = np.where(clip_source == clip)[0]  # already in temporal order
        n = len(idx)
        n_train = int(n * train_frac)
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train + gap:]] = True
    return train_mask, val_mask


def build_model(window, n_features, normalizer):
    model = keras.Sequential([
        keras.Input(shape=(window, n_features)),
        normalizer,
        keras.layers.Conv1D(16, 5, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(32, 5, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def export_tflite(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    blob = converter.convert()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(blob)
    print(f"Exported {out_path} ({len(blob)/1024:.1f} KB)")
    return blob


def tflite_predict(blob, X):
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    preds = []
    for i in range(len(X)):
        interp.set_tensor(inp["index"], X[i:i + 1].astype(np.float32))
        interp.invoke()
        preds.append(float(interp.get_tensor(out["index"]).squeeze()))
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset.npz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", default="model/ddd_classifier.tflite")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)

    d = np.load(args.data, allow_pickle=True)
    X, y, clips = d["X"], d["y"].astype(np.float32), d["clip_source"]
    print(f"Loaded {X.shape[0]} windows, shape per window {X.shape[1:]}, "
          f"features: {list(d['features'])}")

    train_mask, val_mask = chronological_split(clips)
    X_tr, y_tr, X_va, y_va = X[train_mask], y[train_mask], X[val_mask], y[val_mask]
    print(f"Train windows: {len(X_tr)} (alert={int((y_tr==0).sum())}, not_alert={int((y_tr==1).sum())})")
    print(f"Val windows:   {len(X_va)} (alert={int((y_va==0).sum())}, not_alert={int((y_va==1).sum())})")

    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_tr.reshape(-1, X.shape[2]))

    model = build_model(X.shape[1], X.shape[2], normalizer)

    n_pos, n_neg = (y_tr == 1).sum(), (y_tr == 0).sum()
    class_weight = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}

    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epochs,
              batch_size=16, class_weight=class_weight, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=8, restore_best_weights=True)])

    blob = export_tflite(model, args.out)

    # Evaluate the EXPORTED tflite model (what actually runs on the device)
    p = tflite_predict(blob, X_va)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y_va).mean()
    alert_recall = ((pred == 0) & (y_va == 0)).sum() / max((y_va == 0).sum(), 1)
    notalert_recall = ((pred == 1) & (y_va == 1)).sum() / max((y_va == 1).sum(), 1)
    print(f"\nHeld-out validation (TFLite model, chronological split):")
    print(f"  accuracy:            {acc:.3f}")
    print(f"  alert recall:        {alert_recall:.3f}  (alert windows correctly left alone)")
    print(f"  not-alert recall:    {notalert_recall:.3f}  (not-alert windows correctly caught)")

    val_clips = np.asarray(clips)[val_mask]
    print("  per-clip accuracy on held-out windows:")
    for c in np.unique(val_clips):
        m = val_clips == c
        print(f"    {c:22s} n={int(m.sum()):3d}  acc={(pred[m] == y_va[m]).mean():.3f}")


if __name__ == "__main__":
    main()
