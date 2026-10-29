"""
User story: Deep Learning Model Training & Loss Optimization
    "Execute model training loops using Binary Cross-Entropy loss and the Adam optimizer."

Run from the repo root (next to train_drowsiness.py):
    python train_model.py                       # defaults: lr 1e-3, batch 32, dropout 0.4
    python train_model.py --lr 3e-4 --epochs 80
    python train_model.py --use-best            # use the result of tune_hyperparameters.py

What it does
  * Builds the 1D-CNN (normalisation layer inside the model).
  * Runs an EXPLICIT training loop (tf.GradientTape):
        loss = BinaryCrossentropy(y, p, sample_weight=class_weight) + L2 regularisation
        optimizer = Adam(learning_rate)
  * Fresh augmentation every epoch, early stopping on validation loss, best weights restored.
  * Saves models/cnn_drowsiness.keras (the file live_drowsiness.py loads),
    evaluation_results/training_history.csv (loss/accuracy per epoch) and
    evaluation_results/train_model_report.json (val + test metrics).

Data and splits come from train_drowsiness.py (same leak-free time-block split).
NOTE: all clips are from one person, so scores are same-person performance only.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_drowsiness import (FEATURES, MODEL_PATH, SEED, WINDOW, augment,
                              build_splits, load_frames, metrics, per_clip)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "evaluation_results"
BEST_HPARAMS_PATH = RESULTS_DIR / "best_hparams.json"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"
REPORT_PATH = RESULTS_DIR / "train_model_report.json"

DEFAULTS = {"learning_rate": 1e-3, "batch_size": 32, "dropout": 0.4}


def build_model(x_train, dropout=0.4):
    """1D-CNN over a 30-frame window of 8 features; sigmoid output = P(drowsy)."""
    from tensorflow import keras

    norm = keras.layers.Normalization(axis=-1)
    norm.adapt(x_train.reshape(-1, x_train.shape[2]))
    return keras.Sequential([
        keras.Input(shape=(WINDOW, len(FEATURES))),
        norm,
        keras.layers.Conv1D(32, 5, padding="same", activation="relu"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


def fit(model, data, learning_rate, batch_size, epochs=80, patience=10,
        seed=SEED, verbose=True):
    """Custom training loop: Binary Cross-Entropy + Adam. Returns a result dict."""
    import tensorflow as tf
    from tensorflow import keras

    rng = np.random.default_rng(seed)
    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["val"]

    loss_fn = keras.losses.BinaryCrossentropy()                       # <- BCE
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)    # <- Adam

    # Class weights so the minority class is not ignored.
    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    w_pos = len(y_train) / (2 * max(n_pos, 1))
    w_neg = len(y_train) / (2 * max(n_neg, 1))
    sample_w = np.where(y_train == 1, w_pos, w_neg).astype(np.float32)

    @tf.function(input_signature=[
        tf.TensorSpec([None, WINDOW, len(FEATURES)], tf.float32),
        tf.TensorSpec([None, 1], tf.float32),
        tf.TensorSpec([None], tf.float32),
    ])
    def train_step(x, y, w):
        with tf.GradientTape() as tape:
            p = model(x, training=True)
            loss = loss_fn(y, p, sample_weight=w)
            if model.losses:                                          # L2 penalties
                loss += tf.add_n(model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def evaluate(x, y):
        p = model(x, training=False).numpy().reshape(-1)
        loss = float(loss_fn(y.reshape(-1, 1).astype(np.float32),
                             p.reshape(-1, 1)).numpy())
        acc = float(((p >= 0.5).astype(int) == y).mean())
        return loss, acc

    history, best_loss, best_acc, best_weights, wait = [], float("inf"), 0.0, None, 0
    for epoch in range(1, epochs + 1):
        x_aug = augment(x_train, rng)
        order = rng.permutation(len(x_aug))
        batch_losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            loss = train_step(tf.constant(x_aug[idx]),
                              tf.constant(y_train[idx].reshape(-1, 1).astype(np.float32)),
                              tf.constant(sample_w[idx]))
            batch_losses.append(float(loss))
        tr_loss = float(np.mean(batch_losses))
        val_loss, val_acc = evaluate(x_val, y_val)
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 5),
                        "val_loss": round(val_loss, 5), "val_accuracy": round(val_acc, 4)})
        if verbose:
            print(f"  epoch {epoch:3d}  train_loss {tr_loss:.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.3f}")

        if val_loss < best_loss - 1e-4:
            best_loss, best_acc, wait = val_loss, val_acc, 0
            best_weights = model.get_weights()
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break

    if best_weights is not None:
        model.set_weights(best_weights)
    return {"best_val_loss": best_loss, "best_val_accuracy": best_acc,
            "epochs_run": len(history), "history": history}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lr", type=float, default=None, help="Adam learning rate")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--use-best", action="store_true",
                    help="load learning_rate / batch_size / dropout from best_hparams.json")
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    args = ap.parse_args()

    hp = dict(DEFAULTS)
    if args.use_best:
        if BEST_HPARAMS_PATH.is_file():
            hp.update({k: v for k, v in json.loads(BEST_HPARAMS_PATH.read_text()).items() if k in hp})
            print(f"Using tuned hyperparameters: {hp}")
        else:
            print(f"NOTE: {BEST_HPARAMS_PATH.name} not found; run tune_hyperparameters.py first. Using defaults.")
    if args.lr is not None:
        hp["learning_rate"] = args.lr
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size
    if args.dropout is not None:
        hp["dropout"] = args.dropout

    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(SEED)

    data = build_splits(load_frames())
    for name in ("train", "val", "test"):
        y = data[name][1]
        print(f"{name:5s}: {len(y):4d} windows  alert={int((y == 0).sum())}  drowsy={int((y == 1).sum())}")

    model = build_model(data["train"][0], dropout=hp["dropout"])
    print(f"\nTraining with Binary Cross-Entropy + Adam {hp}")
    result = fit(model, data, hp["learning_rate"], hp["batch_size"],
                 epochs=args.epochs, patience=args.patience)
    print(f"Best validation loss {result['best_val_loss']:.4f} after {result['epochs_run']} epochs")

    # Compile (BCE + Adam) so the saved file loads exactly like any compiled Keras model.
    model.compile(optimizer=keras.optimizers.Adam(hp["learning_rate"]),
                  loss="binary_crossentropy", metrics=["accuracy"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved {args.out}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(result["history"])

    report = {"hyperparameters": hp, "loss": "binary_crossentropy", "optimizer": "adam",
              "epochs_run": result["epochs_run"], "best_val_loss": result["best_val_loss"],
              "note": "single-person data; same-person performance only"}
    for name in ("val", "test"):
        x, y, c = data[name]
        pred = (model.predict(x, verbose=0).reshape(-1) >= 0.5).astype(int)
        report[name] = metrics(y, pred)
        report[name]["per_clip"] = per_clip(y, pred, c)
        print(f"{name}: {report[name]['accuracy']:.3f} acc, F1 {report[name]['f1']:.3f}, "
              f"confusion {report[name]['confusion']}")
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"History -> {HISTORY_PATH}\nReport  -> {REPORT_PATH}")


if __name__ == "__main__":
    main()