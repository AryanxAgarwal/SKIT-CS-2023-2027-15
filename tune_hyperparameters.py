"""
User story: Hyperparameter Tuning
    "Fine-tune learning rates ..."

Run from the repo root:
    python tune_hyperparameters.py             # learning-rate sweep only (fast)
    python tune_hyperparameters.py --full      # also sweep batch size and dropout
    python tune_hyperparameters.py --seeds 3   # more repeats = steadier result

For each setting it trains the CNN (Binary Cross-Entropy + Adam, same loop as
train_model.py) for several random seeds, and ranks settings by MEAN VALIDATION LOSS.
Validation loss is used instead of accuracy because accuracy is already ~100% on this
data and cannot tell settings apart. The TEST split is never touched here.

Outputs
    evaluation_results/hparam_results.csv   every trial
    evaluation_results/best_hparams.json    the winner (read by: python train_model.py --use-best)

CAUTION: the validation split is small (about 22 windows) and from one person, so
differences between learning rates will be small and somewhat noisy. Treat the winner
as "a sensible setting", not as a proven optimum.
"""
import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from train_drowsiness import build_splits, load_frames, metrics
from train_model import BEST_HPARAMS_PATH, RESULTS_DIR, build_model, fit

LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
BATCH_SIZES = [16, 32]
DROPOUTS = [0.2, 0.4]
RESULTS_PATH = RESULTS_DIR / "hparam_results.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also sweep batch size and dropout")
    ap.add_argument("--seeds", type=int, default=2, help="random seeds per setting")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    import tensorflow as tf

    data = build_splits(load_frames())
    x_val, y_val, _ = data["val"]
    grid = list(itertools.product(
        LEARNING_RATES,
        BATCH_SIZES if args.full else [32],
        DROPOUTS if args.full else [0.4],
    ))
    print(f"{len(grid)} settings x {args.seeds} seeds = {len(grid) * args.seeds} training runs\n")

    rows = []
    for lr, bs, dr in grid:
        losses, f1s, accs, epochs_run = [], [], [], []
        t0 = time.time()
        for seed in range(args.seeds):
            tf.keras.utils.set_random_seed(1000 + seed)
            model = build_model(data["train"][0], dropout=dr)
            res = fit(model, data, lr, bs, epochs=args.epochs, patience=args.patience,
                      seed=1000 + seed, verbose=False)
            pred = (model(x_val, training=False).numpy().reshape(-1) >= 0.5).astype(int)
            m = metrics(y_val, pred)
            losses.append(res["best_val_loss"]); f1s.append(m["f1"])
            accs.append(m["accuracy"]); epochs_run.append(res["epochs_run"])
        row = {"learning_rate": lr, "batch_size": bs, "dropout": dr,
               "mean_val_loss": round(float(np.mean(losses)), 5),
               "std_val_loss": round(float(np.std(losses)), 5),
               "mean_val_f1": round(float(np.mean(f1s)), 4),
               "mean_val_accuracy": round(float(np.mean(accs)), 4),
               "mean_epochs": round(float(np.mean(epochs_run)), 1)}
        rows.append(row)
        print(f"lr={lr:<7g} batch={bs:<3d} dropout={dr:<4}  val_loss {row['mean_val_loss']:.4f} "
              f"(+/-{row['std_val_loss']:.4f})  val_F1 {row['mean_val_f1']:.3f}  "
              f"epochs {row['mean_epochs']:.0f}  [{time.time() - t0:.0f}s]")

    rows.sort(key=lambda r: (r["mean_val_loss"], -r["mean_val_f1"]))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    BEST_HPARAMS_PATH.write_text(json.dumps(
        {"learning_rate": best["learning_rate"], "batch_size": best["batch_size"],
         "dropout": best["dropout"], "mean_val_loss": best["mean_val_loss"]}, indent=2))

    print("\nRanking (best first):")
    for i, r in enumerate(rows, 1):
        print(f"  {i}. lr={r['learning_rate']:g} batch={r['batch_size']} dropout={r['dropout']}  "
              f"val_loss={r['mean_val_loss']:.4f}")
    print(f"\nBest: learning_rate={best['learning_rate']:g}, batch_size={best['batch_size']}, "
          f"dropout={best['dropout']}")
    print(f"Saved {BEST_HPARAMS_PATH}\nNext: python train_model.py --use-best")


if __name__ == "__main__":
    main()