"""
User story: Sprint 2 - Model Unit Testing & Metric Verification
    "Measure classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC curve)."

Run from the repo root (next to train_drowsiness.py):
    python evaluate_model.py                                   # Keras model, validation + test
    python evaluate_model.py --model models/cnn_drowsiness_int8.tflite
    python evaluate_model.py --threshold 0.4                   # change the decision threshold

Class 1 = drowsy (positive), class 0 = alert. The model outputs P(drowsy) per 30-frame window.

Outputs (evaluation_results/)
    metrics_report.json     all numbers
    roc_curve.png           ROC curves for validation and test, with AUC
    confusion_matrix.png    test-set confusion matrix

How to read the metrics (positive = drowsy)
    Accuracy   share of windows classified correctly
    Precision  of the windows called drowsy, how many really were   (few false alarms)
    Recall     of the truly drowsy windows, how many were caught     (few missed events)
    F1         balance of precision and recall
    ROC-AUC    ranking quality over ALL thresholds (0.5 = guessing, 1.0 = perfect)

IMPORTANT: the data comes from one person, and the test set is small, so these numbers
describe same-person performance and can be unstable. They do not prove real-world accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from train_drowsiness import MODEL_PATH, build_splits, load_frames, per_clip

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "evaluation_results"
REPORT_PATH = RESULTS_DIR / "metrics_report.json"
ROC_PATH = RESULTS_DIR / "roc_curve.png"
CM_PATH = RESULTS_DIR / "confusion_matrix.png"
SMALL_TEST_SET = 50


# ------------------------------------------------------------------ metrics
def compute_metrics(y_true, y_prob, threshold=0.5):
    """Classification metrics for P(drowsy) scores. roc_auc is None if only one class is present."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if y_true.size == 0:
        raise ValueError("No samples to evaluate.")
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true has {y_true.size} values but y_prob has {y_prob.size}.")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or infinite values.")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("y_true must contain only 0 (alert) and 1 (drowsy).")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) == 2
    return {
        "n": int(y_true.size),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def roc_points(y_true, y_prob):
    """False-positive rate, true-positive rate and thresholds for the ROC curve."""
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float))
    return fpr, tpr, thr


# -------------------------------------------------------------------- model
def predict_probabilities(model_path, x):
    """P(drowsy) for every window, from a .keras model or an INT8/float .tflite model."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}\nTrain first: python train_model.py")
    if model_path.suffix == ".tflite":
        from quantize_int8 import tflite_predict
        probs, _ = tflite_predict(model_path, x)
        return np.asarray(probs, dtype=float)
    from tensorflow import keras
    model = keras.models.load_model(model_path, compile=False)
    return model(x, training=False).numpy().reshape(-1).astype(float)


# -------------------------------------------------------------------- plots
def plot_roc(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, (fpr, tpr, auc, point) in curves.items():
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {auc:.3f})")
        if point is not None:
            ax.scatter([point[0]], [point[1]], s=40, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False positive rate (false alarms)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve - drowsy vs alert")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(conf, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1], ["Alert", "Drowsy"])
    ax.set_yticks([0, 1], ["Alert", "Drowsy"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(grid[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if grid[i, j] > grid.max() / 2 else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- main
def fmt(v):
    return "  n/a " if v is None else f"{v:6.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=MODEL_PATH, help=".keras or .tflite model")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    data = build_splits(load_frames())
    report = {"model": str(args.model), "threshold": args.threshold,
              "note": "single-person data; same-person performance only"}
    curves = {}

    print(f"Model: {args.model}   decision threshold: {args.threshold}\n")
    print(f"{'split':6s} {'n':>4s} {'accuracy':>9s} {'precision':>10s} {'recall':>7s} {'F1':>7s} {'ROC-AUC':>8s}   confusion (tn fp fn tp)")
    for name in ("val", "test"):
        x, y, clips = data[name]
        probs = predict_probabilities(args.model, x)
        m = compute_metrics(y, probs, args.threshold)
        pred = (probs >= args.threshold).astype(int)
        m["per_clip_accuracy"] = per_clip(y, pred, clips)
        report[name] = m
        c = m["confusion"]
        print(f"{name:6s} {m['n']:4d} {fmt(m['accuracy']):>9s} {fmt(m['precision']):>10s} "
              f"{fmt(m['recall']):>7s} {fmt(m['f1']):>7s} {fmt(m['roc_auc']):>8s}   "
              f"{c['tn']} {c['fp']} {c['fn']} {c['tp']}")
        if m["roc_auc"] is not None:
            fpr, tpr, _ = roc_points(y, probs)
            point = (c["fp"] / max(c["fp"] + c["tn"], 1), c["tp"] / max(c["tp"] + c["fn"], 1))
            curves[f"{name} set"] = (fpr, tpr, m["roc_auc"], point)
        else:
            print(f"       ({name} has only one class, so ROC-AUC is undefined)")

    if curves:
        plot_roc(curves, ROC_PATH)
        print(f"\nROC curve        -> {ROC_PATH}")
    plot_confusion(report["test"]["confusion"], CM_PATH, f"Test set (threshold {args.threshold})")
    print(f"Confusion matrix -> {CM_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Report           -> {REPORT_PATH}")
    if report["test"]["n"] < SMALL_TEST_SET:
        print(f"\nNOTE: the test set has only {report['test']['n']} windows from one person, so these "
              "numbers are unstable and optimistic. Add recordings of more people before quoting them.")


if __name__ == "__main__":
    main()