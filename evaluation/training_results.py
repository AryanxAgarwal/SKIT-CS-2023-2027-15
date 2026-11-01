import csv
import json
from pathlib import Path

REPORT_PATH = Path("evaluation_results/train_model_report.json")
HISTORY_PATH = Path("evaluation_results/training_history.csv")
MODEL_PATH = Path("models/cnn_drowsiness.keras")

def load_training_results():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else None
    history = []
    if HISTORY_PATH.exists():
        with HISTORY_PATH.open(encoding="utf-8", newline="") as file:
            history = list(csv.DictReader(file))
    return {
        "checkpoint_exists": MODEL_PATH.exists(),
        "training_report_available": report is not None,
        "training_metrics": report,
        "loss_history": history,
        "loss_history_available": bool(history)
    }

def attach_training_results(report):
    results = load_training_results()
    report["training_results"] = results
    report["model_status"]["trained_checkpoint"] = results["checkpoint_exists"]
    report["model_status"]["inference_evaluation"] = results["training_report_available"]
    report["evaluation_status"]["model_metrics_available"] = results["training_report_available"]
    report["evaluation_status"]["loss_curve_available"] = results["loss_history_available"]
    return report
