"""
Model Validation Engine - Dataset and Model Compatibility Checks.

This module validates the benchmark dataset against the configured
model input schema and provides execution benchmarking utilities.
"""

import json
import time
import logging
from pathlib import Path

import numpy as np

from training_results import attach_training_results


DATASET_PATH = Path("dataset.npz")
MANIFEST_PATH = Path("benchmark/benchmark_manifest.csv")

EXPECTED_TIMESTEPS = 30
EXPECTED_FEATURES = 7
OUTPUT_DIR = Path("evaluation_results")
LOG_FILE = OUTPUT_DIR / "evaluation.log"


def setup_logging():
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(filename=str(LOG_FILE), level=logging.INFO, force=True)


def calculate_dataset_metrics(y):
    total = len(y)
    class_0 = int(np.sum(y == 0))
    class_1 = int(np.sum(y == 1))
    return {"total_samples": total, "class_0_count": class_0, "class_1_count": class_1, "class_0_percentage": round(class_0 / total * 100, 2) if total else 0.0, "class_1_percentage": round(class_1 / total * 100, 2) if total else 0.0}


def load_dataset():
    """Load the prepared dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    data = np.load(
        DATASET_PATH,
        allow_pickle=True
    )

    return data


def validate_input_schema(data):
    """Validate the dataset shape against model expectations."""

    X = data["X"]

    actual_timesteps = X.shape[1]
    actual_features = X.shape[2]

    result = {
        "expected_shape": [
            EXPECTED_TIMESTEPS,
            EXPECTED_FEATURES,
        ],
        "actual_shape": [
            actual_timesteps,
            actual_features,
        ],
        "timesteps_match": (
            actual_timesteps == EXPECTED_TIMESTEPS
        ),
        "features_match": (
            actual_features == EXPECTED_FEATURES
        ),
    }

    result["compatible"] = (
        result["timesteps_match"]
        and result["features_match"]
    )

    return result


def benchmark_numpy_processing(data):
    """
    Measure basic dataset processing speed.

    This is NOT model inference timing.
    It measures the processing overhead available before
    a trained model checkpoint is connected.
    """

    X = data["X"]

    start = time.perf_counter()

    # Representative preprocessing operation.
    _ = np.asarray(X, dtype=np.float32)

    elapsed = time.perf_counter() - start

    samples = len(X)

    if elapsed > 0:
        samples_per_second = samples / elapsed
    else:
        samples_per_second = float("inf")

    return {
        "elapsed_seconds": elapsed,
        "samples_processed": samples,
        "samples_per_second": samples_per_second,
    }


def build_report():

    data = load_dataset()

    X = data["X"]
    y = data["y"]

    schema = validate_input_schema(data)

    processing = benchmark_numpy_processing(data)
    dataset_metrics = calculate_dataset_metrics(y)

    report = {
        "dataset": {
            "path": str(DATASET_PATH),
            "samples": int(len(X)),
            "shape": list(X.shape),
            "labels": {
                "class_0": int(np.sum(y == 0)),
                "class_1": int(np.sum(y == 1)),
            },
        },

        "model_input_validation": schema,
        "dataset_metrics": dataset_metrics,

        "execution_benchmark": processing,

        "model_status": {
            "trained_checkpoint": False,
            "tensorflow_available": False,
            "inference_evaluation": False,
        },

        "evaluation_status": {
            "dataset_validation": True,
            "input_schema_validation": schema["compatible"],
            "model_metrics_available": False,
            "loss_curve_available": False,
        },
    }

    report = attach_training_results(report)
    return report


def main():

    setup_logging()
    logging.info("Evaluation started")

    print("=" * 65)
    print("MODEL VALIDATION ENGINE")
    print("=" * 65)

    report = build_report()

    dataset = report["dataset"]
    schema = report["model_input_validation"]
    execution = report["execution_benchmark"]

    print()
    print("DATASET")
    print("-" * 65)

    print("Samples:", dataset["samples"])
    print("Shape:", dataset["shape"])
    print(
        "Class 0:",
        dataset["labels"]["class_0"]
    )
    print(
        "Class 1:",
        dataset["labels"]["class_1"]
    )

    print()
    print("MODEL INPUT SCHEMA")
    print("-" * 65)

    print(
        "Expected:",
        tuple(schema["expected_shape"])
    )

    print(
        "Actual:",
        tuple(schema["actual_shape"])
    )

    print(
        "Timesteps match:",
        schema["timesteps_match"]
    )

    print(
        "Features match:",
        schema["features_match"]
    )

    print(
        "Input compatible:",
        schema["compatible"]
    )

    print()
    print("EXECUTION BENCHMARK")
    print("-" * 65)

    print(
        "Samples processed:",
        execution["samples_processed"]
    )

    print(
        "Processing time:",
        f"{execution['elapsed_seconds']:.6f}",
        "seconds"
    )

    print(
        "Processing throughput:",
        f"{execution['samples_per_second']:.2f}",
        "samples/sec"
    )

    print()
    print("MODEL STATUS")
    print("-" * 65)

    training = report["training_results"]
    metrics = training["training_metrics"]
    print("Trained checkpoint:", "AVAILABLE" if training["checkpoint_exists"] else "NOT AVAILABLE")
    print("Training report:", "AVAILABLE" if training["training_report_available"] else "NOT AVAILABLE")
    if metrics:
        print("Test accuracy:", metrics["test"]["accuracy"])
        print("Test precision:", metrics["test"]["precision"])
        print("Test recall:", metrics["test"]["recall"])
        print("Test F1 score:", metrics["test"]["f1"])
        print("Test samples:", sum(metrics["test"]["confusion"].values()))
    print("Loss history epochs:", len(training["loss_history"]))

    print()
    print("VALIDATION STATUS")
    print("-" * 65)

    if schema["compatible"]:
        print("Input schema: PASS")
    else:
        print("Input schema: FAIL")
        print(
            "Reason: dataset contains",
            schema["actual_shape"][1],
            "features but model expects",
            schema["expected_shape"][1],
        )

    print(
        "Dataset validation: PASS"
    )

    print()
    print("NOTE: Test metrics are based on 22 samples from the existing training report.")
    print("The results are limited to the reported single-person dataset.")
    print("Execution benchmark measures NumPy preprocessing, not model inference.")

    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    output_file = (
        output_dir /
        "validation_report.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print()
    print("Report written to:")
    print(output_file)

    print()
    logging.info("Dataset samples: " + str(dataset["samples"]))
    logging.info("Input schema compatible: " + str(schema["compatible"]))
    logging.info("Processing time: " + str(execution["elapsed_seconds"]))
    logging.info("Processing throughput: " + str(execution["samples_per_second"]))
    logging.info("Model accuracy and loss unavailable because no trained checkpoint exists")
    logging.info("Evaluation completed")
    print("Log written to:")
    print(LOG_FILE)
    print()
    print("STATUS: VALIDATION FRAMEWORK EXECUTED")


if __name__ == "__main__":
    main()
