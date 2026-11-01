
"""Validate and compare TFLite Float32 and INT8 conversion results."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "evaluation_results" / "quantization_report.json"
OUTPUT = ROOT / "evaluation_results" / "conversion_validation_report.json"

ACCURACY_DROP_LIMIT = 0.02
F1_DROP_LIMIT = 0.02
MIN_INT8_ACCURACY = 0.90


def load_report():
    """Load the existing quantization report."""
    if not REPORT.is_file():
        raise FileNotFoundError(f"Quantization report not found: {REPORT}")

    try:
        return json.loads(REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in quantization report: {exc}") from exc


def validate_metrics(section, model_name):
    """Check required metrics and their valid ranges."""
    required = ("accuracy", "f1", "ms_per_window", "size_kb")

    for key in required:
        value = section.get(key)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{model_name}: missing or invalid metric '{key}'")

        if key in ("accuracy", "f1") and not 0 <= value <= 1:
            raise ValueError(f"{model_name}: {key} must be between 0 and 1")

        if key in ("ms_per_window", "size_kb") and value <= 0:
            raise ValueError(f"{model_name}: {key} must be greater than zero")

    confusion = section.get("confusion")

    if not isinstance(confusion, dict):
        raise ValueError(f"{model_name}: confusion matrix is missing")

    for key in ("tn", "fp", "fn", "tp"):
        value = confusion.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{model_name}: confusion matrix value '{key}' is invalid"
            )


def compare_models(data):
    """Compare Float32 and INT8 results and prepare validation findings."""
    results = data.get("results", {})

    if not isinstance(results, dict):
        raise ValueError("The results section is missing or invalid")

    float32 = results.get("tflite_float32")
    int8 = results.get("tflite_int8")

    if not isinstance(float32, dict) or not isinstance(int8, dict):
        raise ValueError("TFLite Float32 or INT8 results are missing")

    validate_metrics(float32, "TFLite Float32")
    validate_metrics(int8, "TFLite INT8")

    accuracy_drop = float32["accuracy"] - int8["accuracy"]
    f1_drop = float32["f1"] - int8["f1"]

    latency_improvement = (
        (float32["ms_per_window"] - int8["ms_per_window"])
        / float32["ms_per_window"]
    ) * 100

    size_reduction = (
        (float32["size_kb"] - int8["size_kb"])
        / float32["size_kb"]
    ) * 100

    float32_fn = float32["confusion"]["fn"]
    int8_fn = int8["confusion"]["fn"]

    findings = []

    if accuracy_drop > ACCURACY_DROP_LIMIT:
        findings.append(
            "INT8 accuracy degradation exceeds 2 percentage points."
        )

    if f1_drop > F1_DROP_LIMIT:
        findings.append(
            "INT8 F1 degradation exceeds 2 percentage points."
        )

    if int8["accuracy"] < MIN_INT8_ACCURACY:
        findings.append(
            "INT8 accuracy is below the 90% validation threshold."
        )

    if int8_fn > float32_fn:
        findings.append(
            f"INT8 produces {int8_fn - float32_fn} additional false negatives."
        )

    if not findings:
        findings.append("No configured validation threshold was breached.")

    status = "REVIEW_REQUIRED" if (
        accuracy_drop > ACCURACY_DROP_LIMIT
        or f1_drop > F1_DROP_LIMIT
        or int8["accuracy"] < MIN_INT8_ACCURACY
        or int8_fn > float32_fn
    ) else "PASS"

    return {
        "source_report": str(REPORT.relative_to(ROOT)),
        "validation_status": status,
        "acceptance_thresholds": {
            "maximum_accuracy_drop": ACCURACY_DROP_LIMIT,
            "maximum_f1_drop": F1_DROP_LIMIT,
            "minimum_int8_accuracy": MIN_INT8_ACCURACY
        },
        "float32": float32,
        "int8": int8,
        "comparison": {
            "accuracy_drop_percentage_points": round(
                accuracy_drop * 100, 2
            ),
            "f1_drop_percentage_points": round(
                f1_drop * 100, 2
            ),
            "latency_improvement_percent": round(
                latency_improvement, 2
            ),
            "model_size_reduction_percent": round(
                size_reduction, 2
            ),
            "float32_false_negatives": float32_fn,
            "int8_false_negatives": int8_fn
        },
        "findings": findings,
        "limitations": [
            "Validation reuses the existing quantization report.",
            "This script does not rerun model inference.",
            "Reported test results are based on a small, single-person dataset."
        ]
    }


def main():
    try:
        data = load_report()
        result = compare_models(data)

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8"
        )

    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"VALIDATION ERROR: {exc}") from exc

    print("=" * 55)
    print("MODEL CONVERSION VALIDATION")
    print("=" * 55)

    print("Status:", result["validation_status"])

    comparison = result["comparison"]

    print(
        "Accuracy degradation:",
        comparison["accuracy_drop_percentage_points"],
        "percentage points"
    )
    print(
        "F1 degradation:",
        comparison["f1_drop_percentage_points"],
        "percentage points"
    )
    print(
        "Latency improvement:",
        comparison["latency_improvement_percent"],
        "%"
    )
    print(
        "Model size reduction:",
        comparison["model_size_reduction_percent"],
        "%"
    )
    print(
        "False negatives (Float32 / INT8):",
        comparison["float32_false_negatives"],
        "/",
        comparison["int8_false_negatives"]
    )

    for finding in result["findings"]:
        print("FINDING:", finding)

    print("Report saved:", OUTPUT)
    print("STATUS: CONVERSION VALIDATION COMPLETED")


if __name__ == "__main__":
    main()
