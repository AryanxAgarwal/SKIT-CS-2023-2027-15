
"""Convert, quantize and validate the trained drowsiness model."""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from train_drowsiness import (
    MODEL_PATH,
    SEED,
    build_splits,
    load_frames,
    metrics,
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
FLOAT_TFLITE = MODELS_DIR / "cnn_drowsiness_float32.tflite"
INT8_TFLITE = MODELS_DIR / "cnn_drowsiness_int8.tflite"
REPORT_PATH = ROOT / "evaluation_results" / "quantization_report.json"

CALIBRATION_WINDOWS = 200
MAX_ACCURACY_DROP = 0.01
EXPECTED_INPUT_SHAPE = (30, 8)


def validate_data(model, x_train, x_test, y_test):
    """Validate model input schema and calibration/test datasets."""

    model_shape = tuple(model.input_shape[1:])

    if model_shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Model input shape mismatch: expected "
            f"{EXPECTED_INPUT_SHAPE}, found {model_shape}"
        )

    if x_train.ndim != 3 or x_test.ndim != 3:
        raise ValueError("Training and test inputs must be 3-dimensional.")

    if tuple(x_train.shape[1:]) != model_shape:
        raise ValueError(
            f"Training data shape {x_train.shape[1:]} "
            f"does not match model input {model_shape}."
        )

    if tuple(x_test.shape[1:]) != model_shape:
        raise ValueError(
            f"Test data shape {x_test.shape[1:]} "
            f"does not match model input {model_shape}."
        )

    if len(x_train) == 0:
        raise ValueError("Calibration dataset is empty.")

    if len(x_test) == 0:
        raise ValueError("Test dataset is empty.")

    if len(x_test) != len(y_test):
        raise ValueError("Test input and label counts do not match.")

    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    print("Input schema:", model_shape)
    print("Calibration windows available:", len(x_train))
    print("Test windows available:", len(x_test))


def make_converter(model):
    import tensorflow as tf
    return tf.lite.TFLiteConverter.from_keras_model(model)


def convert_float32(model):
    return make_converter(model).convert()


def convert_int8(model, x_train, int8_io):
    import tensorflow as tf

    rng = np.random.default_rng(SEED)
    count = min(CALIBRATION_WINDOWS, len(x_train))

    if count == 0:
        raise ValueError("No calibration windows available.")

    selected = rng.choice(len(x_train), size=count, replace=False)

    def representative_dataset():
        for index in selected:
            yield [x_train[index:index + 1].astype(np.float32)]

    def build(ops):
        converter = make_converter(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = ops

        if int8_io:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        return converter.convert()

    try:
        return build([tf.lite.OpsSet.TFLITE_BUILTINS_INT8]), True

    except Exception as exc:
        print("NOTE: Strict INT8 conversion failed:", exc)

        if int8_io:
            raise

        print("Retrying with float fallback for unsupported operations.")

        return build([
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]), False


def quantize_input(values, details):
    """Convert float input into the tensor's required data type."""

    dtype = details["dtype"]

    if dtype in (np.int8, np.uint8):
        scale, zero_point = details["quantization"]

        if scale <= 0:
            raise ValueError("Invalid input quantization scale.")

        limits = np.iinfo(dtype)

        values = np.round(values / scale + zero_point)
        values = np.clip(values, limits.min, limits.max)

    return values.astype(dtype)


def dequantize_output(values, details):
    """Convert quantized output tensors into float probabilities."""

    if details["dtype"] in (np.int8, np.uint8):
        scale, zero_point = details["quantization"]

        if scale <= 0:
            raise ValueError("Invalid output quantization scale.")

        values = (values.astype(np.float32) - zero_point) * scale

    return values


def tflite_predict(path, x):
    """Validate a TFLite model and run inference on test windows."""

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    expected = tuple(input_details["shape"][1:])

    if expected != tuple(x.shape[1:]):
        raise ValueError(
            f"TFLite input shape {expected} does not match "
            f"test data shape {x.shape[1:]}."
        )

    probabilities = []
    start = time.perf_counter()

    for window in x:
        values = quantize_input(window[None].astype(np.float32), input_details)

        interpreter.set_tensor(input_details["index"], values)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details["index"])
        output = dequantize_output(output, output_details)

        probabilities.append(float(np.asarray(output).reshape(-1)[0]))

    elapsed = time.perf_counter() - start
    latency = elapsed / len(x) * 1000

    probabilities = np.asarray(probabilities)

    if not np.isfinite(probabilities).all():
        raise ValueError("TFLite inference produced invalid probabilities.")

    return probabilities, latency


def kb(path):
    return round(Path(path).stat().st_size / 1024, 1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--int8-io",
        action="store_true",
        help="Use INT8 input and output tensors."
    )

    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")

    try:
        from tensorflow import keras

        print("Loading trained Keras model...")
        model = keras.models.load_model(args.model, compile=False)

        data = build_splits(load_frames())

        x_train = data["train"][0]
        x_test, y_test, _ = data["test"]

        validate_data(model, x_train, x_test, y_test)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Temporary files ensure existing models remain unchanged
        # until both converted models pass interpreter validation.
        with tempfile.TemporaryDirectory(dir=str(MODELS_DIR)) as temp:
            temp_dir = Path(temp)

            temp_float = temp_dir / "float32.tflite"
            temp_int8 = temp_dir / "int8.tflite"

            print("\nConverting Float32 TFLite model...")
            temp_float.write_bytes(convert_float32(model))

            print("Converting INT8 TFLite model...")
            int8_bytes, pure_int8 = convert_int8(
                model, x_train, args.int8_io
            )
            temp_int8.write_bytes(int8_bytes)

            print("\nValidating converted models...")

            # Run actual inference before publishing model files.
            p_float, float_ms = tflite_predict(temp_float, x_test)
            p_int8, int8_ms = tflite_predict(temp_int8, x_test)

            print("Float32 interpreter: PASS")
            print("INT8 interpreter: PASS")

            t0 = time.perf_counter()
            p_keras = model(x_test, training=False).numpy().reshape(-1)
            keras_ms = (
                time.perf_counter() - t0
            ) / len(x_test) * 1000

            results = {}

            models = (
                ("keras_float32", p_keras, keras_ms, args.model),
                ("tflite_float32", p_float, float_ms, temp_float),
                ("tflite_int8", p_int8, int8_ms, temp_int8),
            )

            for name, probabilities, latency, path in models:
                predictions = (probabilities >= 0.5).astype(int)
                model_metrics = metrics(y_test, predictions)

                results[name] = {
                    "size_kb": kb(path),
                    "accuracy": model_metrics["accuracy"],
                    "f1": model_metrics["f1"],
                    "ms_per_window": round(latency, 3),
                    "confusion": model_metrics["confusion"],
                }

            agreement = float(
                ((p_int8 >= 0.5) == (p_float >= 0.5)).mean()
            )

            max_deviation = float(
                np.max(np.abs(p_int8 - p_float))
            )

            ratio = (
                results["keras_float32"]["size_kb"]
                / max(results["tflite_int8"]["size_kb"], 1e-9)
            )

            accuracy_drop = (
                results["tflite_float32"]["accuracy"]
                - results["tflite_int8"]["accuracy"]
            )

            # Publish only after both TFLite models pass inference.
            temp_float.replace(FLOAT_TFLITE)
            temp_int8.replace(INT8_TFLITE)

        print("\nMODEL COMPARISON")
        print("-" * 60)

        for name, result in results.items():
            print(
                f"{name:16s} "
                f"{result['size_kb']:8.1f} KB "
                f"{result['accuracy']:8.3f} accuracy "
                f"{result['f1']:8.3f} F1 "
                f"{result['ms_per_window']:8.3f} ms"
            )

        print("\nINT8 agreement:", round(agreement * 100, 2), "%")
        print("Maximum probability difference:", round(max_deviation, 4))
        print("Size reduction:", round(ratio, 2), "x")

        if not pure_int8:
            print("WARNING: Some operations use float fallback.")

        if accuracy_drop > MAX_ACCURACY_DROP:
            print(
                "WARNING: INT8 accuracy degradation:",
                round(accuracy_drop * 100, 2),
                "percentage points."
            )

        report = {
            "results": results,
            "int8_float_agreement": round(agreement, 4),
            "max_probability_difference": round(max_deviation, 4),
            "size_reduction_x": round(ratio, 2),
            "pure_int8": pure_int8,
            "int8_io": args.int8_io,
            "calibration_windows": min(
                CALIBRATION_WINDOWS, len(x_train)
            ),
            "conversion_validation": "PASSED",
            "note": (
                "Validation uses the existing single-person test dataset. "
                "Further testing on diverse subjects is required."
            ),
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8"
        )

        print("\nSaved:", FLOAT_TFLITE)
        print("Saved:", INT8_TFLITE)
        print("Report:", REPORT_PATH)
        print("STATUS: TFLITE CONVERSION AND VALIDATION COMPLETED")

    except Exception as exc:
        print("\nCONVERSION FAILED:", type(exc).__name__)
        print("Reason:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
