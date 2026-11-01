
"""Convert a trained Keras model to ONNX through SavedModel."""

import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cnn_drowsiness.keras"
ONNX_PATH = ROOT / "models" / "cnn_drowsiness.onnx"
REPORT_PATH = ROOT / "evaluation_results" / "onnx_conversion_report.json"

EXPECTED_SHAPE = (30, 8)
OPSET_VERSION = 13


def write_report(report):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )


def main():
    report = {
        "source_model": str(MODEL_PATH.relative_to(ROOT)),
        "output_model": str(ONNX_PATH.relative_to(ROOT)),
        "expected_input_shape": [None, *EXPECTED_SHAPE],
        "opset": OPSET_VERSION,
        "conversion_method": "Keras SavedModel via tf2onnx.from_function",
        "conversion_status": "FAILED",
        "onnx_validation": "NOT_RUN"
    }

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Keras checkpoint not found: {MODEL_PATH}"
            )

        import tensorflow as tf
        import tf2onnx
        import onnx

        print("Loading trained Keras model...")

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        input_shape = model.input_shape

        if isinstance(input_shape, list):
            raise ValueError("Multiple model inputs are not supported.")

        actual_shape = tuple(input_shape[1:])

        if actual_shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Expected input {EXPECTED_SHAPE}, found {actual_shape}"
            )

        ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            dir=str(ONNX_PATH.parent)
        ) as temp_dir:

            saved_model_dir = Path(temp_dir) / "saved_model"
            temporary_onnx = Path(temp_dir) / "converted.onnx"

            print("Exporting Keras model to TensorFlow SavedModel...")

            model.export(
                str(saved_model_dir),
                format="tf_saved_model"
            )

            loaded_saved_model = tf.saved_model.load(
                str(saved_model_dir)
            )

            signatures = loaded_saved_model.signatures

            if not signatures:
                raise RuntimeError(
                    "SavedModel has no export signatures."
                )

            signature_name = (
                "serve" if "serve" in signatures
                else "serving_default" if "serving_default" in signatures
                else next(iter(signatures))
            )

            print("SavedModel signature:", signature_name)

            serve_function = signatures[signature_name]

            input_signature = [
                tf.TensorSpec(
                    shape=(None, *EXPECTED_SHAPE),
                    dtype=tf.float32,
                    name="input"
                )
            ]

            @tf.function(input_signature=input_signature)
            def serving_function(inputs):
                result = serve_function(inputs)
                return result

            print("Converting SavedModel function to ONNX...")

            converted, _ = tf2onnx.convert.from_function(
                serving_function,
                input_signature=input_signature,
                opset=OPSET_VERSION,
                output_path=str(temporary_onnx)
            )

            print("Validating ONNX graph...")

            onnx.checker.check_model(converted)
            onnx.checker.check_model(str(temporary_onnx))

            temporary_onnx.replace(ONNX_PATH)

        report.update({
            "conversion_status": "SUCCESS",
            "onnx_validation": "PASSED",
            "actual_input_shape": list(actual_shape),
            "saved_model_signature": signature_name,
            "model_size_bytes": ONNX_PATH.stat().st_size
        })

        write_report(report)

        print("ONNX model:", ONNX_PATH)
        print("Report:", REPORT_PATH)
        print("STATUS: ONNX CONVERSION SUCCESSFUL")

    except Exception as exc:
        report.update({
            "conversion_status": "FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc)
        })

        write_report(report)

        print("CONVERSION FAILED:", type(exc).__name__)
        print("Reason:", exc)
        print("Failure report:", REPORT_PATH)

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
