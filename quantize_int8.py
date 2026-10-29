"""
User story: Quantization
    "... apply INT8 post-training quantization to shrink model footprint."

Run from the repo root, after training (python train_model.py):
    python quantize_int8.py
    python quantize_int8.py --int8-io      # also make input/output tensors int8 (microcontrollers)

What it does
  1. Loads models/cnn_drowsiness.keras.
  2. Converts it to a float32 TFLite model (reference) and a full-INT8 TFLite model.
     Post-training quantization needs no retraining; it uses ~200 real training windows
     as a "representative dataset" to calibrate the INT8 ranges.
  3. Compares Keras float / TFLite float32 / TFLite INT8 on the held-out TEST windows:
     file size, accuracy, F1, average latency, and how often INT8 agrees with float.
  4. Saves models/cnn_drowsiness_float32.tflite, models/cnn_drowsiness_int8.tflite and
     evaluation_results/quantization_report.json.

By default the INT8 model keeps float32 input/output (easy to use from Python);
the weights and internal maths are INT8. --int8-io makes the model INT8 end to end.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from train_drowsiness import MODEL_PATH, SEED, build_splits, load_frames, metrics

ROOT = Path(__file__).resolve().parent
FLOAT_TFLITE = ROOT / "models" / "cnn_drowsiness_float32.tflite"
INT8_TFLITE = ROOT / "models" / "cnn_drowsiness_int8.tflite"
REPORT_PATH = ROOT / "evaluation_results" / "quantization_report.json"
CALIBRATION_WINDOWS = 200
MAX_ACCURACY_DROP = 0.01        # warn if INT8 loses more than 1 point of accuracy


def make_converter(model):
    import tensorflow as tf
    return tf.lite.TFLiteConverter.from_keras_model(model)


def convert_float32(model):
    return make_converter(model).convert()


def convert_int8(model, x_train, int8_io):
    import tensorflow as tf

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(x_train), size=min(CALIBRATION_WINDOWS, len(x_train)), replace=False)

    def representative_dataset():
        for i in pick:
            yield [x_train[i][None].astype(np.float32)]

    def build(ops):
        conv = make_converter(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_dataset
        conv.target_spec.supported_ops = ops
        if int8_io:
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
        return conv.convert()

    try:
        return build([tf.lite.OpsSet.TFLITE_BUILTINS_INT8]), True
    except Exception as exc:                       # an op without an INT8 kernel
        print(f"NOTE: strict INT8 conversion failed ({exc}).")
        if int8_io:
            raise
        print("      Retrying with float fallback for unsupported ops (not pure INT8).")
        return build([tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]), False


def tflite_predict(path, x):
    """Run a .tflite model one window at a time. Returns (probabilities, ms per window)."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    probs = []
    start = time.perf_counter()
    for window in x:
        xi = window[None].astype(np.float32)
        if inp["dtype"] == np.int8:
            scale, zero = inp["quantization"]
            xi = np.clip(np.round(xi / scale + zero), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], xi)
        interp.invoke()
        yo = interp.get_tensor(out["index"])
        if out["dtype"] == np.int8:
            scale, zero = out["quantization"]
            yo = (yo.astype(np.float32) - zero) * scale
        probs.append(float(np.asarray(yo).reshape(-1)[0]))
    ms = (time.perf_counter() - start) / max(len(x), 1) * 1000
    return np.asarray(probs), ms


def kb(path):
    return round(Path(path).stat().st_size / 1024, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--int8-io", action="store_true", help="int8 input/output tensors too")
    args = ap.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}\nTrain first: python train_model.py")

    from tensorflow import keras
    model = keras.models.load_model(args.model, compile=False)
    data = build_splits(load_frames())
    x_train = data["train"][0]
    x_test, y_test, _ = data["test"]

    print("Converting to float32 TFLite ...")
    FLOAT_TFLITE.parent.mkdir(parents=True, exist_ok=True)
    FLOAT_TFLITE.write_bytes(convert_float32(model))

    print("Converting to INT8 TFLite (post-training quantization) ...")
    int8_bytes, pure_int8 = convert_int8(model, x_train, args.int8_io)
    INT8_TFLITE.write_bytes(int8_bytes)

    print("Evaluating on the test windows ...")
    t0 = time.perf_counter()
    p_keras = model(x_test, training=False).numpy().reshape(-1)
    keras_ms = (time.perf_counter() - t0) / max(len(x_test), 1) * 1000
    p_float, float_ms = tflite_predict(FLOAT_TFLITE, x_test)
    p_int8, int8_ms = tflite_predict(INT8_TFLITE, x_test)

    results = {}
    for name, p, ms, path in (("keras_float32", p_keras, keras_ms, args.model),
                              ("tflite_float32", p_float, float_ms, FLOAT_TFLITE),
                              ("tflite_int8", p_int8, int8_ms, INT8_TFLITE)):
        pred = (p >= 0.5).astype(int)
        m = metrics(y_test, pred)
        results[name] = {"size_kb": kb(path), "accuracy": m["accuracy"], "f1": m["f1"],
                         "ms_per_window": round(ms, 3), "confusion": m["confusion"]}

    agree = float(((p_int8 >= 0.5) == (p_float >= 0.5)).mean())
    max_dev = float(np.max(np.abs(p_int8 - p_float)))
    ratio = results["keras_float32"]["size_kb"] / max(results["tflite_int8"]["size_kb"], 1e-9)
    drop = results["tflite_float32"]["accuracy"] - results["tflite_int8"]["accuracy"]

    print(f"\n{'model':16s} {'size KB':>9s} {'accuracy':>9s} {'F1':>6s} {'ms/window':>10s}")
    for name, r in results.items():
        print(f"{name:16s} {r['size_kb']:9.1f} {r['accuracy']:9.3f} {r['f1']:6.3f} {r['ms_per_window']:10.3f}")
    print(f"\nINT8 vs float agreement: {agree * 100:.1f}% of windows (max probability difference {max_dev:.3f})")
    print(f"Size reduction vs Keras file: {ratio:.1f}x smaller")
    if not pure_int8:
        print("NOTE: some ops stayed in float, so this is not pure INT8.")
    if drop > MAX_ACCURACY_DROP:
        print(f"WARNING: INT8 lost {drop * 100:.1f} accuracy points; try more calibration windows or check the data.")

    report = {"results": results, "int8_float_agreement": round(agree, 4),
              "max_probability_difference": round(max_dev, 4),
              "size_reduction_x": round(ratio, 2), "pure_int8": pure_int8,
              "int8_io": args.int8_io, "calibration_windows": min(CALIBRATION_WINDOWS, len(x_train)),
              "note": "test windows are from a single person; small sample"}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {INT8_TFLITE}\nSaved {FLOAT_TFLITE}\nReport -> {REPORT_PATH}")


if __name__ == "__main__":
    main()