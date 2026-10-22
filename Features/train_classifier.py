
"""
Train the driver drowsiness CNN and export it to TensorFlow Lite.

Expected inputs:
    training/X_sequences.npy
    training/y_sequences.npy
    training/sequence_clips.npy

Labels:
    0 = alert
    1 = drowsy

Validation:
    Chronological split within each source video, with a 29-window
    gap to prevent train/validation sequences from sharing frames.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.cnn_model import (
    build_cnn_model,
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "training"
KERAS_OUTPUT = ROOT / "models" / "cnn_drowsiness.keras"
TFLITE_OUTPUT = ROOT / "model" / "ddd_classifier.tflite"

# With stride 1, each 30-frame sequence shares 29 frames with
# the next sequence. A gap of 29 sequence windows prevents overlap.
VALIDATION_GAP = SEQUENCE_LENGTH - 1


def chronological_split(clips, train_fraction=0.75):
    """Split each video chronologically without shared frames."""

    clips = np.asarray(clips)
    train_mask = np.zeros(len(clips), dtype=bool)
    val_mask = np.zeros(len(clips), dtype=bool)

    for clip in np.unique(clips):
        indices = np.flatnonzero(clips == clip)
        count = len(indices)

        train_end = int(count * train_fraction)
        val_start = train_end + VALIDATION_GAP

        if train_end == 0 or val_start >= count:
            print(f"Warning: too few sequences for validation in {clip}")
            continue

        train_mask[indices[:train_end]] = True
        val_mask[indices[val_start:]] = True

    return train_mask, val_mask


def calculate_metrics(y_true, probabilities):
    """Calculate metrics using 0=alert and 1=drowsy."""

    predictions = (probabilities >= 0.5).astype(np.int32)
    y_true = np.asarray(y_true).astype(np.int32)

    true_alert = y_true == 0
    true_drowsy = y_true == 1
    pred_alert = predictions == 0
    pred_drowsy = predictions == 1

    accuracy = np.mean(predictions == y_true)

    alert_recall = (
        np.sum(true_alert & pred_alert) / max(np.sum(true_alert), 1)
    )
    drowsy_recall = (
        np.sum(true_drowsy & pred_drowsy) / max(np.sum(true_drowsy), 1)
    )
    drowsy_precision = (
        np.sum(true_drowsy & pred_drowsy) / max(np.sum(pred_drowsy), 1)
    )

    f1 = (
        2 * drowsy_precision * drowsy_recall
        / max(drowsy_precision + drowsy_recall, 1e-12)
    )

    print("\nValidation results (threshold = 0.5)")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Alert recall:      {alert_recall:.4f}")
    print(f"Drowsy recall:     {drowsy_recall:.4f}")
    print(f"Drowsy precision:  {drowsy_precision:.4f}")
    print(f"Drowsy F1-score:   {f1:.4f}")

    print("\nConfusion matrix")
    print("Rows = actual, columns = predicted")
    print("                 Predicted alert   Predicted drowsy")
    print(
        f"Actual alert     "
        f"{np.sum(true_alert & pred_alert):15d}"
        f"{np.sum(true_alert & pred_drowsy):19d}"
    )
    print(
        f"Actual drowsy    "
        f"{np.sum(true_drowsy & pred_alert):15d}"
        f"{np.sum(true_drowsy & pred_drowsy):19d}"
    )

    return predictions


def export_tflite(model, output_path):
    """Export the complete model, including normalization."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    print(f"\nTFLite model saved: {output_path}")
    print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")

    return tflite_model


def predict_tflite(tflite_model, X):
    """Run inference using the exported TFLite model."""

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []

    for sequence in X:
        batch = sequence[np.newaxis, ...].astype(
            input_details["dtype"]
        )

        interpreter.set_tensor(input_details["index"], batch)
        interpreter.invoke()

        probability = interpreter.get_tensor(
            output_details["index"]
        ).reshape(-1)[0]

        predictions.append(float(probability))

    return np.asarray(predictions, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    x_path = DATA_DIR / "X_sequences.npy"
    y_path = DATA_DIR / "y_sequences.npy"
    clips_path = DATA_DIR / "sequence_clips.npy"

    for path in (x_path, y_path, clips_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run python training\\create_sequences.py first."
            )

    # Load the outputs from the corrected sequence-generation pipeline.
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int32).reshape(-1)
    clips = np.load(clips_path, allow_pickle=True).astype(str).reshape(-1)

    print("Loaded training data")
    print(f"Sequences: {len(X)}")
    print(f"Sequence shape: {X.shape[1:]}")
    print(f"Features per frame: {FEATURE_COUNT}")
    print(f"Unique source videos: {len(np.unique(clips))}")
    print(f"Alert sequences: {np.sum(y == 0)}")
    print(f"Drowsy sequences: {np.sum(y == 1)}")

    if X.ndim != 3:
        raise ValueError(f"Expected 3D X array; received shape {X.shape}.")

    if X.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(
            f"Expected each sequence to have shape "
            f"({SEQUENCE_LENGTH}, {FEATURE_COUNT}), got {X.shape[1:]}."
        )

    if len(X) != len(y) or len(y) != len(clips):
        raise ValueError("X, y, and clips must contain the same number of items.")

    if len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("Input sequences are empty or contain invalid values.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must contain only 0 (alert) and 1 (drowsy).")

    if len(np.unique(y)) != 2:
        raise ValueError("Training data must contain both alert and drowsy labels.")

    train_mask, val_mask = chronological_split(clips)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_clips = clips[val_mask]

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            "The chronological split produced an empty training or "
            "validation set. Check the number of sequences per video."
        )

    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError(
            "Both alert and drowsy examples must appear in training and "
            "validation. Review the source videos and their labels."
        )

    print(f"\nTraining sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    print(f"Gap between training and validation: {VALIDATION_GAP} windows")
    print(f"Training labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Validation labels: {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # Fit normalization only on training data.
    # It is included in the final model so inference can use raw features.
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train.reshape(-1, FEATURE_COUNT))

    cnn = build_cnn_model(
        timesteps=SEQUENCE_LENGTH,
        features=FEATURE_COUNT,
    )

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(SEQUENCE_LENGTH, FEATURE_COUNT)),
            normalizer,
            cnn,
        ],
        name="normalized_driver_drowsiness_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Balance the two classes during training.
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = {
        label: len(y_train) / (2.0 * class_counts[label])
        for label in (0, 1)
    }

    tf.keras.utils.set_random_seed(42)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        )
    ]

    print("\nStarting training...")

    model.fit(
        X_train,
        y_train.astype(np.float32),
        validation_data=(X_val, y_val.astype(np.float32)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    KERAS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(KERAS_OUTPUT)
    print(f"\nKeras model saved: {KERAS_OUTPUT}")

    # Evaluate the exported model, not only the in-memory model.
    tflite_model = export_tflite(model, TFLITE_OUTPUT)
    probabilities = predict_tflite(tflite_model, X_val)
    predictions = calculate_metrics(y_val, probabilities)

    print("\nValidation accuracy by source video")
    for clip in np.unique(val_clips):
        mask = val_clips == clip
        accuracy = np.mean(predictions[mask] == y_val[mask])
        print(f"{clip}: {accuracy:.4f} ({int(mask.sum())} sequences)")

    print("\nTraining pipeline completed.")
    print("Remember: these results are internal temporal validation,")
    print("not proof of real-world driver-drowsiness accuracy.")


if __name__ == "__main__":
    main()
