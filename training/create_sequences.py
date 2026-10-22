
"""
Create consecutive 30-frame sequences for CNN training.

Sequences never cross clip boundaries or frame-number gaps.
Each frame contains 8 features.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "training" / "prepared_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "training"

SEQUENCE_LENGTH = 30
STRIDE = 1

FEATURE_COLUMNS = [
    "ear",
    "mar",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "gaze_vertical",
    "gaze_horizontal",
    "face_detected",
]

TARGET_COLUMN = "drowsy"
CLIP_COLUMN = "source_clip"
FRAME_COLUMN = "frame"


def create_sequences():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}\n"
            "Run: python training\\prepare_dataset.py"
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        CLIP_COLUMN,
        FRAME_COLUMN,
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("The prepared dataset is empty.")

    # Validate numeric columns.
    numeric_columns = [FRAME_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[required].isnull().any().any():
        raise ValueError(
            "Missing values detected in required columns. "
            "Fix the prepared dataset before continuing."
        )

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite numeric values detected.")

    # Labels must be binary.
    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("The drowsy column must contain only 0 and 1.")

    # Frame numbers must be whole numbers.
    frame_values = df[FRAME_COLUMN].to_numpy(dtype=np.float64)
    if not np.equal(frame_values, np.floor(frame_values)).all():
        raise ValueError("Frame numbers must be integers.")

    df[FRAME_COLUMN] = df[FRAME_COLUMN].astype(np.int64)
    df[CLIP_COLUMN] = df[CLIP_COLUMN].astype(str)

    if not df[CLIP_COLUMN].str.strip().ne("").all():
        raise ValueError("Empty clip names detected.")

    # Sort chronologically within each video.
    df = df.sort_values(
        [CLIP_COLUMN, FRAME_COLUMN],
        kind="stable",
    ).reset_index(drop=True)

    # Reject duplicate frame numbers within the same clip.
    duplicate_mask = df.duplicated(
        subset=[CLIP_COLUMN, FRAME_COLUMN]
    )
    if duplicate_mask.any():
        examples = df.loc[
            duplicate_mask, [CLIP_COLUMN, FRAME_COLUMN]
        ].head(5)

        raise ValueError(
            "Duplicate frame numbers detected:\n"
            + examples.to_string(index=False)
        )

    sequences = []
    labels = []
    sequence_clips = []
    skipped_gap_windows = 0

    # Process each clip independently.
    for clip_name, clip_df in df.groupby(CLIP_COLUMN, sort=True):
        clip_df = clip_df.reset_index(drop=True)

        features = clip_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        targets = clip_df[TARGET_COLUMN].to_numpy(dtype=np.int32)
        frames = clip_df[FRAME_COLUMN].to_numpy(dtype=np.int64)

        if len(clip_df) < SEQUENCE_LENGTH:
            print(
                f"Skipping {clip_name}: only {len(clip_df)} frames."
            )
            continue

        # Sliding windows of 30 frames, with stride 1.
        for start in range(
            0,
            len(clip_df) - SEQUENCE_LENGTH + 1,
            STRIDE,
        ):
            end = start + SEQUENCE_LENGTH
            frame_window = frames[start:end]

            # Reject windows containing frame-number gaps.
            if not np.all(np.diff(frame_window) == 1):
                skipped_gap_windows += 1
                continue

            target_window = targets[start:end]

            # Positive when at least half the frames are labeled drowsy.
            sequence_label = int(np.mean(target_window) >= 0.5)

            sequences.append(features[start:end])
            labels.append(sequence_label)
            sequence_clips.append(clip_name)

    if not sequences:
        raise ValueError(
            "No valid sequences created. Check clip lengths "
            "and frame continuity."
        )

    X = np.stack(sequences).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    clips = np.asarray(sequence_clips, dtype=str)

    # Verify CNN-compatible shapes and valid values.
    expected_shape = (SEQUENCE_LENGTH, len(FEATURE_COLUMNS))

    if X.ndim != 3 or X.shape[1:] != expected_shape:
        raise ValueError(
            f"Incorrect X shape: {X.shape}; expected "
            f"(samples, {expected_shape[0]}, {expected_shape[1]})."
        )

    if len(X) != len(y) or len(y) != len(clips):
        raise ValueError("X, y, and clip identifiers have different lengths.")

    if not np.isfinite(X).all():
        raise ValueError("Generated sequences contain non-finite values.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_sequences.npy", X)
    np.save(OUTPUT_DIR / "y_sequences.npy", y)
    np.save(OUTPUT_DIR / "sequence_clips.npy", clips)

    unique_labels, label_counts = np.unique(y, return_counts=True)

    print("\nSequence creation successful.")
    print("Input file:", INPUT_PATH)
    print("Input frames:", len(df))
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Features per frame:", len(FEATURE_COLUMNS))
    print("Skipped windows due to frame gaps:", skipped_gap_windows)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Clip IDs shape:", clips.shape)
    print("Clips represented:", len(np.unique(clips)))
    print(
        "Label distribution:",
        dict(zip(unique_labels.tolist(), label_counts.tolist())),
    )
    print("Finite feature values:", np.isfinite(X).all())
    print("Saved files:")
    print(OUTPUT_DIR / "X_sequences.npy")
    print(OUTPUT_DIR / "y_sequences.npy")
    print(OUTPUT_DIR / "sequence_clips.npy")


if __name__ == "__main__":
    create_sequences()
