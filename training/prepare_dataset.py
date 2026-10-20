
"""
Prepare labeled driver-monitoring features for CNN training.
"""

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "ayush-labelling" / "labeled_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "training" / "prepared_features.csv"

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


def prepare_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = (
        ["source_clip", "frame", "behavior_label", TARGET_COLUMN]
        + FEATURE_COLUMNS
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    original_rows = len(df)

    # Convert the target into numeric binary labels.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(
        {True: 1, False: 0}
    )

    # Remove rows without a detected face or usable features.
    df = df[df["face_detected"] == True].copy()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Ensure all model inputs are numeric and finite.
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(subset=FEATURE_COLUMNS)

    # Reject invalid target values.
    df = df[df[TARGET_COLUMN].isin([0, 1])].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int32")

    if df.empty:
        raise ValueError("No usable training rows remain.")

    # Preserve clip/frame information for sequence creation later.
    output_columns = [
        "source_clip",
        "frame",
        "behavior_label",
    ] + FEATURE_COLUMNS + [TARGET_COLUMN]

    prepared = df[output_columns].copy()

    # Verify that all model features are finite.
    if not np.isfinite(
       prepared[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
       raise ValueError("Non-finite feature values remain.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(OUTPUT_PATH, index=False)

    print("Dataset preparation completed.")
    print(f"Input rows: {original_rows}")
    print(f"Prepared rows: {len(prepared)}")
    print(f"Rows removed: {original_rows - len(prepared)}")
    print(f"Output file: {OUTPUT_PATH}")
    print("\nTarget distribution:")
    print(prepared[TARGET_COLUMN].value_counts().sort_index())
    print("\nOutput shape:", prepared.shape)
    print("Missing values remaining:", int(prepared.isnull().sum().sum()))


if __name__ == "__main__":
    prepare_dataset()
