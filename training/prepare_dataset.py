
"""
Prepare labeled driver-monitoring features for model training.

- Preserves every frame, including frames without a detected face.
- Separates face visibility from the drowsiness target.
- Validates labels and feature values.
- Produces eight features in a fixed order.
"""

import argparse
from pathlib import Path

import pandas as pd


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

REQUIRED_COLUMNS = {
    "source_clip",
    "frame",
    "face_detected",
    "drowsy",
    *FEATURE_COLUMNS[:-1],
}


def normalize_face_detected(value):
    """Convert common CSV values to 0 or 1."""
    if pd.isna(value):
        raise ValueError("face_detected contains a missing value.")

    if isinstance(value, bool):
        return int(value)

    value = str(value).strip().lower()

    if value in {"true", "1", "yes"}:
        return 1

    if value in {"false", "0", "no"}:
        return 0

    raise ValueError(
        f"Invalid face_detected value: {value!r}"
    )


def prepare_dataset(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if df.empty:
        raise ValueError("The input dataset is empty.")

    original_rows = len(df)

    # Normalize clip names and validate frame numbers.
    df["source_clip"] = (
        df["source_clip"]
        .astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.rsplit("/", n=1)
        .str[-1]
    )

    if df["source_clip"].eq("").any():
        raise ValueError("Empty source_clip values found.")

    df["frame"] = pd.to_numeric(
        df["frame"], errors="raise"
    )

    if df["frame"].isna().any():
        raise ValueError("Missing frame numbers found.")

    if (df["frame"] < 0).any():
        raise ValueError("Negative frame numbers found.")

    if (df["frame"] % 1 != 0).any():
        raise ValueError("Frame numbers must be integers.")

    df["frame"] = df["frame"].astype("int64")

    # A clip must not contain duplicate frame numbers.
    duplicate_frames = df.duplicated(
        subset=["source_clip", "frame"]
    )

    if duplicate_frames.any():
        examples = df.loc[
            duplicate_frames, ["source_clip", "frame"]
        ].head(5)

        raise ValueError(
            "Duplicate clip/frame pairs found:\n"
            + examples.to_string(index=False)
        )

    # Normalize visibility independently of drowsiness.
    df["face_detected"] = df["face_detected"].map(
        normalize_face_detected
    )

    # Validate the target labels.
    df["drowsy"] = pd.to_numeric(
        df["drowsy"], errors="raise"
    )

    if df["drowsy"].isna().any():
        raise ValueError(
            "Missing drowsy labels found. Label these clips "
            "before training; do not guess their labels."
        )

    if not df["drowsy"].isin([0, 1]).all():
        raise ValueError(
            "The drowsy column must contain only 0 or 1."
        )

    df["drowsy"] = df["drowsy"].astype("int64")

    # Convert measured features to numeric values.
    measurement_columns = FEATURE_COLUMNS[:-1]

    for column in measurement_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Missing measurements are acceptable only when no face
    # was detected. Zero is a placeholder, not a measurement.
    detected = df["face_detected"].eq(1)
    not_detected = ~detected

    missing_detected = (
        df.loc[detected, measurement_columns]
        .isna()
        .any(axis=1)
    )

    if missing_detected.any():
        bad_rows = df.loc[
            detected
            & df[measurement_columns].isna().any(axis=1),
            ["source_clip", "frame"],
        ].head(10)

        raise ValueError(
            "Missing facial measurements in frames marked "
            "face_detected=True. Review these rows:\n"
            + bad_rows.to_string(index=False)
        )

    # Reject infinity and invalid measurements.
    for column in measurement_columns:
        values = df[column]

        if values.loc[detected].isin(
            [float("inf"), float("-inf")]
        ).any():
            raise ValueError(
                f"Infinite values found in {column}."
            )

    # Only fill missing measurements for frames with no face.
    df.loc[not_detected, measurement_columns] = (
        df.loc[not_detected, measurement_columns].fillna(0.0)
    )

    # Explicitly check that all model inputs are finite.
    numeric_features = df[measurement_columns].to_numpy(
        dtype=float
    )

    import numpy as np

    if not np.isfinite(numeric_features).all():
        raise ValueError(
            "Non-finite feature values remain after cleaning."
        )

    # Include face_detected as the eighth model feature.
    df["face_detected"] = df["face_detected"].astype("float32")

    # Keep the original behavior and visibility labels when present.
    output_columns = [
        "source_clip",
        "frame",
    ]

    if "time_sec" in df.columns:
        df["time_sec"] = pd.to_numeric(
            df["time_sec"], errors="coerce"
        )
        output_columns.append("time_sec")

    if "behavior_label" in df.columns:
        output_columns.append("behavior_label")

    if "visibility_label" in df.columns:
        output_columns.append("visibility_label")

    output_columns += FEATURE_COLUMNS
    output_columns.append("drowsy")

    df = df.sort_values(
        ["source_clip", "frame"]
    ).reset_index(drop=True)

    prepared = df[output_columns].copy()

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    prepared.to_csv(output_path, index=False)

    no_face_count = int(
        (prepared["face_detected"] == 0).sum()
    )

    print("Dataset preparation successful.")
    print(f"Input rows: {original_rows}")
    print(f"Output rows: {len(prepared)}")
    print(f"Rows removed: {original_rows - len(prepared)}")
    print(f"Frames without a detected face: {no_face_count}")
    print(f"Number of model features: {len(FEATURE_COLUMNS)}")
    print(f"Output file: {output_path}")
    print("\nDrowsiness-label counts:")
    print(prepared["drowsy"].value_counts().sort_index())
    print("\nFeature columns:")
    print(FEATURE_COLUMNS)

    if len(prepared) != original_rows:
        raise RuntimeError(
            "Row count changed unexpectedly during preparation."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Validate and prepare the labeled dataset."
    )

    parser.add_argument(
        "--in",
        dest="input_path",
        default="ayush-labelling/labeled_features.csv",
        help="Input labeled CSV",
    )

    parser.add_argument(
        "--out",
        dest="output_path",
        default="training/prepared_dataset.csv",
        help="Output prepared CSV",
    )

    args = parser.parse_args()

    prepare_dataset(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
