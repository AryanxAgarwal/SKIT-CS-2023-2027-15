import os
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "raw_features.csv"
)

REQUIRED_COLUMNS = [
    "source_clip",
    "frame",
    "time_sec",
    "face_detected",
    "ear",
    "mar",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "gaze_vertical",
    "gaze_horizontal",
]


# ---------------------------------------------------------
# Dataset Integrity Tests
# ---------------------------------------------------------

def test_dataset_exists():
    assert os.path.exists(DATASET_PATH), (
        f"Dataset not found: {DATASET_PATH}"
    )


def test_required_columns():
    df = pd.read_csv(DATASET_PATH)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )


def test_missing_values():
    df = pd.read_csv(DATASET_PATH)

    missing_values = df.isnull().sum()

    print("\nMissing values:")
    print(missing_values)

    # Feature values can be missing when no face is detected.
    feature_columns = [
        "ear",
        "mar",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "gaze_vertical",
        "gaze_horizontal",
    ]

    # Check that missing feature values occur only
    # when face_detected == 0.
    for column in feature_columns:

        invalid_missing = df[
            (df["face_detected"] == 1) &
            (df[column].isnull())
        ]

        assert len(invalid_missing) == 0, (
            f"Column '{column}' has missing values "
            "even though a face was detected."
        )

    print(
        "\nMissing feature values are consistent with "
        "frames where no face was detected."
    )


def test_duplicate_rows():
    df = pd.read_csv(DATASET_PATH)

    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate rows: {duplicate_count}")

    assert duplicate_count == 0, (
        f"Dataset contains {duplicate_count} duplicate rows."
    )


def test_numeric_values():
    df = pd.read_csv(DATASET_PATH)

    numeric_columns = [
        "frame",
        "time_sec",
        "face_detected",
        "ear",
        "mar",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "gaze_vertical",
        "gaze_horizontal",
    ]

    for column in numeric_columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # For feature columns, NaN is allowed when no face
        # was detected.
        if column in [
            "ear",
            "mar",
            "head_pitch",
            "head_yaw",
            "head_roll",
            "gaze_vertical",
            "gaze_horizontal",
        ]:
            invalid_count = (
                converted.isnull() &
                (df["face_detected"] == 1)
            ).sum()

        else:
            invalid_count = converted.isnull().sum()

        print(
            f"{column}: {invalid_count} invalid values"
        )

        assert invalid_count == 0, (
            f"Column '{column}' contains invalid numeric values."
        )

def test_ear_mar_values():
    df = pd.read_csv(DATASET_PATH)

    # Only validate EAR and MAR for frames
    # where a face was successfully detected.
    detected_faces = df[df["face_detected"] == 1]

    assert (detected_faces["ear"] >= 0).all(), (
        "Negative EAR value detected in a frame "
        "where a face was detected."
    )

    assert (detected_faces["mar"] >= 0).all(), (
        "Negative MAR value detected in a frame "
        "where a face was detected."
    )

    print("\nEAR/MAR values are valid for detected faces.")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("========================================")
    print("DATASET INTEGRITY VALIDATION")
    print("========================================")

    df = pd.read_csv(DATASET_PATH)

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    test_dataset_exists()
    test_required_columns()
    test_missing_values()
    test_duplicate_rows()
    test_numeric_values()
    test_ear_mar_values()

    print("\n========================================")
    print("ALL DATASET INTEGRITY TESTS PASSED")
    print("========================================")