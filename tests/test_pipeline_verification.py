import os
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FEATURES_PATH = os.path.join(BASE_DIR, "raw_features.csv")
LABELS_PATH = os.path.join(
    BASE_DIR, "ayush-labelling", "labeled_features.csv"
)

BATCH_SIZE = 128

# Existing ground-truth mapping from label_clips.py
CLIP_LABELS = {
    "normal.mp4": ("alert", False),
    "eyes_closed.mp4": ("microsleep", True),
    "yawn.mp4": ("yawn", True),
    "head_nod.mp4": ("head_nod", True),
    "phone_glance.mp4": ("phone_glance", True),
    "head_tilt_left.mp4": ("head_tilt", True),
    "head_tilt_right.mp4": ("head_tilt", True),
}

FEATURE_COLUMNS = [
    "face_detected",
    "ear",
    "mar",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "gaze_vertical",
    "gaze_horizontal",
]

KEY_COLUMNS = ["source_clip", "frame"]


def test_pipeline_verification():
    assert os.path.exists(FEATURES_PATH), "Feature file not found."
    assert os.path.exists(LABELS_PATH), "Label file not found."

    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)

    print("\n========================================")
    print("SPRINT 1 PIPELINE VERIFICATION")
    print("========================================")

    print("Feature rows:", len(features))
    print("Label rows:", len(labels))

    # 1. Required columns
    required_features = KEY_COLUMNS + FEATURE_COLUMNS
    required_labels = KEY_COLUMNS + [
        "behavior_label",
        "drowsy",
    ] + FEATURE_COLUMNS

    for column in required_features:
        assert column in features.columns, (
            f"Missing feature column: {column}"
        )

    for column in required_labels:
        assert column in labels.columns, (
            f"Missing label column: {column}"
        )

    # 2. Check matching keys and duplicate records
    assert not features.duplicated(KEY_COLUMNS).any(), (
        "Duplicate feature keys found."
    )

    assert not labels.duplicated(KEY_COLUMNS).any(), (
        "Duplicate label keys found."
    )

    assert len(features) == len(labels), (
        "Feature and label row counts differ."
    )

    # 3. Match features to their corresponding labeled records
    matched = features.merge(
        labels,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_feature", "_label"),
        indicator=True,
        validate="one_to_one",
    )

    assert (matched["_merge"] == "both").all(), (
        "Some feature rows and labels do not match."
    )

    for column in FEATURE_COLUMNS:
        left = matched[f"{column}_feature"]
        right = matched[f"{column}_label"]

        if pd.api.types.is_numeric_dtype(left):
            equal = (
                left.eq(right) |
                (left.isna() & right.isna())
            )
        else:
            equal = left.eq(right) | (
                left.isna() & right.isna()
            )

        assert equal.all(), (
            f"Feature mismatch found in column: {column}"
        )

    print("\nFeature-to-label records matched.")

    # 4. Validate ground-truth labels
    for _, row in labels.iterrows():
        clip = row["source_clip"]
        behavior = row["behavior_label"]
        drowsy = row["drowsy"]

        assert clip in CLIP_LABELS, (
            f"Unknown source clip: {clip}"
        )

        if not bool(row["face_detected"]):
            expected_behavior = "distracted_no_face"
            expected_drowsy = True
        else:
            expected_behavior, expected_drowsy = CLIP_LABELS[clip]

        assert behavior == expected_behavior, (
            f"Incorrect behavior label for {clip}, "
            f"frame {row['frame']}: {behavior}"
        )

        assert bool(drowsy) == expected_drowsy, (
            f"Incorrect drowsy label for {clip}, "
            f"frame {row['frame']}: {drowsy}"
        )

    print("Ground-truth labels verified.")

    # 5. Verify records in batches
    verified_rows = 0
    batch_count = 0

    for start in range(0, len(features), BATCH_SIZE):
        feature_batch = features.iloc[
            start:start + BATCH_SIZE
        ]

        batch_keys = pd.MultiIndex.from_frame(
            feature_batch[KEY_COLUMNS]
        )

        label_keys = pd.MultiIndex.from_frame(
            labels[KEY_COLUMNS]
        )

        assert batch_keys.isin(label_keys).all(), (
            f"Unmatched records in batch {batch_count + 1}."
        )

        batch_count += 1
        verified_rows += len(feature_batch)

        print(
            f"Batch {batch_count}: "
            f"{len(feature_batch)} rows verified"
        )

    assert verified_rows == len(features)

    print("\nTotal batches:", batch_count)
    print("Total rows verified:", verified_rows)

    print("\n========================================")
    print("PIPELINE VERIFICATION PASSED")
    print("========================================")


if __name__ == "__main__":
    test_pipeline_verification()