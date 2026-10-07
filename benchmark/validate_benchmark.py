"""
Validate the integrity of the benchmark manifest.
"""

import csv
from pathlib import Path

import numpy as np


DATASET_PATH = Path("dataset.npz")
MANIFEST_PATH = Path("benchmark/benchmark_manifest.csv")


def main():
    print("=" * 60)
    print("BENCHMARK INTEGRITY VALIDATION")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError("dataset.npz not found.")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "benchmark_manifest.csv not found. "
            "Run create_benchmark.py first."
        )

    data = np.load(DATASET_PATH, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    clips = data["clip_source"]

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    errors = []

    # Check dataset and manifest sample count
    if len(rows) != len(X):
        errors.append(
            f"Manifest contains {len(rows)} rows but "
            f"dataset contains {len(X)} samples."
        )

    # Check every manifest entry against original dataset
    for row in rows:
        index = int(row["sample_index"])

        if index < 0 or index >= len(X):
            errors.append(f"Invalid sample index: {index}")
            continue

        if str(clips[index]) != row["clip_source"]:
            errors.append(
                f"Clip mismatch at sample {index}: "
                f"{clips[index]} != {row['clip_source']}"
            )

        if int(y[index]) != int(row["label"]):
            errors.append(
                f"Label mismatch at sample {index}: "
                f"{y[index]} != {row['label']}"
            )

    # Check valid split names
    valid_splits = {"development", "holdout_test"}

    for row in rows:
        if row["split"] not in valid_splits:
            errors.append(
                f"Invalid split '{row['split']}' "
                f"at sample {row['sample_index']}"
            )

    # Hold-out must contain both classes
    holdout_labels = {
        int(row["label"])
        for row in rows
        if row["split"] == "holdout_test"
    }

    if holdout_labels != {0, 1}:
        errors.append(
            f"Hold-out benchmark must contain labels 0 and 1. "
            f"Found: {sorted(holdout_labels)}"
        )

    # A source clip must never appear in multiple splits
    clip_splits = {}

    for row in rows:
        clip = row["clip_source"]
        clip_splits.setdefault(clip, set()).add(row["split"])

    for clip, splits in clip_splits.items():
        if len(splits) != 1:
            errors.append(
                f"DATA LEAKAGE: clip {clip} appears "
                f"in multiple splits: {splits}"
            )

    print(f"Dataset samples: {len(X)}")
    print(f"Manifest rows:    {len(rows)}")
    print(f"Input shape:      {X.shape}")
    print()

    print("Hold-out clips:")

    holdout_clips = sorted(
        {
            row["clip_source"]
            for row in rows
            if row["split"] == "holdout_test"
        }
    )

    for clip in holdout_clips:
        print(" ", clip)

    print()

    if errors:
        print("STATUS: FAIL")
        print()
        print("Errors:")

        for error in errors:
            print(" -", error)

        raise SystemExit(1)

    print("Duplicate sample indices: 0")
    print("Label mismatches: 0")
    print("Split leakage: 0")
    print("Hold-out classes: 0 and 1")
    print()
    print("STATUS: PASS")


if __name__ == "__main__":
    main()