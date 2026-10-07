"""
Create a deterministic clip-level hold-out benchmark.

The benchmark is split by source clip instead of individual samples
to reduce data leakage between development and hold-out data.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np


DATASET_PATH = Path("dataset.npz")
OUTPUT_DIR = Path("benchmark")
MANIFEST_PATH = OUTPUT_DIR / "benchmark_manifest.csv"

# Hold-out clips.
# normal = class 0
# yawn = class 1
TEST_CLIPS = {
    "normal.mp4",
    "yawn.mp4",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def main():

    print("=" * 60)
    print("BENCHMARK DATA BATCH CREATION")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(
        DATASET_PATH,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]
    clip_source = data["clip_source"]

    # Basic dataset consistency check.
    if len(X) != len(y) or len(X) != len(clip_source):
        raise ValueError(
            "X, y and clip_source must contain "
            "the same number of samples."
        )

    unique_clips = sorted(
        set(str(x) for x in clip_source)
    )

    # Make sure configured test clips actually exist.
    missing_test_clips = (
        TEST_CLIPS - set(unique_clips)
    )

    if missing_test_clips:
        raise ValueError(
            "Configured hold-out clips were not found: "
            f"{sorted(missing_test_clips)}"
        )

    rows = []

    for index in range(len(X)):

        clip = str(clip_source[index])

        if clip in TEST_CLIPS:
            split = "holdout_test"
        else:
            split = "development"

        rows.append(
            {
                "sample_index": index,
                "clip_source": clip,
                "label": int(y[index]),
                "split": split,
            }
        )

    # Write benchmark manifest.
    with open(
        MANIFEST_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_index",
                "clip_source",
                "label",
                "split",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    development = [
        row
        for row in rows
        if row["split"] == "development"
    ]

    holdout = [
        row
        for row in rows
        if row["split"] == "holdout_test"
    ]

    print()
    print("Dataset:", DATASET_PATH)
    print("Total samples:", len(X))
    print("Input shape:", X.shape)
    print("Timesteps:", X.shape[1])
    print("Features per timestep:", X.shape[2])

    print()
    print("CLIP DISTRIBUTION")
    print("-" * 60)

    for clip in unique_clips:

        clip_rows = [
            row
            for row in rows
            if row["clip_source"] == clip
        ]

        labels = sorted(
            set(
                row["label"]
                for row in clip_rows
            )
        )

        split = clip_rows[0]["split"]

        print(
            f"{clip:<25}"
            f"samples={len(clip_rows):<4}"
            f"label={labels}"
            f" split={split}"
        )

    print()
    print("DEVELOPMENT SAMPLES:", len(development))
    print("HOLD-OUT TEST SAMPLES:", len(holdout))

    print()
    print("HOLD-OUT LABEL DISTRIBUTION")

    for label in sorted(
        set(row["label"] for row in holdout)
    ):

        count = sum(
            row["label"] == label
            for row in holdout
        )

        print(
            f"label {label}: {count}"
        )

    print()
    print("DATASET SHA-256:")
    print(sha256_file(DATASET_PATH))

    print()
    print("MANIFEST:")
    print(MANIFEST_PATH)

    print()
    print("STATUS: BENCHMARK CREATED")


if __name__ == "__main__":
    main()