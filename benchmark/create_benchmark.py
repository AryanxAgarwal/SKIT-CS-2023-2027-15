
"""
Create a deterministic clip-level hold-out benchmark.

The benchmark is split by source clip instead of individual samples
to reduce data leakage between development and hold-out data.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DATASET_PATH = Path("dataset.npz")
OUTPUT_DIR = Path("benchmark")
MANIFEST_PATH = OUTPUT_DIR / "benchmark_manifest.csv"
HASH_PATH = OUTPUT_DIR / "benchmark_hashes.json"

TEST_CLIPS = {
    "normal.mp4",
    "yawn.mp4",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("=" * 60)
    print("BENCHMARK DATA BATCH CREATION")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(DATASET_PATH, allow_pickle=True) as data:
        X = data["X"]
        y = data["y"]
        clip_source = data["clip_source"]

    if len(X) != len(y) or len(X) != len(clip_source):
        raise ValueError("X, y and clip_source must contain the same number of samples.")

    unique_clips = sorted(set(str(x) for x in clip_source))

    missing_test_clips = TEST_CLIPS - set(unique_clips)

    if missing_test_clips:
        raise ValueError(
            f"Configured hold-out clips were not found: {sorted(missing_test_clips)}"
        )

    rows = []

    for index in range(len(X)):
        clip = str(clip_source[index])
        split = "holdout_test" if clip in TEST_CLIPS else "development"

        rows.append({
            "sample_index": index,
            "clip_source": clip,
            "label": int(y[index]),
            "split": split,
        })

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_index", "clip_source", "label", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    hashes = {
        DATASET_PATH.as_posix(): sha256_file(DATASET_PATH),
        MANIFEST_PATH.as_posix(): sha256_file(MANIFEST_PATH),
    }

    HASH_PATH.write_text(
        json.dumps(hashes, indent=2),
        encoding="utf-8"
    )

    development = [r for r in rows if r["split"] == "development"]
    holdout = [r for r in rows if r["split"] == "holdout_test"]

    print("Dataset samples:", len(X))
    print("Input shape:", X.shape)
    print("Development samples:", len(development))
    print("Hold-out test samples:", len(holdout))
    print("Hold-out clips:", ", ".join(sorted(TEST_CLIPS)))
    print("Dataset SHA-256:", hashes[DATASET_PATH.as_posix()])
    print("Manifest SHA-256:", hashes[MANIFEST_PATH.as_posix()])
    print("Hash report:", HASH_PATH)
    print("STATUS: BENCHMARK CREATED")


if __name__ == "__main__":
    main()
