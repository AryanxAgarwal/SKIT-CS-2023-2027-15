#!/usr/bin/env python3
"""
pipeline_health_check.py

Sprint 2 (Off-Sprint Bridging Task) - Pipeline Integration & Evaluation
Validates that the Sprint 1 data pipeline (frames/ + metadata.csv) is still
internally consistent, so downstream Sprint 2 preprocessing can trust it.

Checks performed:
  1. Every row in metadata.csv points to a file that actually exists.
  2. Every file in frames/ has a corresponding metadata.csv row.
  3. Each image is readable (not corrupt) and its actual dimensions match
     the width/height recorded in metadata.csv.
  4. metadata.csv has the expected column schema.

Usage:
    python3 pipeline_health_check.py --dataset ../../sprint1/dataset
"""

import argparse
import csv
import os
import sys

import cv2

EXPECTED_COLUMNS = ["frame_id", "filename", "timestamp", "width", "height", "fps_at_capture"]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the frame + metadata pipeline")
    parser.add_argument("--dataset", required=True,
                         help="Path to the dataset directory containing frames/ and metadata.csv")
    parser.add_argument("--report", default=None,
                         help="Optional path to write a text report to (defaults to stdout only)")
    return parser.parse_args()


def main():
    args = parse_args()
    frames_dir = os.path.join(args.dataset, "frames")
    metadata_path = os.path.join(args.dataset, "metadata.csv")

    issues = []
    checked = 0

    if not os.path.isdir(frames_dir):
        issues.append(f"frames/ directory not found at {frames_dir}")
    if not os.path.isfile(metadata_path):
        issues.append(f"metadata.csv not found at {metadata_path}")

    if issues:
        _report(issues, checked, args.report)
        sys.exit(1)

    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        actual_columns = reader.fieldnames or []
        if actual_columns != EXPECTED_COLUMNS:
            issues.append(
                f"Schema mismatch. Expected {EXPECTED_COLUMNS}, found {actual_columns}"
            )

        metadata_files = set()
        for row in reader:
            checked += 1
            filename = row.get("filename", "")
            metadata_files.add(filename)
            frame_path = os.path.join(args.dataset, filename)

            if not os.path.isfile(frame_path):
                issues.append(f"[row {row.get('frame_id')}] missing file: {filename}")
                continue

            img = cv2.imread(frame_path)
            if img is None:
                issues.append(f"[row {row.get('frame_id')}] unreadable/corrupt image: {filename}")
                continue

            h, w = img.shape[:2]
            try:
                expected_w = int(row.get("width", -1))
                expected_h = int(row.get("height", -1))
            except ValueError:
                expected_w = expected_h = -1

            if (w, h) != (expected_w, expected_h):
                issues.append(
                    f"[row {row.get('frame_id')}] dimension mismatch for {filename}: "
                    f"metadata says {expected_w}x{expected_h}, actual is {w}x{h}"
                )

    # Reverse check: files on disk with no metadata row
    on_disk = set(os.path.join("frames", f) for f in os.listdir(frames_dir)) if os.path.isdir(frames_dir) else set()
    orphaned = on_disk - metadata_files
    for orphan in sorted(orphaned):
        issues.append(f"file on disk with no metadata.csv row: {orphan}")

    _report(issues, checked, args.report)
    sys.exit(1 if issues else 0)


def _report(issues, checked, report_path):
    lines = []
    lines.append("Pipeline Health Check")
    lines.append(f"Rows checked: {checked}")
    lines.append(f"Issues found: {len(issues)}")
    lines.append("-" * 60)
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
        lines.append("-" * 60)
        lines.append("RESULT: FAIL")
    else:
        lines.append("RESULT: PASS — pipeline is consistent")

    text = "\n".join(lines)
    print(text)
    if report_path:
        with open(report_path, "w") as f:
            f.write(text + "\n")
        print(f"\n[pipeline_health_check] Report written to {report_path}")


if __name__ == "__main__":
    main()
