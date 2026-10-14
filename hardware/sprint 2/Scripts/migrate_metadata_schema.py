#!/usr/bin/env python3
"""
migrate_metadata_schema.py

Sprint 2 (Off-Sprint Bridging Task) - Pipeline Integration & Evaluation

Sprint 1's metadata.csv schema was fixed at:
    frame_id, filename, timestamp, width, height, fps_at_capture

If Sprint 2 preprocessing needs additional columns (e.g. a landmark
reference to join against Aryan's facial landmark output, or a
preprocessing-status flag), this script adds them in place, without
hand-editing the CSV and risking a mismatch with pipeline_health_check.py's
expected schema.

This is the tool referenced in docs/pipeline_maintenance_log.md — every
time it's run, add a row to that log's change table.

Usage:
    # Add a new column called 'landmark_ref', default value blank
    python3 migrate_metadata_schema.py --dataset ../../sprint1/dataset \
        --add-column landmark_ref --default ""

    # Add a schema_version column, defaulting existing rows to v1
    python3 migrate_metadata_schema.py --dataset ../../sprint1/dataset \
        --add-column schema_version --default v1
"""

import argparse
import csv
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Add a column to metadata.csv safely")
    parser.add_argument("--dataset", required=True,
                         help="Path to the dataset directory containing metadata.csv")
    parser.add_argument("--add-column", required=True, help="Name of the new column")
    parser.add_argument("--default", default="", help="Default value for existing rows")
    parser.add_argument("--no-backup", action="store_true",
                         help="Skip writing a metadata.csv.bak backup before migrating")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = os.path.join(args.dataset, "metadata.csv")

    if not os.path.isfile(metadata_path):
        print(f"[migrate_metadata_schema] No metadata.csv found at {metadata_path}")
        return

    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if args.add_column in fieldnames:
        print(f"[migrate_metadata_schema] Column '{args.add_column}' already exists. No changes made.")
        return

    if not args.no_backup:
        backup_path = metadata_path + ".bak"
        shutil.copy2(metadata_path, backup_path)
        print(f"[migrate_metadata_schema] Backup written to {backup_path}")

    new_fieldnames = fieldnames + [args.add_column]
    for row in rows:
        row[args.add_column] = args.default

    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[migrate_metadata_schema] Added column '{args.add_column}' "
          f"(default='{args.default}') to {len(rows)} rows.")
    print("[migrate_metadata_schema] IMPORTANT: update EXPECTED_COLUMNS in "
          "pipeline_health_check.py to match, and log this change in "
          "docs/pipeline_maintenance_log.md.")


if __name__ == "__main__":
    main()
