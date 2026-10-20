"""
Takes raw_features.csv (from extract_features.py) and attaches labels
based on which clip each row came from -- a separate step/commit from
extraction itself.

Usage:
    python label_clips.py
    python label_clips.py --in raw_features.csv --out labeled_features.csv
"""

import argparse
import csv

# Map: filename (as it appears in source_clip) -> (behavior label, drowsy)
# drowsy: 1 = Drowsy, 0 = Alert
CLIP_LABELS = {
    "normal.mp4":             ("alert", 0),
    "eyes_closed.mp4":        ("microsleep", 1),
    "yawn.mp4":               ("yawn", 1),
    "head_nod.mp4":           ("head_nod", 1),
    "phone_glance.mp4":       ("phone_glance", 1),
    "head_tilt_left.mp4":     ("head_tilt", 1),
    "head_tilt_right.mp4":    ("head_tilt", 1),
}


def label(in_path: str, out_path: str):
    with open(in_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames + ["behavior_label", "drowsy"]

    unmatched = set()
    no_face_count = 0

    for row in rows:
        if row["face_detected"] == "False":
            # No face detected.
            row["behavior_label"] = "distracted_no_face"
            row["drowsy"] = 1
            no_face_count += 1
            continue

        clip = row["source_clip"]

        if clip in CLIP_LABELS:
            behavior, drowsy = CLIP_LABELS[clip]
        else:
            unmatched.add(clip)
            behavior, drowsy = "unknown", ""

        row["behavior_label"] = behavior
        row["drowsy"] = drowsy

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if unmatched:
        print(
            f"WARNING: these clip names in the CSV don't match CLIP_LABELS "
            f"and were labeled 'unknown': {unmatched}"
        )
        print(
            "Edit CLIP_LABELS in this script to match your actual filenames, "
            "then re-run."
        )

    print(
        f"{no_face_count} frames had no face detected "
        f"-> labeled 'distracted_no_face'"
    )

    counts = {}
    for row in rows:
        counts[row["behavior_label"]] = (
            counts.get(row["behavior_label"], 0) + 1
        )

    print(f"Labeled {len(rows)} rows: {counts}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="raw_features.csv")
    parser.add_argument(
        "--out",
        dest="out_path",
        default="labeled_features.csv"
    )

    args = parser.parse_args()
    label(args.in_path, args.out_path)