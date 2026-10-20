"""
Extracts per-frame features (EAR, MAR, head pitch/yaw/roll, gaze) from one
or more recorded video clips -- no labels attached yet, that's a separate
step (see label_clips.py). Each row is tagged with which clip it came from,
so labeling later just needs to map clip name -> label, no timestamp math.

Usage (one clip):
    python extract_features.py normal.mp4

Usage (all your clips at once, recommended):
    python extract_features.py normal.mp4 eyes_closed.mp4 yawn.mp4 head_nod.mp4 phone_glance.mp4 head_tilt.mp4 --out raw_features.csv

Output: raw_features.csv, one row per frame across ALL given clips, columns:
    source_clip, frame, time_sec, face_detected, ear, mar,
    head_pitch, head_yaw, head_roll, gaze_vertical, gaze_horizontal
"""

import argparse
import csv
import os

import cv2

from landmarks import LandmarkExtractor


def extract_one(video_path: str, extractor: LandmarkExtractor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clip_name = os.path.basename(video_path)

    rows = []
    frame_idx = 0
    detected_count = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        feats = extractor.process(frame_rgb)
        time_sec = frame_idx / fps

        if feats is None:
            rows.append([clip_name, frame_idx, f"{time_sec:.3f}", False, "", "", "", "", "", "", ""])
        else:
            detected_count += 1
            rows.append([
                clip_name, frame_idx, f"{time_sec:.3f}", True,
                f"{feats['ear']:.4f}", f"{feats['mar']:.4f}",
                f"{feats['head_pitch']:.2f}", f"{feats['head_yaw']:.2f}", f"{feats['head_roll']:.2f}",
                f"{feats['gaze_vertical']:.3f}", f"{feats['gaze_horizontal']:.3f}",
            ])
        frame_idx += 1

    cap.release()
    print(f"  {clip_name}: {frame_idx} frames, face detected in "
          f"{detected_count}/{frame_idx} ({100*detected_count/max(frame_idx,1):.1f}%)")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", help="One or more recorded video clip paths")
    parser.add_argument("--out", default="raw_features.csv")
    args = parser.parse_args()

    extractor = LandmarkExtractor()
    all_rows = []
    for path in args.videos:
        all_rows.extend(extract_one(path, extractor))
    extractor.close()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_clip", "frame", "time_sec", "face_detected", "ear", "mar",
                          "head_pitch", "head_yaw", "head_roll", "gaze_vertical", "gaze_horizontal"])
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} total rows across {len(args.videos)} clip(s) to {args.out}")


if __name__ == "__main__":
    main()