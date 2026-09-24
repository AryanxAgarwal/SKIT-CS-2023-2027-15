#!/usr/bin/env python3
"""
save_frames_metadata.py

Sprint 1 - Baseline Integration & Data Flow Verification
Captures N frames from the camera, saves them as JPEGs under
dataset/frames/, and writes a matching metadata.csv row for each frame.

This defines the frame naming + metadata schema that Aryan and Chiranshu
consume downstream (see docs/data_flow_verification_report.md).

Usage:
    python3 save_frames_metadata.py --config ../configs/config.yaml --num-frames 50
    python3 save_frames_metadata.py --out ../../dataset --num-frames 20
"""

import argparse
import csv
import datetime
import os
import time

import cv2
import yaml


def load_config(path):
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[save_frames_metadata] config file not found at {path}, using CLI/defaults")
        return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Save frames + metadata.csv")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--index", type=int, default=None, help="Camera index (overrides config)")
    parser.add_argument("--width", type=int, default=None, help="Capture width (overrides config)")
    parser.add_argument("--height", type=int, default=None, help="Capture height (overrides config)")
    parser.add_argument("--out", default=None,
                         help="Output dataset directory (overrides config's save_path)")
    parser.add_argument("--num-frames", type=int, default=50,
                         help="Number of frames to capture and save")
    parser.add_argument("--interval-ms", type=int, default=100,
                         help="Delay between saved frames, in milliseconds")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    index = args.index if args.index is not None else cfg.get("camera_index", 0)
    width = args.width if args.width is not None else cfg.get("width", 640)
    height = args.height if args.height is not None else cfg.get("height", 480)
    out_dir = args.out or cfg.get("save_path", "../../dataset")

    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, "metadata.csv")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[save_frames_metadata] FAILED to open camera at index {index}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"[save_frames_metadata] Capturing {args.num_frames} frames to {out_dir}")

    write_header = not os.path.exists(metadata_path)
    with open(metadata_path, "a", newline="") as meta_file:
        writer = csv.writer(meta_file)
        if write_header:
            writer.writerow(["frame_id", "filename", "timestamp", "width", "height", "fps_at_capture"])

        saved = 0
        prev_time = time.time()
        while saved < args.num_frames:
            ok, frame = cap.read()
            if not ok:
                print("[save_frames_metadata] Frame read failed, retrying...")
                continue

            now = time.time()
            instant_fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            saved += 1
            filename = f"frame_{saved:06d}.jpg"
            frame_path = os.path.join(frames_dir, filename)
            cv2.imwrite(frame_path, frame)

            h, w = frame.shape[:2]
            timestamp = datetime.datetime.now().isoformat()
            writer.writerow([saved, os.path.join("frames", filename), timestamp, w, h, f"{instant_fps:.2f}"])

            time.sleep(args.interval_ms / 1000.0)

    cap.release()
    print(f"[save_frames_metadata] Saved {saved} frames to {frames_dir}")
    print(f"[save_frames_metadata] Metadata written to {metadata_path}")
    print("[save_frames_metadata] Copy a handful of rows into logs/metadata_sample.csv "
          "(do NOT commit the full dataset/ directory)")


if __name__ == "__main__":
    main()
