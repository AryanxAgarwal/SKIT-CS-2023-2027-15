#!/usr/bin/env python3
"""
camera_test.py

Sprint 1 - Baseline Integration & Data Flow Verification
Detects the configured camera, opens it, and displays a live preview window
so you can confirm the physical setup works before running the longer
stability test.

Usage:
    python3 camera_test.py --config ../configs/config.yaml
    python3 camera_test.py --index 0 --width 640 --height 480

Press 'q' in the preview window to quit.
"""

import argparse
import sys
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
        print(f"[camera_test] config file not found at {path}, using CLI/defaults")
        return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Detect camera and show live stream")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--index", type=int, default=None, help="Camera index (overrides config)")
    parser.add_argument("--width", type=int, default=None, help="Capture width (overrides config)")
    parser.add_argument("--height", type=int, default=None, help="Capture height (overrides config)")
    parser.add_argument("--headless", action="store_true",
                         help="Don't open a preview window; just confirm frames are readable")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    index = args.index if args.index is not None else cfg.get("camera_index", 0)
    width = args.width if args.width is not None else cfg.get("width", 640)
    height = args.height if args.height is not None else cfg.get("height", 480)

    print(f"[camera_test] Opening camera index={index}, target resolution={width}x{height}")

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[camera_test] FAILED to open camera at index {index}.")
        print("[camera_test] Check `ls /dev/video*` and that no other process has it open.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[camera_test] Camera opened. Actual resolution reported: {actual_w}x{actual_h}")

    ok, frame = cap.read()
    if not ok:
        print("[camera_test] FAILED to read a frame from the camera.")
        cap.release()
        sys.exit(1)

    print(f"[camera_test] Successfully read a frame with shape {frame.shape}")

    if args.headless:
        print("[camera_test] Headless mode - skipping preview window. PASS.")
        cap.release()
        return

    print("[camera_test] Opening preview window. Press 'q' to quit.")
    frame_count = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[camera_test] Frame read failed mid-stream.")
            break
        frame_count += 1
        cv2.imshow("camera_test - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.time() - start
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"[camera_test] Preview ended. ~{fps:.1f} fps over {elapsed:.1f}s ({frame_count} frames)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
