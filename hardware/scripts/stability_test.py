#!/usr/bin/env python3
"""
stability_test.py

Sprint 1 - Baseline Integration & Data Flow Verification
Runs a sustained camera capture session and logs FPS, CPU %, RAM %, and
CPU temperature at a fixed interval, so we can prove the raw video stream
is stable over time (not just for a few seconds).

Usage:
    python3 stability_test.py --config ../configs/config.yaml --duration 1800

Output:
    Writes a log file to ../logs/stability_log_<YYYY-MM-DD_HHMM>.txt
"""

import argparse
import datetime
import os
import time

import cv2
import psutil
import yaml

LOG_INTERVAL_SEC = 10  # how often to record a metrics line


def load_config(path):
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[stability_test] config file not found at {path}, using CLI/defaults")
        return {}


def read_cpu_temp():
    """Reads CPU temp on a Raspberry Pi. Returns None if unavailable (e.g. Jetson, dev machine)."""
    thermal_path = "/sys/class/thermal/thermal_zone0/temp"
    try:
        with open(thermal_path, "r") as f:
            millideg = int(f.read().strip())
        return millideg / 1000.0
    except (FileNotFoundError, ValueError):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Sustained camera stability test")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--index", type=int, default=None, help="Camera index (overrides config)")
    parser.add_argument("--width", type=int, default=None, help="Capture width (overrides config)")
    parser.add_argument("--height", type=int, default=None, help="Capture height (overrides config)")
    parser.add_argument("--duration", type=int, default=1800,
                         help="Test duration in seconds (default 1800 = 30 min)")
    parser.add_argument("--log-dir", default="../logs", help="Directory to write the log file to")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    index = args.index if args.index is not None else cfg.get("camera_index", 0)
    width = args.width if args.width is not None else cfg.get("width", 640)
    height = args.height if args.height is not None else cfg.get("height", 480)

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[stability_test] FAILED to open camera at index {index}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    log_path = os.path.join(args.log_dir, f"stability_log_{stamp}.txt")

    print(f"[stability_test] Running for {args.duration}s, logging to {log_path}")

    failed_reads = 0
    frames_this_window = 0
    window_start = time.time()
    test_start = time.time()

    with open(log_path, "w") as log_file:
        header = (
            f"Sprint 1 Stability Test\n"
            f"Started: {datetime.datetime.now().isoformat()}\n"
            f"Camera index: {index}, target resolution: {width}x{height}\n"
            f"Planned duration: {args.duration}s, log interval: {LOG_INTERVAL_SEC}s\n"
            f"{'-'*70}\n"
            f"{'elapsed_s':>10} {'fps':>8} {'cpu_%':>8} {'ram_%':>8} {'temp_C':>8} {'failed_reads':>13}\n"
        )
        print(header)
        log_file.write(header)
        log_file.flush()

        while time.time() - test_start < args.duration:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
            else:
                frames_this_window += 1

            if time.time() - window_start >= LOG_INTERVAL_SEC:
                elapsed = time.time() - test_start
                window_elapsed = time.time() - window_start
                fps = frames_this_window / window_elapsed if window_elapsed > 0 else 0
                cpu_pct = psutil.cpu_percent(interval=None)
                ram_pct = psutil.virtual_memory().percent
                temp = read_cpu_temp()
                temp_str = f"{temp:.1f}" if temp is not None else "n/a"

                line = (
                    f"{elapsed:10.1f} {fps:8.2f} {cpu_pct:8.1f} {ram_pct:8.1f} "
                    f"{temp_str:>8} {failed_reads:13d}\n"
                )
                print(line, end="")
                log_file.write(line)
                log_file.flush()

                frames_this_window = 0
                window_start = time.time()

    cap.release()

    summary = (
        f"\n{'-'*70}\n"
        f"Test finished: {datetime.datetime.now().isoformat()}\n"
        f"Total failed reads: {failed_reads}\n"
    )
    print(summary)
    with open(log_path, "a") as log_file:
        log_file.write(summary)

    print(f"[stability_test] Done. Log written to {log_path}")
    print("[stability_test] Copy the average/peak values from this log into "
          "docs/raw_stream_verification_report.md")


if __name__ == "__main__":
    main()
