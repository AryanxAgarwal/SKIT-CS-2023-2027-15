#!/usr/bin/env python3
"""
concurrent_load_test.py

Sprint 2 (Off-Sprint Bridging Task) - Pipeline Integration & Evaluation

resource_monitor.py measures ONE workload at a time. But Sprint 3's control
loop ("Threading & Parallel Processing Architecture", "Real-Time Computer
Vision & Model Inference Link") will need the board to run camera capture
AND model inference/preprocessing at the same time. This script answers the
question resource_monitor.py can't: what happens to CPU/RAM/temp when both
run concurrently, not sequentially?

It starts a capture-side process (defaults to Sprint 1's camera_test.py in
headless mode) and a preprocessing-side command at the same time, and logs
system-wide resource use for the duration both are running.

Usage:
    python3 concurrent_load_test.py \
        --capture-cmd "python3 ../../sprint1/hardware/scripts/camera_test.py --headless" \
        --preprocess-cmd "python3 /path/to/preprocessing_script.py" \
        --label concurrent_capture_and_augmentation
"""

import argparse
import datetime
import os
import shlex
import subprocess
import time

import psutil
import yaml


def load_config(path):
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def read_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure resource use when capture and preprocessing run concurrently"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--capture-cmd", required=True,
                         help="Command representing the capture-side workload")
    parser.add_argument("--preprocess-cmd", required=True,
                         help="Command representing the preprocessing-side workload")
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--label", default="concurrent_run")
    parser.add_argument("--log-dir", default="../logs")
    parser.add_argument("--timeout", type=int, default=600,
                         help="Safety cap in seconds in case either process hangs")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    interval = args.interval if args.interval is not None else cfg.get("interval_sec", 5)

    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    log_path = os.path.join(args.log_dir, f"concurrent_log_{args.label}_{stamp}.txt")

    print(f"[concurrent_load_test] Starting capture process: {args.capture_cmd}")
    capture_proc = subprocess.Popen(shlex.split(args.capture_cmd))

    print(f"[concurrent_load_test] Starting preprocessing process: {args.preprocess_cmd}")
    preprocess_proc = subprocess.Popen(shlex.split(args.preprocess_cmd))

    start = time.time()
    samples = []

    with open(log_path, "w") as log_file:
        header = (
            f"Sprint 2 Concurrent Load Test\n"
            f"Label: {args.label}\n"
            f"Started: {datetime.datetime.now().isoformat()}\n"
            f"Capture-side command: {args.capture_cmd}\n"
            f"Preprocessing-side command: {args.preprocess_cmd}\n"
            f"Sampling interval: {interval}s\n"
            f"{'-'*76}\n"
            f"{'elapsed_s':>10} {'cpu_%':>8} {'ram_%':>8} {'temp_C':>8} "
            f"{'capture_alive':>14} {'preprocess_alive':>17}\n"
        )
        print(header)
        log_file.write(header)
        log_file.flush()

        while True:
            elapsed = time.time() - start
            capture_alive = capture_proc.poll() is None
            preprocess_alive = preprocess_proc.poll() is None

            if not capture_alive and not preprocess_alive:
                break
            if elapsed > args.timeout:
                print(f"[concurrent_load_test] Timeout ({args.timeout}s) reached, stopping.")
                for p in (capture_proc, preprocess_proc):
                    if p.poll() is None:
                        p.terminate()
                break

            cpu_pct = psutil.cpu_percent(interval=None)
            ram_pct = psutil.virtual_memory().percent
            temp = read_cpu_temp()
            temp_str = f"{temp:.1f}" if temp is not None else "n/a"
            samples.append((cpu_pct, ram_pct, temp))

            line = (
                f"{elapsed:10.1f} {cpu_pct:8.1f} {ram_pct:8.1f} {temp_str:>8} "
                f"{str(capture_alive):>14} {str(preprocess_alive):>17}\n"
            )
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            time.sleep(interval)

        total_elapsed = time.time() - start
        if samples:
            avg_cpu = sum(s[0] for s in samples) / len(samples)
            peak_cpu = max(s[0] for s in samples)
            avg_ram = sum(s[1] for s in samples) / len(samples)
            peak_ram = max(s[1] for s in samples)
            temps = [s[2] for s in samples if s[2] is not None]
            peak_temp = max(temps) if temps else None
        else:
            avg_cpu = peak_cpu = avg_ram = peak_ram = 0
            peak_temp = None

        summary = (
            f"\n{'-'*76}\n"
            f"Finished: {datetime.datetime.now().isoformat()}\n"
            f"Total runtime: {total_elapsed:.1f}s\n"
            f"Average CPU %: {avg_cpu:.1f}\n"
            f"Peak CPU %: {peak_cpu:.1f}\n"
            f"Average RAM %: {avg_ram:.1f}\n"
            f"Peak RAM %: {peak_ram:.1f}\n"
            f"Peak temp C: {peak_temp if peak_temp is not None else 'n/a'}\n"
        )
        print(summary)
        log_file.write(summary)

    print(f"[concurrent_load_test] Log written to {log_path}")
    print("[concurrent_load_test] Compare peak CPU/temp here against the "
          "single-workload numbers in resource_consumption_report.md — a "
          "big jump tells Sprint 3 that concurrent capture+inference needs "
          "throttling or a lighter model.")


if __name__ == "__main__":
    main()
