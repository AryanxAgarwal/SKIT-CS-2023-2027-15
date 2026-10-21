#!/usr/bin/env python3
"""
resource_monitor.py

Sprint 2 (Off-Sprint Bridging Task) - Pipeline Integration & Evaluation
Runs an arbitrary command (e.g. a preprocessing script) as a subprocess and
logs CPU %, RAM %, and CPU temperature at a fixed interval for as long as
that command runs. This is how we assess hardware resource consumption
during model preprocessing without needing to modify Sprint 2's own code.

Usage:
    python3 resource_monitor.py --config ../configs/monitor_config.yaml \
        --command "python3 preprocessing_script.py" --label frame_augmentation

    # Or just monitor the whole system for a fixed duration with no command:
    python3 resource_monitor.py --duration 300 --label idle_baseline
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
        print(f"[resource_monitor] config file not found at {path}, using CLI/defaults")
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
    parser = argparse.ArgumentParser(
        description="Monitor CPU/RAM/temp while a command runs (or for a fixed duration)"
    )
    parser.add_argument("--config", default=None, help="Path to monitor_config.yaml")
    parser.add_argument("--command", default=None,
                         help="Command to run and monitor, e.g. 'python3 preprocess.py'")
    parser.add_argument("--duration", type=int, default=None,
                         help="If no --command given, monitor the system for this many seconds")
    parser.add_argument("--interval", type=int, default=None,
                         help="Sampling interval in seconds (overrides config, default 5)")
    parser.add_argument("--label", default="run",
                         help="Short label used in the log filename, e.g. 'frame_augmentation'")
    parser.add_argument("--log-dir", default="../logs", help="Directory to write the log file to")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    interval = args.interval if args.interval is not None else cfg.get("interval_sec", 5)

    if not args.command and not args.duration:
        print("[resource_monitor] Provide either --command or --duration.")
        return

    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    log_path = os.path.join(args.log_dir, f"resource_log_{args.label}_{stamp}.txt")

    proc = None
    if args.command:
        print(f"[resource_monitor] Starting subprocess: {args.command}")
        proc = subprocess.Popen(shlex.split(args.command))

    start = time.time()
    samples = []

    with open(log_path, "w") as log_file:
        header = (
            f"Sprint 2 Resource Consumption Monitor\n"
            f"Label: {args.label}\n"
            f"Started: {datetime.datetime.now().isoformat()}\n"
            f"Monitored command: {args.command or f'(no command — idle monitor for {args.duration}s)'}\n"
            f"Sampling interval: {interval}s\n"
            f"{'-'*70}\n"
            f"{'elapsed_s':>10} {'cpu_%':>8} {'ram_%':>8} {'temp_C':>8}\n"
        )
        print(header)
        log_file.write(header)
        log_file.flush()

        while True:
            if proc is not None:
                if proc.poll() is not None:
                    break
            elif time.time() - start >= args.duration:
                break

            cpu_pct = psutil.cpu_percent(interval=None)
            ram_pct = psutil.virtual_memory().percent
            temp = read_cpu_temp()
            temp_str = f"{temp:.1f}" if temp is not None else "n/a"
            elapsed = time.time() - start

            samples.append((cpu_pct, ram_pct, temp))

            line = f"{elapsed:10.1f} {cpu_pct:8.1f} {ram_pct:8.1f} {temp_str:>8}\n"
            print(line, end="")
            log_file.write(line)
            log_file.flush()

            time.sleep(interval)

        if proc is not None:
            proc.wait()

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
            f"\n{'-'*70}\n"
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

    print(f"[resource_monitor] Log written to {log_path}")
    print("[resource_monitor] Copy the summary values into "
          "docs/resource_consumption_report.md")


if __name__ == "__main__":
    main()
