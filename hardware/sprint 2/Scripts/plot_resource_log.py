#!/usr/bin/env python3
"""
plot_resource_log.py

Sprint 2 (Off-Sprint Bridging Task) - Pipeline Integration & Evaluation

Parses a resource_monitor.py / concurrent_load_test.py log file and
renders CPU %, RAM %, and temperature over time as a PNG chart, for
dropping straight into resource_consumption_report.md or a sprint review
slide.

Usage:
    python3 plot_resource_log.py --log ../logs/resource_log_frame_augmentation_2026-10-15_1400.txt \
        --out ../logs/resource_log_frame_augmentation.png
"""

import argparse
import re

import matplotlib
matplotlib.use("Agg")  # no display needed on a headless board
import matplotlib.pyplot as plt

ROW_PATTERN = re.compile(
    r"^\s*(?P<elapsed>[\d.]+)\s+(?P<cpu>[\d.]+)\s+(?P<ram>[\d.]+)\s+(?P<temp>[\d.]+|n/a)"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a resource_monitor.py log")
    parser.add_argument("--log", required=True, help="Path to the log .txt file")
    parser.add_argument("--out", required=True, help="Path to write the output .png to")
    parser.add_argument("--title", default=None, help="Chart title (defaults to the log filename)")
    return parser.parse_args()


def parse_log(path):
    elapsed, cpu, ram, temp = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            match = ROW_PATTERN.match(line)
            if not match:
                continue
            try:
                e, c, r = (float(match.group(g)) for g in ("elapsed", "cpu", "ram"))
            except ValueError:
                # Skips placeholder/ellipsis rows (e.g. "..." in template logs)
                continue
            elapsed.append(e)
            cpu.append(c)
            ram.append(r)
            t = match.group("temp")
            try:
                temp.append(float(t) if t not in ("n/a", None) else None)
            except ValueError:
                temp.append(None)
    return elapsed, cpu, ram, temp


def main():
    args = parse_args()
    elapsed, cpu, ram, temp = parse_log(args.log)

    if not elapsed:
        print(f"[plot_resource_log] No parseable data rows found in {args.log}")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Percent (%)")
    ax1.plot(elapsed, cpu, label="CPU %", color="tab:blue")
    ax1.plot(elapsed, ram, label="RAM %", color="tab:green")
    ax1.set_ylim(0, 100)

    if any(t is not None for t in temp):
        ax2 = ax1.twinx()
        ax2.set_ylabel("Temperature (°C)")
        ax2.plot(elapsed, [t if t is not None else float("nan") for t in temp],
                 label="Temp (°C)", color="tab:red", linestyle="--")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    plt.title(args.title or args.log.split("/")[-1])
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[plot_resource_log] Chart written to {args.out}")


if __name__ == "__main__":
    main()
