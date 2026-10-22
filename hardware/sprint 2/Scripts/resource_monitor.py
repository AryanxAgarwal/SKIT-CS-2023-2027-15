
#!/usr/bin/env python3
"""
resource_monitor.py

Sprint 2 - Pipeline Integration and Evaluation

Tracks CPU usage, memory consumption, and CPU temperature while
a selected command is running or for a specified time period.
Results are displayed on the terminal and stored in a log file.
"""

import argparse
import datetime
import os
import shlex
import subprocess
import time

import psutil
import yaml


def get_config(config_path):
    """Load monitor settings from a YAML file."""
    if not config_path:
        return {}

    try:
        with open(config_path, "r") as config_file:
            settings = yaml.safe_load(config_file)
            return settings or {}
    except FileNotFoundError:
        print(
            f"[resource_monitor] config file not found at {config_path}, "
            "using CLI/defaults"
        )
        return {}


def get_temperature():
    """Return CPU temperature in Celsius when the system exposes it."""
    sensor_file = "/sys/class/thermal/thermal_zone0/temp"

    try:
        with open(sensor_file, "r") as sensor:
            temperature = int(sensor.read().strip())
        return temperature / 1000.0
    except (FileNotFoundError, ValueError):
        return None


def get_arguments():
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description="Monitor CPU/RAM/temp while a command runs "
                    "(or for a fixed duration)"
    )

    parser.add_argument("--config", default=None,
                        help="Path to monitor_config.yaml")
    parser.add_argument(
        "--command",
        default=None,
        help="Command to run and monitor, e.g. 'python3 preprocess.py'"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Monitoring duration in seconds when no command is supplied"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Sampling interval in seconds (overrides config, default 5)"
    )
    parser.add_argument(
        "--label",
        default="run",
        help="Short label used in the log filename"
    )
    parser.add_argument(
        "--log-dir",
        default="../logs",
        help="Directory to write the log file to"
    )

    return parser.parse_args()


def main():
    options = get_arguments()
    config = get_config(options.config)

    sample_interval = (
        options.interval
        if options.interval is not None
        else config.get("interval_sec", 5)
    )

    if not options.command and not options.duration:
        print("[resource_monitor] Provide either --command or --duration.")
        return

    os.makedirs(options.log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    output_file = os.path.join(
        options.log_dir,
        f"resource_log_{options.label}_{timestamp}.txt"
    )

    process = None

    if options.command:
        print(f"[resource_monitor] Starting subprocess: {options.command}")
        process = subprocess.Popen(shlex.split(options.command))

    start_time = time.time()
    readings = []

    with open(output_file, "w") as log:
        description = (
            "Sprint 2 Resource Consumption Monitor\n"
            f"Label: {options.label}\n"
            f"Started: {datetime.datetime.now().isoformat()}\n"
            f"Monitored command: {options.command or f'(no command — idle monitor for {options.duration}s)'}\n"
            f"Sampling interval: {sample_interval}s\n"
            f"{'-' * 70}\n"
            f"{'elapsed_s':>10} {'cpu_%':>8} {'ram_%':>8} {'temp_C':>8}\n"
        )

        print(description)
        log.write(description)
        log.flush()

        while True:
            elapsed_time = time.time() - start_time

            if process is not None:
                if process.poll() is not None:
                    break
            elif elapsed_time >= options.duration:
                break

            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            temperature = get_temperature()

            temp_text = (
                f"{temperature:.1f}"
                if temperature is not None
                else "n/a"
            )

            readings.append((cpu_usage, memory_usage, temperature))

            record = (
                f"{elapsed_time:10.1f} "
                f"{cpu_usage:8.1f} "
                f"{memory_usage:8.1f} "
                f"{temp_text:>8}\n"
            )

            print(record, end="")
            log.write(record)
            log.flush()

            time.sleep(sample_interval)

        if process is not None:
            process.wait()

        runtime = time.time() - start_time

        if readings:
            cpu_average = sum(item[0] for item in readings) / len(readings)
            cpu_maximum = max(item[0] for item in readings)

            ram_average = sum(item[1] for item in readings) / len(readings)
            ram_maximum = max(item[1] for item in readings)

            valid_temperatures = [
                item[2] for item in readings if item[2] is not None
            ]
            temperature_maximum = (
                max(valid_temperatures) if valid_temperatures else None
            )
        else:
            cpu_average = cpu_maximum = 0
            ram_average = ram_maximum = 0
            temperature_maximum = None

        report = (
            f"\n{'-' * 70}\n"
            f"Finished: {datetime.datetime.now().isoformat()}\n"
            f"Total runtime: {runtime:.1f}s\n"
            f"Average CPU %: {cpu_average:.1f}\n"
            f"Peak CPU %: {cpu_maximum:.1f}\n"
            f"Average RAM %: {ram_average:.1f}\n"
            f"Peak RAM %: {ram_maximum:.1f}\n"
            f"Peak temp C: {temperature_maximum if temperature_maximum is not None else 'n/a'}\n"
        )

        print(report)
        log.write(report)

    print(f"[resource_monitor] Log written to {output_file}")
    print(
        "[resource_monitor] Copy the summary values into "
        "docs/resource_consumption_report.md"
    )


if __name__ == "__main__":
    main()
