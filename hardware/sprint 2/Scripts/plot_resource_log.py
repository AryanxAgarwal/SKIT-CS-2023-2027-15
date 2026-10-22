
#!/usr/bin/env python3
"""
plot_resource_log.py

Sprint 2 - Pipeline Integration & Evaluation

Reads a resource monitoring log and creates a PNG graph
showing CPU usage, RAM usage, and temperature over time.
"""

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATTERN = re.compile(
    r"^\s*(?P<time>[\d.]+)\s+(?P<cpu>[\d.]+)\s+"
    r"(?P<memory>[\d.]+)\s+(?P<temperature>[\d.]+|n/a)"
)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a graph from resource usage logs"
    )
    parser.add_argument("--log", required=True, help="Input log file")
    parser.add_argument("--out", required=True, help="Output PNG file")
    parser.add_argument("--title", default=None, help="Optional graph title")
    return parser.parse_args()


def read_log(filename):
    times, cpu_values, memory_values, temperatures = [], [], [], []

    with open(filename, "r") as file:
        for line in file:
            result = LOG_PATTERN.match(line)

            if result is None:
                continue

            try:
                time_value = float(result.group("time"))
                cpu_value = float(result.group("cpu"))
                memory_value = float(result.group("memory"))
            except ValueError:
                continue

            times.append(time_value)
            cpu_values.append(cpu_value)
            memory_values.append(memory_value)

            temp_value = result.group("temperature")
            try:
                temperatures.append(
                    float(temp_value) if temp_value != "n/a" else None
                )
            except ValueError:
                temperatures.append(None)

    return times, cpu_values, memory_values, temperatures


def main():
    options = get_arguments()

    times, cpu, memory, temperature = read_log(options.log)

    if len(times) == 0:
        print(f"No valid resource data found in {options.log}")
        return

    figure, cpu_axis = plt.subplots(figsize=(10, 5))

    cpu_axis.set_xlabel("Elapsed Time (seconds)")
    cpu_axis.set_ylabel("CPU / RAM Usage (%)")

    cpu_axis.plot(times, cpu, label="CPU Usage")
    cpu_axis.plot(times, memory, label="RAM Usage")
    cpu_axis.set_ylim(0, 100)

    if any(value is not None for value in temperature):
        temp_axis = cpu_axis.twinx()
        temp_axis.set_ylabel("Temperature (°C)")

        temp_axis.plot(
            times,
            [value if value is not None else float("nan")
             for value in temperature],
            label="Temperature",
            linestyle="--"
        )

        first_lines, first_labels = cpu_axis.get_legend_handles_labels()
        second_lines, second_labels = temp_axis.get_legend_handles_labels()

        cpu_axis.legend(
            first_lines + second_lines,
            first_labels + second_labels,
            loc="upper left"
        )
    else:
        cpu_axis.legend(loc="upper left")

    figure.suptitle(options.title or options.log.split("/")[-1])
    figure.tight_layout()
    figure.savefig(options.out, dpi=150)
    plt.close(figure)

    print(f"Resource usage graph saved to: {options.out}")


if __name__ == "__main__":
    main()
