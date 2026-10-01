# Hardware Resource Consumption Report — Sprint 2 Bridging Task

**Story:** Sprint 2 Pipeline Integration & Evaluation (Off-Sprint Bridging Task)
**Owner:** Aman Raj
**Test script:** `scripts/resource_monitor.py`

## 1. Objective

Measure how much CPU, RAM, and thermal headroom the embedded target board
uses while Sprint 2 preprocessing (tensor formatting / frame augmentation /
normalization) runs, so Sprint 3 hardware assembly can plan around a known
resource budget instead of an assumption.

## 2. Test method

| Parameter | Value |
|---|---|
| Board | *(same board as Sprint 1 — Pi model / Jetson)* |
| Workload monitored | *(e.g. Ayush's frame augmentation script, or a representative stand-in if that script wasn't runnable on this board)* |
| Monitoring interval | 5s |
| Duration | *(length of the preprocessing run tested)* |

`resource_monitor.py` runs the target command as a subprocess and samples
CPU %, RAM %, and CPU temperature (where available) at a fixed interval for
the duration of that command, using `psutil`.

## 3. Results

*(Fill in from `logs/resource_log_<label>_<date>.txt`)*

| Metric | Result |
|---|---|
| Average CPU % | — |
| Peak CPU % | — |
| Average RAM % | — |
| Peak RAM % | — |
| Peak temperature | — |
| Preprocessing runtime | — |
| Thermal throttling observed? | Yes / No |

## 4. Assessment

**Fits on the embedded board as-is / needs offloading to a host machine /
needs a lighter preprocessing step** — *pick one once you have real
numbers, and say why.*

## 5. Recommendation for Sprint 3

*(e.g. "CPU headroom is tight during augmentation — GPIO alert driver
should avoid running concurrently with any on-device preprocessing" or
"resource use was low; safe to run preprocessing and capture
simultaneously.")*

## 6. Notes / issues encountered

*(e.g. couldn't run the actual TensorFlow/PyTorch preprocessing script on
the Pi due to missing wheels — used a CPU-bound stand-in load instead;
note that limitation here so the numbers are read in context.)*
