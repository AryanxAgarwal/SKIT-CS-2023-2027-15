# Raw Video Stream Verification Report — Sprint 1

**Story:** Sprint 1 Baseline Integration & Data Flow Verification
**Owner:** Aman Raj
**Test script:** `scripts/stability_test.py`
**Log file:** `logs/stability_log_<date>.txt`

## 1. Objective

Verify that the target board can sustain a raw video capture stream at the
project's target frame rate, without dropped frames, overheating, or
resource exhaustion, over a realistic continuous session.

## 2. Test method

| Parameter | Value |
|---|---|
| Duration | 30 min (1800 s) — *edit if you ran a different length* |
| Target resolution | 640×480 — *edit to match config.yaml* |
| Target FPS | 30 |
| Camera | *USB / CSI, model* |
| Board | *Pi model* |
| Ambient conditions | *e.g. indoor, ~25°C room* |

The test captures frames continuously via `cv2.VideoCapture`, and every
`LOG_INTERVAL_SEC` seconds records: instantaneous FPS, CPU % (`psutil`),
RAM % (`psutil`), and CPU temperature (`/sys/class/thermal/thermal_zone0/temp`
on Raspberry Pi).

## 3. Results

*(Fill in after running `stability_test.py` for the real session — pull the
numbers from `logs/stability_log_<date>.txt`.)*

| Metric | Result |
|---|---|
| Average FPS | — |
| Minimum FPS | — |
| Average CPU % | — |
| Peak CPU % | — |
| Average RAM % | — |
| Peak RAM % | — |
| Peak temperature | — |
| Dropped/failed frame reads | — |
| Thermal throttling observed? | Yes / No |

## 4. Result

**PASS / FAIL** — *pick one once you have real numbers.*

Pass criteria: average FPS ≥ 25 (within 5 fps of the 30 fps target), no
sustained thermal throttling, no capture failures over the full duration.

## 5. Notes / issues encountered

*(e.g. driver quirks, USB bandwidth limits, camera auto-exposure hunting,
anything you had to work around.)*
