# Hardware Setup Guide — Sprint 1

## 1. Target board

| Item | Value (fill in with what you actually used) |
|---|---|
| Board | Raspberry Pi 4B / 5 (or Jetson Nano) — *edit to match* |
| RAM | e.g. 4 GB / 8 GB |
| OS | Raspberry Pi OS (64-bit, Bookworm) — *edit* |
| Storage | e.g. 32 GB microSD, Class 10 |
| Camera | USB webcam / Pi Camera Module 3 — *edit* |
| Power | 5V/3A bench supply (Sprint 1 uses bench power, not vehicle power) |

## 2. Flash the OS

1. Use Raspberry Pi Imager to flash Raspberry Pi OS (64-bit) to the microSD card.
2. Enable SSH and set hostname/user during imaging (Imager → Advanced Options), or run `sudo raspi-config` after first boot.
3. Boot the board, connect to Wi-Fi/Ethernet, and confirm SSH access:
   ```bash
   ssh <user>@<pi-hostname>.local
   ```

## 3. Connect the camera

- **USB camera:** plug into any USB port. Confirm detection with:
  ```bash
  ls /dev/video*
  v4l2-ctl --list-devices
  ```
- **CSI camera (Pi Camera Module):** connect the ribbon cable to the CSI port (blue side facing the USB ports), enable the camera interface via `sudo raspi-config` → Interface Options → Camera, then reboot.

## 4. Install dependencies

Run the provided installer (see `scripts/setup_env.sh`), which installs:
- Python 3 + pip
- OpenCV (`opencv-python`)
- NumPy
- `psutil` (for CPU/RAM logging in the stability test)

```bash
bash scripts/setup_env.sh
```

## 5. Power setup (Sprint 1 scope only)

Sprint 1 only requires **bench power** — a direct 5V/3A supply into the board. Vehicle power (12V → 5V buck converter) is **not** in scope until later sprints; if you did test it anyway, note the converter model and measured output voltage here and add a note to `data_flow_verification_report.md`.

## 6. Cooling

If you added a heatsink or fan, note the model here. If not, flag it as a known thermal risk for the sustained-capture test (temperatures are logged in the stability test and should be watched for throttling above ~80°C).

## 7. Verify

Run `scripts/camera_test.py` to confirm the camera opens and streams before moving to the stability test.
