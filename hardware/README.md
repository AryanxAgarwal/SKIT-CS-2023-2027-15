# Sprint 1 — Baseline Integration & Data Flow Verification

**Owner:** Aman Raj (Hardware & Embedded Lead)
**User story (from Sprint plan):** Setup embedded hardware target board (Raspberry Pi / Jetson) and verify raw video stream stability.
**Sprint window:** Aug 10, 2026 – Sep 27, 2026
**Branch:** `feature/sprint1-hardware-bringup`

## What this folder proves

This is the hardware bring-up evidence for Sprint 1 of the Driver Drowsiness Detection project. It shows that:

1. The embedded target board (Raspberry Pi, or Jetson if that's what you used) is set up and can run OpenCV.
2. A camera is connected and produces a live video stream.
3. That stream is **stable** over a sustained period (target: 30 fps, logged CPU/RAM/temperature).
4. Captured frames + metadata are saved in a format that Aryan (Camera Stream & Data Acquisition) and Chiranshu (CI/CD & frame ingest verification) can consume downstream.

## Folder structure

```
sprint1/hardware/
├── README.md                              (this file)
├── docs/
│   ├── hardware_setup_guide.md            board, OS, camera, power, install steps
│   ├── raw_stream_verification_report.md  test method + results template
│   └── data_flow_verification_report.md   frame/metadata format + integration notes
├── scripts/
│   ├── camera_test.py                     detect camera, show live stream
│   ├── stability_test.py                  timed run logging FPS/CPU/RAM/temp
│   ├── save_frames_metadata.py            saves frames + metadata.csv
│   ├── setup_env.sh                       one-shot dependency installer
│   └── requirements.txt
├── configs/
│   ├── config.yaml                        camera index, resolution, fps, save path
│   └── .gitignore
├── logs/
│   ├── stability_log_TEMPLATE.txt         what stability_test.py writes — fill by running it
│   └── metadata_sample.csv                small sample only, NOT the full dataset
└── images/                                (add setup photo + wiring diagram here)
```

## How to run

```bash
cd scripts
bash setup_env.sh
python3 camera_test.py --config ../configs/config.yaml
python3 stability_test.py --config ../configs/config.yaml --duration 1800
python3 save_frames_metadata.py --config ../configs/config.yaml --num-frames 50
```

`stability_test.py` writes its log to `../logs/`. Rename the output from the
template name to `stability_log_<date-you-ran-it>.txt` before committing.

## Do NOT commit

- Full frame dataset / full video files
- `venv/`, `.venv/`, `__pycache__/`
- Wi-Fi credentials, tokens, passwords
- Anything from Sprint 2–4 scope (buzzer, GPIO, enclosure, field testing)

## Status / what's left before this is submission-ready

- [ ] Run `stability_test.py` for a real 30 min session on the actual board and drop the real log in `logs/`
- [ ] Fill in the measured values (fps/CPU/RAM/temp/result) in `docs/raw_stream_verification_report.md`
- [ ] Add `images/setup_photo.jpg` and `images/power_connection_diagram.png`
- [ ] Confirm frame-naming/metadata schema with Aryan and Chiranshu, note any changes in `docs/data_flow_verification_report.md`
