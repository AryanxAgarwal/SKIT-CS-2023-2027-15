# Data Flow Verification Report — Sprint 1

**Story:** Sprint 1 Baseline Integration & Data Flow Verification
**Owner:** Aman Raj

## 1. Purpose

Document how frames captured on the embedded board are saved and described,
so that downstream teammates can consume them without needing to ask:

- **Aryan Agarwal** (Camera Stream & Video Capture Module / Dataset Schema) — needs
  frame files in the agreed naming convention and folder layout.
- **Chiranshu Mudgal** (Sprint 1 Baseline Integration & Data Flow Verification,
  off-sprint bridging task — CI/CD + frame ingest verification) — needs the
  metadata schema to write an automated ingest check.

## 2. Folder layout produced by `save_frames_metadata.py`

```
dataset/
├── frames/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
└── metadata.csv
```

## 3. Frame naming convention

`frame_<6-digit zero-padded index>.jpg` — sequential, reset per capture session.

*(Confirm with Aryan whether a session ID or timestamp prefix should be added
instead — update this section once agreed.)*

## 4. metadata.csv schema

| Column | Type | Description |
|---|---|---|
| `frame_id` | int | Sequential frame index, matches the number in the filename |
| `filename` | string | Relative path under `frames/` |
| `timestamp` | ISO 8601 string | Capture time (local) |
| `width` | int | Frame width in pixels |
| `height` | int | Frame height in pixels |
| `fps_at_capture` | float | Measured instantaneous FPS when this frame was captured |

A small sample (not the full dataset) is committed at `logs/metadata_sample.csv`.

## 5. Integration notes

- Frames are saved as JPEG to keep the committed sample small; raw capture
  format inside the script is BGR (OpenCV default) before encoding.
- Full dataset directories (`dataset/frames/`) are excluded via `.gitignore`
  and should be shared via the team's storage location, not GitHub.
- *(Add anything Aryan/Chiranshu asked you to change after they review this —
  e.g. a different metadata column, a session-level manifest file, etc.)*

## 6. Open questions for the team

- [ ] Confirm final metadata schema with Aryan before Sprint 2 dataset formatting starts
- [ ] Confirm whether Chiranshu's ingest check expects a manifest file per session
