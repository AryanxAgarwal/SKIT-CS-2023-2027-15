# Hardware Pipeline Handoff Notes — For Ayush / Aryan (Sprint 2 Deep Learning Work)

**From:** Aman Raj (Hardware & Embedded Lead)
**Story:** Sprint 2 Pipeline Integration & Evaluation (Off-Sprint Bridging Task)
**Purpose:** Everything the DL side needs to know about the data pipeline
you're building on top of, in one place, so you don't have to go dig
through Sprint 1's hardware code.

## 1. What you're reading from

```
dataset/
├── frames/frame_NNNNNN.jpg      (6-digit zero-padded, sequential per session)
└── metadata.csv
```

`metadata.csv` columns (Sprint 1 baseline — see
`pipeline_maintenance_log.md` for any changes made since):

| Column | Type | Notes |
|---|---|---|
| `frame_id` | int | Matches the number in the filename |
| `filename` | string | Relative path, e.g. `frames/frame_000001.jpg` |
| `timestamp` | ISO 8601 string | Local capture time |
| `width` / `height` | int | Actual saved frame dimensions |
| `fps_at_capture` | float | Instantaneous FPS at the moment this frame was captured |

## 2. If you need a schema change

Don't hand-edit `metadata.csv`. Tell me the column you need (name, type,
default for existing rows) and I'll run it through
`scripts/migrate_metadata_schema.py`, which adds it safely and keeps a
backup. I'll log the change in `pipeline_maintenance_log.md` so there's a
record of why the schema looks the way it does.

## 3. How to check the data is trustworthy before you build on it

```bash
python3 scripts/pipeline_health_check.py --dataset /path/to/dataset
```

This confirms every metadata row has a matching, readable, correctly-sized
image file, and flags orphaned files. Run it any time you pull a fresh
dataset drop, or if something in your preprocessing step looks off — it's
often faster than debugging your own code first.

## 4. Known hardware constraints (see `resource_consumption_report.md` and
`capacity_planning.md` for full numbers)

- Board: *(Pi model / Jetson — fill in)*
- Available CPU/RAM headroom during preprocessing: *(fill in from the
  resource report)*
- If your preprocessing step is heavier than what we measured, let me know
  — I'd rather re-run `resource_monitor.py` against your actual script
  than have Sprint 3 discover a bottleneck late.

## 5. What I can't help with

I'm off-sprint for the actual model work (CNN architecture, training,
augmentation logic) — that's your and Ayush's call. My scope here is
keeping the data underneath it solid and telling you honestly what the
hardware can and can't afford.

## 6. Open items I need from you

- [ ] Confirm the Sprint 1 metadata schema is sufficient, or tell me what
      to add
- [ ] Send me (or point me to) your preprocessing script so I can run the
      concurrent load test against it, rather than a synthetic stand-in
