# Pipeline Maintenance Log — Sprint 2 Bridging Task

**Story:** Sprint 2 Pipeline Integration & Evaluation (Off-Sprint Bridging Task)
**Owner:** Aman Raj

## Purpose

Sprint 1 defined the frame-capture pipeline (`frames/frame_NNNNNN.jpg` +
`metadata.csv`). Sprint 2 now consumes that data for preprocessing
(tensor formatting, augmentation, normalization). This log tracks any
changes made to the pipeline to support that, so there's a record of why
the schema looks the way it does at any point in time.

## Baseline (carried over from Sprint 1)

`metadata.csv` columns: `frame_id, filename, timestamp, width, height, fps_at_capture`
Frame naming: `frame_<6-digit index>.jpg`

## Change log

| Date | Requested by | Change | Reason |
|---|---|---|---|
| *(fill in)* | *(e.g. Ayush)* | *(e.g. added `landmark_ref` column)* | *(e.g. needed to join frames to landmark coordinates for tensor formatting)* |
| | | | |

*(If no changes were needed this sprint, write "No schema changes required
— Sprint 1 format was sufficient for Sprint 2 preprocessing" and note the
date you confirmed that with the team.)*

## Compatibility check

Before Sprint 2 preprocessing work depends on this pipeline, confirm:

- [ ] `pipeline_health_check.py` passes against the current dataset
- [ ] Ayush's tensor-formatting script (Sprint 2: "Preprocessed Image &
      Landmark Dataset Formatting") can read `metadata.csv` without a
      custom parser change on their end
- [ ] Aryan's facial landmark output (Sprint 1: "Facial Landmark Detection
      Module") is joinable to frames via `frame_id` or `filename`

## Known limitations / follow-ups

*(e.g. "metadata.csv doesn't yet carry a landmark reference — if Sprint 2
needs frame-to-landmark joins, add a column and note it above.")*
