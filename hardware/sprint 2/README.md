# Sprint 2 (Off-Sprint Bridging Task) — Pipeline Integration & Evaluation

**Owner:** Aman Raj (Hardware & Embedded Lead)
**User story:** Sprint 2 Pipeline Integration & Evaluation *(Off-Sprint Bridging Task)*
**Deliverable:** Maintain embedded board data pipeline and assess hardware resource consumption during model preprocessing.
**Window:** Sep 28, 2026 – Nov 15, 2026
**Branch:** `feature/sprint2-hardware-bridging`

## Why this task exists

You're off-sprint for Sprint 2's main work (Ayush/Aryan building the CNN and
training loop). This bridging task has two jobs instead:

1. **Keep the Sprint 1 data pipeline working** while Sprint 2 consumes it —
   if the dataset schema or frame format needs to change to support
   preprocessing (tensor conversion, augmentation), you adapt the pipeline
   and verify it still holds together.
2. **Measure hardware cost** — log CPU/RAM/temperature on the embedded board
   while preprocessing-style work runs, so Sprint 3 (real hardware assembly)
   starts with a known resource budget instead of a guess.

## Folder structure

```
sprint2/hardware/
├── README.md                              (this file)
├── docs/
│   ├── pipeline_maintenance_log.md        what changed in the Sprint 1 pipeline, and why
│   ├── resource_consumption_report.md     preprocessing-alone test method + results template
│   ├── capacity_planning.md               what the numbers mean for Sprint 3's design
│   └── handoff_notes_to_dl_team.md        one-page brief for Ayush/Aryan on the pipeline
├── scripts/
│   ├── pipeline_health_check.py           validates dataset/metadata.csv is still consistent
│   ├── resource_monitor.py                wraps ONE command, logs CPU/RAM/temp while it runs
│   ├── concurrent_load_test.py            runs capture + preprocessing TOGETHER, logs combined load
│   ├── migrate_metadata_schema.py         safely adds columns to metadata.csv (with backup)
│   ├── plot_resource_log.py               turns a resource/concurrent log into a PNG chart
│   └── requirements.txt
├── configs/
│   └── monitor_config.yaml
├── logs/
│   ├── health_check_report_TEMPLATE.txt
│   └── resource_log_TEMPLATE.txt
└── tests/
    └── test_pipeline_health_check.py      automated tests for the health-check logic (pytest)
```

## How to run

**1. Check the pipeline is still healthy** (run this any time Sprint 2 asks
for a schema/format change, or on a schedule):

```bash
python3 scripts/pipeline_health_check.py --dataset ../../sprint1/dataset
```

**2. Measure resource cost of a preprocessing step.** Point it at whatever
command Ayush/Aryan's preprocessing script is — `resource_monitor.py` runs
it as a subprocess and logs system metrics alongside it:

```bash
python3 scripts/resource_monitor.py \
    --config ../configs/monitor_config.yaml \
    --command "python3 /path/to/preprocessing_script.py" \
    --label "frame_augmentation"
```

Log lands in `../logs/resource_log_<label>_<date>.txt`. Copy the
average/peak values into `docs/resource_consumption_report.md`.

**3. Measure the realistic case: capture + preprocessing at the same time.**
This is the number Sprint 3 actually needs, since the real system runs both
concurrently:

```bash
python3 scripts/concurrent_load_test.py \
    --capture-cmd "python3 ../../sprint1/hardware/scripts/camera_test.py --headless" \
    --preprocess-cmd "python3 /path/to/preprocessing_script.py" \
    --label concurrent_capture_and_preprocessing
```

Log results feed `docs/capacity_planning.md`.

**4. Visualize any log as a chart:**

```bash
python3 scripts/plot_resource_log.py --log ../logs/<some_log>.txt --out ../logs/<some_log>.png
```

**5. Run the automated tests** (validates the health-check logic itself,
not just the dataset):

```bash
cd tests && pytest test_pipeline_health_check.py -v
```

**6. If the DL team needs a schema change**, don't hand-edit the CSV:

```bash
python3 scripts/migrate_metadata_schema.py --dataset ../../sprint1/dataset \
    --add-column landmark_ref --default ""
```

## Do NOT commit

- Full dataset / preprocessed tensor dumps
- `venv/`, `.venv/`, `__pycache__/`
- Anything belonging to Sprint 2's actual model code (that's Ayush's/Aryan's
  deliverable, not yours) — you're only committing the pipeline maintenance
  and resource-measurement evidence

## Status / what's left before this is submission-ready

- [ ] Confirm with Ayush/Aryan whether the Sprint 1 metadata schema needs any
      changes for their preprocessing step; record the answer (or "no
      change needed") in `docs/pipeline_maintenance_log.md`
- [ ] Run `pipeline_health_check.py` against the real dataset and commit the
      real output instead of the template
- [ ] Run `resource_monitor.py` around an actual preprocessing run (or a
      representative stand-in load) on the real board, and commit the real
      log + filled-in report
- [ ] Run `concurrent_load_test.py` with real capture + real (or stand-in)
      preprocessing, and fill in `docs/capacity_planning.md`
- [ ] Send `docs/handoff_notes_to_dl_team.md` to Ayush/Aryan and get their
      sign-off that the schema and constraints are correct
- [ ] Confirm `pytest tests/test_pipeline_health_check.py` passes in your
      environment (all 5 tests passed when authored — see repo history)
