# Capacity Planning for Sprint 3 — Based on Sprint 2 Resource Findings

**Story:** Sprint 2 Pipeline Integration & Evaluation (Off-Sprint Bridging Task)
**Owner:** Aman Raj
**Feeds into:** Sprint 3 — "Threading & Parallel Processing Architecture",
"Real-Time Computer Vision & Model Inference Link", "Control Loop State
Machine & Orchestrator"

## Why this doc exists

Sprint 3 assumes the board can run camera capture, model inference, and the
alert/buzzer control loop all at once. This doc turns the raw numbers from
`resource_consumption_report.md` and the concurrent-load test into concrete
guidance for that design, instead of leaving Sprint 3 to discover a
resource ceiling the hard way.

## Inputs

| Source | What it tells us |
|---|---|
| `resource_monitor.py` runs | Cost of preprocessing alone |
| `concurrent_load_test.py` runs | Cost of capture + preprocessing together |
| Sprint 1 `stability_test.py` log | Baseline cost of capture alone |

## Findings

*(Fill in once real runs are done — this is the section that actually
drives the Sprint 3 decisions below.)*

| Scenario | Avg CPU % | Peak CPU % | Peak Temp °C | Notes |
|---|---|---|---|---|
| Capture only (Sprint 1 baseline) | — | — | — | |
| Preprocessing only | — | — | — | |
| Capture + preprocessing concurrent | — | — | — | |

## Decisions for Sprint 3

*(Pick based on the table above once it's filled in — these are the
questions the numbers need to answer.)*

- [ ] **Can inference run on the same board as capture**, or does the
      control loop need to offload inference to a host machine / more
      powerful board?
- [ ] **Is a lighter/quantized model required** given the CPU headroom left
      after capture, or is the current model architecture fine?
- [ ] **Does the buzzer/alert GPIO signal need to run on a separate thread
      with priority**, given how much CPU capture + inference already use?
- [ ] **Is active cooling (fan/heatsink) required** for Sprint 3's longer
      real-time sessions, based on the peak temperatures observed here?

## Risk flag

If peak CPU during the concurrent test is already near saturation (>85%)
or thermal throttling is observed, flag this to the team **before** Sprint
3 hardware assembly starts — it changes the control-loop architecture
(e.g. may need `Temporal Decision & Moving Average Buffer` to run at a
lower sampling rate to leave headroom for inference).
