# Figure registry

One row per catalogue figure (spec `specs/001-final-deliverable/spec.md`). Files: `{id}_{slug}.png` and `.svg`,
written by `ride_classifier.viz.theme.save_fig`. Status: planned / draft / final.

| ID | Slug | What it shows | Story | Core / bonus | Source | Status |
|---|---|---|---|---|---|---|
| F01 |  | The two datasets side by side (rides, readings, cities, labels) | MLOps, infra | core | `01_data_overview` | planned |
| F02 |  | Category counts (500 rides) and imbalance | MLOps | core | `01_data_overview` | planned |
| F03 |  | One ride per category: the 6 sensor channels over time | MLOps | core | `01_data_overview` | planned |
| F04 | rotation_units | Rotation units: GBG500 vs SRV3490 distributions before and after conversion | MLOps | core | `02_units_and_cities` | draft |
| F05 | rides_falls_per_city | Rides and falls per city (SRV3490) | MLOps | core | `02_units_and_cities` | draft |
| F06 | ride_windows | A ride cut into 30 s windows (diagram) | both | core | `03_live_classifier` | draft |
| F07 | window_vs_cumulative | Per-window and cumulative probabilities along one golden ride | both | core | `03_live_classifier` | draft |
| F08 | golden_f1_vs_elapsed | Macro-F1 vs elapsed time with whole-ride reference scores | MLOps | core | `03_live_classifier` | draft |
| F09 | golden_confusion_matrix | Golden confusion matrix of the whole-ride label | MLOps | core | `04_evaluation` | draft |
| F10 | golden_quality_gate | Golden macro-F1 vs floor, majority and random references | MLOps | core | `04_evaluation` | draft |
| F11 | golden_reckless_predictions | Reckless predictions by known category on golden rides, with counts | both | core | `04_evaluation` | draft |
| F12 |  | Pseudo-label mix on SRV3490 by city vs GBG500 mix | MLOps | core | `05_srv3490_pseudo_labels` | planned |
| F13 |  | Model confidence on GBG500 vs SRV3490 | MLOps | core | `05_srv3490_pseudo_labels` | planned |
| F14 |  | Current system architecture diagram | both | core | `ai-infrastructure/architecture` | planned |
| F15 |  | Proposed (target) architecture diagram | infra | core | `ai-infrastructure/architecture` | planned |
| F16 |  | Pipeline graph (prepare, train, evaluate) | MLOps | core | pipeline tool output | planned |
| F17 |  | Experiment tracker: runs compared, registry versions and aliases (screenshots) | MLOps | core | screenshots | planned |
| F18 |  | CI/CD run: jobs, approval, deployment (screenshots) | MLOps | core | screenshots | planned |
| F19 |  | API documentation page and one answer with timeline (screenshots) | MLOps | core | screenshots | planned |
| F20 |  | Monitoring dashboards: service metrics during normal traffic (screenshot) | MLOps | core | screenshots | planned |
| F21 |  | Drift dashboard during each of the three scenarios (screenshots) | MLOps | core | `06_drift_scenarios` + screenshots | planned |
| F22 |  | Drift indicator per feature and scenario vs alert thresholds | MLOps | core | `06_drift_scenarios` | planned |
| F23 |  | Resource sizing table: data, model size, memory, training time | infra | core | `ai-infrastructure/benchmarks` | planned |
| F24 |  | Theoretical training memory vs available memory (log scale) | infra | core | `ai-infrastructure/benchmarks` | planned |
| F25 |  | Latency (mean, p95) by stage: cleaning, windows, network | infra | core | `ai-infrastructure/benchmarks` | planned |
| F26 |  | Latency and throughput vs batch size, CPU vs 1 GPU vs 2 GPUs | infra | core | `ai-infrastructure/benchmarks` | planned |
| F27 |  | Memory (RAM / VRAM) by hardware | infra | core | `ai-infrastructure/benchmarks` | planned |
| F28 |  | Optimization before/after: latency, throughput, memory, quality | infra | core | `ai-infrastructure/optimization` | planned |
| F29 |  | Training time and memory vs data volume (x1-x8), CPU / 1 GPU / 2 GPUs | infra | core | `ai-infrastructure/distributed-training` | planned |
| F30 |  | Data-parallel speedup and efficiency; small vs large batches (communication-bound case) | infra | core | `ai-infrastructure/distributed-training` | planned |
| F31 |  | Live load test: p95 latency vs number of live rides, limit line, capacity per instance | both | core | `ai-infrastructure/cost-resilience` | planned |
| F32 |  | Load test seen live on the dashboard (screenshot or recording) | both | core | screenshots | planned |
| F33 |  | Instances and monthly cost vs request volume, CPU vs GPU | infra | core | `ai-infrastructure/cost-resilience` | planned |
| F34 |  | Rollback timeline measured on the cluster | both | core | `ai-infrastructure/cost-resilience` | planned |
| F35 |  | Decision table: recommended infrastructure by volume, model size and latency target | infra | core | `ai-infrastructure/README` | planned |
| F36 | palette_test | Palette test (categories, heatmap, risk scales) | none (charter) | core | `00_visual_charter` | draft |
| F37 |  | Live replay of a ride (recording) | both | bonus | web app | planned |
| F38 |  | Fall forecast: probability before the fall vs comparison rides, alert trade-off | both | bonus | `09_fall_forecast` | planned |
| F39 |  | Falls as weak supervision: with vs without, same evaluation rides (only if F49 shows a signal) | MLOps | bonus | `08_falls_weak_supervision` | planned |
| F40 |  | Drift alert triggering retraining end to end (screenshot of the run) | MLOps | bonus | screenshots | planned |
| F41 |  | Four-layer monitoring dashboard: baseline and each incident (screenshots) | MLOps | core | screenshots | planned |
| F42 |  | Alerts going inactive → pending → firing during an incident (screenshot) | MLOps | core | screenshots | planned |
| F43 |  | Retraining policy diagram: triggers → workflow → gate → human approval → production, and the runbook fork (retrain vs fix upstream) | both | core | `docs/` diagram | planned |
| F44 |  | Outcome layer: performance on labels received later, before and after a drift scenario | MLOps | core | `06_drift_scenarios` | planned |
| F45 |  | Drift report (per-feature PSI and test) for each scenario | MLOps | core | drift job output | planned |
| F46 |  | Where to retrain: options vs measured training cost (local, CI runner, inference machine, Vertex AI) | both | core | `ai-infrastructure/distributed-training` | planned |
| F47 | window_length_study | Development fall-group paired AUC vs window length for the earlier label-free score | both | core | `03_live_classifier` | draft |
| F48 | cumulative_vs_sliding | Cumulative ride type vs current sliding-window fall risk | both | core | `03_live_classifier` | draft |
| F49 | fall_risk_gap_check | Held-out window-level ROC curves for the 5 s and 10 s anti-leakage gaps (SRV3490) | both | core | `07_fall_risk` | draft |
