# Figure registry

One row per catalogue figure (spec `specs/001-final-deliverable/spec.md`). Files: `{id}_{slug}.png` and `.svg`,
written by `ride_classifier.viz.theme.save_fig`. Status: planned / draft / final.

| ID | Slug | What it shows | Story | Core / bonus | Source | Status |
|---|---|---|---|---|---|---|
| F01 | | The two datasets side by side (rides, readings, cities, labels) | MLOps, infra | core | `01_data_overview` | planned |
| F02 | | Category counts (500 rides) and imbalance | MLOps | core | `01_data_overview` | planned |
| F03 | | One ride per category: the 6 sensor channels over time | MLOps | core | `01_data_overview` | planned |
| F04 | | Rotation units: GBG500 vs SRV3490 distributions before and after conversion | MLOps | core | `02_units_and_cities` | planned |
| F05 | | Rides and falls per city (SRV3490) | MLOps | core | `02_units_and_cities` | planned |
| F06 | | A ride cut into 30 s windows (diagram) | both | core | `03_live_classifier` | planned |
| F07 | | Per-window scores vs cumulative label along one ride | both | core | `03_live_classifier` | planned |
| F08 | | Macro-F1 vs elapsed ride time (cumulative classifier vs reference vs random) | MLOps | core | `03_live_classifier` | planned |
| F09 | | Golden confusion matrix of the whole-ride label | MLOps | core | `04_evaluation` | planned |
| F10 | | Gate: golden macro-F1 of candidates vs floor, majority and random references | MLOps | core | `04_evaluation` | planned |
| F11 | | Reckless rides: detected k / n with false alarms | MLOps | core | `04_evaluation` | planned |
| F12 | | Pseudo-label mix on SRV3490 by city vs GBG500 mix | MLOps | core | `05_srv3490_pseudo_labels` | planned |
| F13 | | Model confidence on GBG500 vs SRV3490 | MLOps | core | `05_srv3490_pseudo_labels` | planned |
| F14 | | Current system architecture diagram | both | core | `ai-infrastructure/architecture` | planned |
| F15 | | Proposed (target) architecture diagram | infra | core | `ai-infrastructure/architecture` | planned |
| F16 | | Pipeline graph (prepare, train, evaluate) | MLOps | core | pipeline tool output | planned |
| F17 | | Experiment tracker: runs compared, registry versions and aliases (screenshots) | MLOps | core | screenshots | planned |
| F18 | | CI/CD run: jobs, approval, deployment (screenshots) | MLOps | core | screenshots | planned |
| F19 | | API documentation page and one answer with timeline (screenshots) | MLOps | core | screenshots | planned |
| F20 | | Monitoring dashboards: service metrics during normal traffic (screenshot) | MLOps | core | screenshots | planned |
| F21 | | Drift dashboard during each of the three scenarios (screenshots) | MLOps | core | `06_drift_scenarios` + screenshots | planned |
| F22 | | Drift indicator per feature and scenario vs alert thresholds | MLOps | core | `06_drift_scenarios` | planned |
| F23 | | Resource sizing table: data, model size, memory, training time | infra | core | `ai-infrastructure/benchmarks` | planned |
| F24 | | Theoretical training memory vs available memory (log scale) | infra | core | `ai-infrastructure/benchmarks` | planned |
| F25 | | Latency (mean, p95) by stage: cleaning, windows, network | infra | core | `ai-infrastructure/benchmarks` | planned |
| F26 | | Latency and throughput vs batch size, CPU vs 1 GPU vs 2 GPUs | infra | core | `ai-infrastructure/benchmarks` | planned |
| F27 | | Memory (RAM / VRAM) by hardware | infra | core | `ai-infrastructure/benchmarks` | planned |
| F28 | | Optimization before/after: latency, throughput, memory, quality | infra | core | `ai-infrastructure/optimization` | planned |
| F29 | | Training time and memory vs data volume (x1-x8), CPU / 1 GPU / 2 GPUs | infra | core | `ai-infrastructure/distributed-training` | planned |
| F30 | | Data-parallel speedup and efficiency; small vs large batches (communication-bound case) | infra | core | `ai-infrastructure/distributed-training` | planned |
| F31 | | Live load test: p95 latency vs number of live rides, limit line, capacity per instance | both | core | `ai-infrastructure/cost-resilience` | planned |
| F32 | | Load test seen live on the dashboard (screenshot or recording) | both | core | screenshots | planned |
| F33 | | Instances and monthly cost vs request volume, CPU vs GPU | infra | core | `ai-infrastructure/cost-resilience` | planned |
| F34 | | Rollback timeline measured on the cluster | both | core | `ai-infrastructure/cost-resilience` | planned |
| F35 | | Decision table: recommended infrastructure by volume, model size and latency target | infra | core | `ai-infrastructure/README` | planned |
| F36 | | Palette test (categories, heatmap, risk scales) | none (charter) | core | `00_visual_charter` | planned |
| F37 | | Live replay of a ride (recording) | both | bonus | web app | planned |
| F38 | | Fall forecast: probability before the fall vs comparison rides, alert trade-off | both | bonus | `07_fall_forecast` | planned |
| F39 | | Encoder vs core model on golden rides | MLOps | bonus | `08_encoder` | planned |
| F40 | | Drift alert triggering retraining end to end (screenshot of the run) | MLOps | bonus | screenshots | planned |
| F41 | | Four-layer monitoring dashboard: baseline and each incident (screenshots) | MLOps | core | screenshots | planned |
| F42 | | Alerts going inactive → pending → firing during an incident (screenshot) | MLOps | core | screenshots | planned |
| F43 | | Retraining policy diagram: triggers → workflow → gate → human approval → production, and the runbook fork (retrain vs fix upstream) | both | core | `docs/` diagram | planned |
| F44 | | Outcome layer: performance on labels received later, before and after a drift scenario | MLOps | core | `06_drift_scenarios` | planned |
| F45 | | Drift report (per-feature PSI and test) for each scenario | MLOps | core | drift job output | planned |
| F46 | | Where to retrain: options vs measured training cost (local, CI runner, inference machine, Vertex AI) | both | core | `ai-infrastructure/distributed-training` | planned |
| F47 | | Window-length study: whole-ride CV macro-F1 and cumulative macro-F1 at 60 s per window length (5-60 s), with the decision rule | both | core | `03_live_classifier` | planned |
