# Implementation Plan: Final deliverable (two presentations on one shared base)

**Branch**: none (work stays on `dev_vincent`) | **Date**: 2026-11-20 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-final-deliverable/spec.md`; decisions in `docs/PROJECT_FRAMING.md`;
rules in `.specify/memory/constitution.md` v1.2.0; course labs `lectures/mlops-labs/06-monitoring` (four-layer
monitoring, drift job, simulator, runbooks) and `05-kubernetes` (Locust, capacity plot, canary).

## Summary

Turn the existing ride classifier (E3, a 1D CNN that already scores 30 s windows and averages them per ride) into
a **cumulative live classifier** by exposing the running mean of the window scores, without retraining first:
the whole-ride label stays exactly E3's, so the gate result is unchanged and the live view is new. Around it,
build what both juries must see: a stateless window-by-window API, four-layer monitoring with Prometheus and
Grafana, a drift monitor and a drift job fed by real resampled rides (unseen city, real unit bug, riskier mix),
GitHub Actions workflows for training, deployment with human approval and drift flags, a live load test (one
Locust user = one ride at real speed), and the `ai-infrastructure/` measurements (one `bench.py`, Vertex AI SPOT
jobs for 1 and 2 GPUs, ONNX Runtime / int8, training cost on x1-x8 data). Bonus (live replay, fall forecast,
encoder, automatic retraining) comes last.

Order: **Phase A (core MLOps, safe first)** and **Phase B (AI infra measurements)** run in parallel with
different owners; **Phase C (presentation assets)** is fed continuously; **Phase D (bonus)** starts only after
A and B pass their checkpoints.

## Technical Context

**Language/Version**: Python 3.14 (uv, `uv.lock`); Vertex AI jobs use the prebuilt
`europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310` image (Python 3.10: `bench.py` and the model
code must stay 3.10-compatible, see research R10).

**Primary Dependencies**: existing: PyTorch (CPU index), pandas, scikit-learn, scipy, DVC[s3], MLflow 3.17,
FastAPI + Pydantic, prometheus-client, Streamlit, httpx. To add: `locust` (group `loadtest`), `onnx` +
`onnxruntime` (group `infra`), `seaborn` (group `train`, for the Mako scale), `evidently` (group `train`, drift
report, optional). Prometheus and Grafana as container images.

**Storage**: SeaweedFS (S3-compatible) for DVC data and MLflow artifacts; Postgres for MLflow; Prometheus TSDB;
API prediction log as JSON lines (existing `PREDICTION_LOG_PATH`); GCP: private Artifact Registry repository in
`advanced-cloud-ai-i2`, private bucket in `agentics-workflows-2` only if needed (research R9).

**Testing**: pytest (unit, integration, e2e folders exist; e2e is empty and gets the replay test), ruff;
CI runs unit + integration + container smoke test.

**Target Platform**: laptop (Windows, ~12 GB RAM) with Docker Compose (dev) and kind (production); GitHub Actions
hosted runners for CI, a self-hosted runner on the laptop for training and deployment; Vertex AI custom jobs
(europe-west4, SPOT) for GPU measurements; Colab/Kaggle as free fallback.

**Project Type**: ML system (pipeline + web service + web app + monitoring) with a measurement sub-project.

**Performance Goals**: one live window answered under 200 ms p95 up to the measured capacity; a whole ride
under 1 s (SC-004); capacity of one CPU pod expressed in live rides (SC-005).

**Constraints**: no synthetic data; golden rides only in `evaluate`; no data or model published (public GitHub
repo; published Docker Hub images carry no data and no trained weights, the API loads the model from the MLflow
registry at startup: see research R12); every GPU run self-terminating; laptop
memory: one heavy service set at a time (compose **or** kind, not both).

**Scale/Scope**: GBG500 500 rides / 1.5 M readings; SRV3490 3,490 rides / 16.9 M readings; model 45 k parameters,
2.5 MB; 46 catalogue figures (40 core); 5 people, deadline 25/11/2026 20:00.

One choice is deliberately **open**: the window length (R15). 30 s / 15 s is E3's untested choice, not the data's
granularity (10 Hz); every number derived from it (300-reading minimum, window payload, update every 15 s, load
rate) is provisional until the study A1b decides. Other open points are resolved in [research.md](research.md)
(R1-R14). The one external
blocker is getting SRV3490 (R3): everything that needs it has a GBG500-only fallback.

## Constitution Check

*GATE: checked before Phase 0 and again after Phase 1 design.*

| Principle | How this plan complies | Status |
|---|---|---|
| I. Teachers' requirements | Every MLOps objective and format item and every AI infra step 1-5 maps to a phase item below and to the README compliance table (SC-012). All taught tools used: DVC, MLflow, SeaweedFS, FastAPI, Docker Hub, kind, GitHub Actions + self-hosted runner, Prometheus + Grafana, Locust, PSI/KS/chi², PyTorch profiler, ONNX Runtime, DDP, Vertex AI. Stated gaps: image input, GCP instead of AWS, feature store (FR-008) | pass |
| II. Presentation first | Each work item names its figure IDs (F01-F46); theme module + `save_fig` in Phase A0, before any figure; bonus isolated in Phase D | pass |
| III. Measure, do not claim | One `bench.py`, raw CSVs with environment columns, `status` column = measured/estimated (contract `benchmark-results.md`) | pass |
| IV. Honest ML | v1 cumulative model = running mean of E3, named as such; golden untouched (thresholds calibrated on training rides); fall groups never split; per-window scores labelled "hint" in API and UI | pass |
| V. Reuse | Model contract kept (`predict_timeline` added next to `predict_proba`, research R1); same pipeline, registry, images, kind manifests; E3 stays the production fallback | pass |
| VI. Real data only | Scenarios = resampled real rides; replication x2-x8 labelled; unit factor verified before conversion (R2) | pass |
| VII. Cost and data safety | SPOT Vertex jobs only, teardown checklist and credit log per session (SC-009); data stays in private stores | pass |
| VIII. Discipline | English, `dev_vincent`, explicit staging, typed code, params in `params.yaml` (new `drift`, `monitoring`, `loadtest` sections), tests with each non-trivial module | pass |

Post-design re-check (after Phase 1): still pass. One recorded deviation: the feedback endpoint receives the
predicted label from the caller instead of looking it up (research R6): a demo simplification, stated in the
README and the runbook.

## Project Structure

### Documentation (this feature)

```text
specs/001-final-deliverable/
├── plan.md              # this file
├── research.md          # Phase 0: decisions R1-R14
├── data-model.md        # Phase 1: entities, fields, states
├── quickstart.md        # Phase 1: validation scenarios
├── contracts/
│   ├── api.md                 # REST endpoints: timeline, window streaming, feedback, metrics
│   ├── metrics-and-alerts.md  # Prometheus metric names, alert rules, runbook anchors
│   ├── drift-report.md        # drift job CLI, exit codes, report schema
│   ├── benchmark-results.md   # bench.py CLI and CSV schema
│   ├── workflows.md           # GitHub Actions workflows, triggers, inputs, gates
│   └── figures.md             # theme module, save_fig, figure registry
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks (next)
```

### Source code (repository root)

```text
src/ride_classifier/
├── models/cnn.py            # + predict_timeline(): per-window probabilities and running mean (R1)
├── models/timeline.py       # NEW pure functions: cumulative mean, F1-vs-elapsed-time curve
├── data/srv3490.py          # NEW load + unit harmonisation + fall groups (R2, R3)
├── pipeline/prepare.py      # + SRV3490 stage outputs; + drift reference (window feature deciles)
├── pipeline/evaluate.py     # + macro-F1 vs elapsed time (golden), saved as metrics + CSV
├── pipeline/pseudo_label.py # NEW: SRV3490 pseudo-labels with confidence (F12, F13)
├── features/window_stats.py # NEW: per-window monitored features (shared by API, drift job, reference)
├── api/main.py, schemas.py  # + /v1/predict timeline, /v1/predict/window, /v1/feedback (contracts/api.md)
├── api/metrics.py           # + layer 2 (input deciles) and layer 4 (feedback) metrics
├── drift/psi.py             # NEW: psi / ks / chi2 (ported from lab 06, typed)
├── drift/job.py             # NEW: offline drift job CLI, report, exit code 3
├── drift/monitor.py         # NEW: live drift exporter (queries Prometheus, exposes drift_* gauges)
├── drift/scenarios.py       # NEW: build the scenario streams from real rides (R7)
├── replay/client.py         # NEW: replays rides window by window to the API (+ feedback); used by scenarios
└── viz/palette.py, theme.py # NEW: charter constants (no deps) + matplotlib theme + save_fig

loadtest/locustfile.py, shape.py, plot_capacity.py   # live-ride users, stepped load, F31
monitoring/prometheus/{prometheus.yml,alerts.yml}, monitoring/grafana/{provisioning,dashboards}/
k8s/monitoring/                                      # same config as ConfigMaps for kind
docs/runbooks.md, docs/retraining-policy.md          # four layers + runbook fork (F43)
.github/workflows/ci.yml (extended), train.yml, cd.yml, drift.yml   # contracts/workflows.md
notebooks/00_visual_charter … 08_encoder.ipynb       # figure notebooks (F01-F13, F22, F36, F44)
assets/figures/                                      # PNG + SVG, one pair per catalogue ID
ai-infrastructure/                                   # layout imposed by the brief
├── README.md                     # inventory, summary tables, decision table (F35)
├── architecture/current-system.png, proposed-system.png (F14, F15)
├── benchmarks/bench.py, benchmark.ipynb, baseline-results.csv, results/*.csv   # F23-F27
├── optimization/comparison.ipynb, export_onnx.py                              # F28
├── distributed-training/technical-analysis.md, train_ddp.py, results/          # F29, F30, F46
├── cost-resilience/infrastructure-plan.md, capacity.ipynb                      # F31, F33, F34
└── vertex/submit.py, README.md   # SPOT job submission (code tarball, CSV in logs)
tests/unit/{models,drift,features,api,viz}/…, tests/integration/…, tests/e2e/test_replay.py
```

**Structure Decision**: keep the single-package layout; new code lands in the existing sub-packages
(`models`, `api`, `drift`, `features`, `pipeline`) plus two small new ones (`replay`, `viz`). The course-imposed
`ai-infrastructure/` folder holds the measurement code that is not part of the product. Monitoring config sits in
`monitoring/` (compose) and is mounted as ConfigMaps in `k8s/monitoring/` (kind), one source of truth.

## Phases and checkpoints

Each item lists: what it delivers → figures/demos → requirement. Owners are assigned in `/speckit-tasks`.

### Phase A - Core MLOps (safe first)

| # | Item | Figures / demo | FR |
|---|---|---|---|
| A0 | Visual charter: `viz/palette.py`, `viz/theme.py`, `save_fig`, notebook `00_visual_charter` (Mako vs alternatives, Heat), freeze the palette | F36 | FR-003 |
| A1 | Data overview notebook on GBG500 (SRV3490 parts added when available) | F01-F03 | FR-002 |
| A1b | **Window-length study (R15), before A2/A3 freeze anything**: 5/10/15/30/60 s on train-rides CV, rules written before running, golden untouched; decision written in `docs/PLAN.md` §6.5 | F47 | FR-010, FR-011 |
| A2 | Cumulative classifier v1 (window from A1b): `predict_timeline` + `timeline.py` (pure, tested); evaluate adds macro-F1 vs elapsed time (30, 60, 90, 120, 180, 240 s, end) for model, E3-as-whole-ride and random/majority | F06-F11 | FR-010, FR-011, SC-003 |
| A3 | API: timeline in `/v1/predict`, stateless `/v1/predict/window`, `/v1/feedback`; schemas and Swagger examples; unit tests | F19 | FR-013, FR-019 |
| A4 | Monitoring stack in compose: Prometheus + Grafana provisioned, four-layer dashboard, 5 alerts (contracts/metrics-and-alerts.md), runbooks | F20, F41, F42 | FR-015 |
| A5 | Drift: `window_stats`, drift reference stage, `drift/psi.py`, offline `drift/job.py` (report + exit 3), live `drift/monitor.py` exporter | F45 | FR-018 |
| A6 | Replay client + scenarios: normal, riskier mix (GBG500 only, available now), unseen city and unit bug (need SRV3490) | F21, F22, F44 | FR-016, FR-019 |
| A7 | Workflows: `train.yml` (schedule, dispatch with reason, self-hosted), `cd.yml` (build, push, `production` environment approval, kind rollout, smoke, rollback), `drift.yml` (scheduled drift job, flag + summary); `ci.yml` adds e2e | F18, F43 | FR-014, FR-034, FR-036 |
| A8 | kind: monitoring manifests, CPU-limited API (one "instance" = 1 replica with a fixed CPU limit), canary kept | F32, F34 | FR-014, FR-025 |
| A9 | SRV3490 in the pipeline: DVC tracking, unit factor verified, harmonised, fall groups, pseudo-labels | F04, F05, F12, F13 | FR-012 |
| A10 | Drift demo in action: run each scenario on the kind deployment, record Grafana and the alert list, apply the runbook (retrain via `train.yml` for the new city; fix upstream for the unit bug) | F21, F40-F44 | FR-005, FR-034 |

**Checkpoint A (target 23/11 evening)**: quickstart scenarios Q1-Q6 pass on kind; F01-F13, F16-F22, F41-F45
saved. If SRV3490 is still missing, A9 and two scenarios are reported as blocked and the riskier-mix scenario
carries the demo.

### Phase B - AI infrastructure measurements (parallel to A)

| # | Item | Figures | FR |
|---|---|---|---|
| B1 | Inventory + current-system diagram; resource sizing table (data, model, memory, training time) | F14, F23, F24 | FR-009, FR-020 |
| B2 | `bench.py` (one tool; CSV per contract) + local CPU baseline; profiler stage breakdown | F25, F26 (CPU) | FR-021 |
| B3 | Vertex SPOT jobs: CPU, 1 GPU, 2 GPUs inference with the same `bench.py`; refusals kept as evidence | F26, F27 | FR-022 |
| B4 | Optimization: ONNX Runtime and dynamic int8 on CPU vs PyTorch baseline, quality agreement | F28 | FR-024 |
| B5 | Training cost: x1/x2/x4/x8 data on CPU, 1 GPU, 2 GPUs (DDP); speedup, efficiency; where to retrain | F29, F30, F46 | FR-023, FR-035 |
| B6 | Live load test on kind (needs A3, A8): stepped live rides until p95 > 200 ms; Grafana recording | F31, F32 | FR-025 |
| B7 | Scaling, cost, resilience plan: instances and monthly cost for 3+ volumes, rollback timed on kind; proposed architecture; decision table | F15, F33-F35 | FR-026 |

**Checkpoint B (target 24/11 evening)**: every number in `ai-infrastructure/` labelled and traceable (SC-006);
GCP credit log and teardown check filled (SC-009).

### Phase C - Presentation and submission (continuous, closes 25/11)

C1 story outlines per course (FR-001) · C2 demo fallbacks recorded (FR-004) · C3 README: choices, results,
screenshots, Docker Hub URLs, stated gaps, compliance table (FR-006, FR-008, SC-012) · C4 timed dry runs
(SC-001) · C5 submission email before 25/11 20:00 (FR-007).

### Phase D - Bonus (only after checkpoints A and B)

D1 live replay in Streamlit (F37, FR-030) · D2 fall forecast (F38, FR-031) · D3 automatic retraining on alert
via Alertmanager webhook → `repository_dispatch` (F40, FR-033) · D4 self-supervised encoder (F39, FR-032).

## Downstream updates (ripple effects)

README (architecture, how to run monitoring, compliance table), `docs/how_to_use.md` (new endpoints, make
targets), `docs/PLAN.md` DEC-1/2/4 and `docs/USE_CASE.md` §3/§5 aligned with PROJECT_FRAMING, `params.yaml` new
sections, `Makefile` targets (`monitoring-up`, `drift`, `scenario`, `loadtest`, `bench`), `pyproject.toml` groups,
`dvc.yaml` stages (`prepare_srv`, `reference`, `pseudo_label`), webapp palette, CI image list.

## Complexity Tracking

| Choice | Why needed | Simpler alternative rejected because |
|---|---|---|
| Separate live drift exporter (`drift/monitor.py`) next to the offline job | PSI must move live in Grafana (FR-005); API replicas cannot share a log file on kind | PSI in PromQL alone is unreadable; a shared RWX volume is not available on kind |
| Two GitHub environments-based gates (gate in pipeline + human approval in `cd.yml`) | FR-034 human in the loop, graded CI/CD | Auto-promotion contradicts the hybrid policy |
