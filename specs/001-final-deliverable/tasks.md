# Tasks: Final deliverable (two presentations on one shared base)

**Input**: `specs/001-final-deliverable/` (plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md)

**Granularity**: the first steps (P0: #10 A0, #11 A1b, #12 SRV3490, #3 bench tool) are unit tasks; later work is
one task per deliverable and is split further when it starts (`/speckit-tasks` rerun or `/speckit-converge`).

**Tests**: included where the constitution requires them (VIII: non-trivial logic ships with a test).

**Format**: `- [ ] T### [P?] [US?] description (#issue)`. `[P]` = different files, no unfinished dependency.
US1 = MLOps jury, US2 = AI infra jury, US3 = live replay (bonus), US4 = fall warning (bonus).

---

## Phase 1: Setup

- [x] T001 Add dependency groups in `pyproject.toml`: `seaborn` to `train`, new `loadtest` (`locust`) and `infra` (`onnx`, `onnxruntime`, `psutil`); `uv lock`; check `uv sync` installs (#10, #3)
- [x] T002 [P] Create `assets/figures/`, `assets/demos/` with `assets/figures/README.md` registry table F01-F47 (id, slug, story, core/bonus, source, status `planned`) from the spec Figure Catalogue (#10)
- [x] T003 [P] Create the `ai-infrastructure/` skeleton of the brief: `README.md`, `architecture/`, `benchmarks/`, `optimization/`, `distributed-training/`, `cost-resilience/`, `vertex/` with one-line READMEs (#3, #8)
- [x] T004 (dropped: each task adds its own params section) [P] Add empty `drift`, `monitoring`, `loadtest` sections with comments to `params.yaml` (values filled by their tasks) (#16, #17, #7)

---

## Phase 2: Foundational (P0, blocks everything else)

### A0 - Visual charter (#10)

- [x] T005 [P] Write `src/ride_classifier/viz/__init__.py` and `src/ride_classifier/viz/palette.py`: `CATEGORY_COLORS` (Okabe-Ito, Reckless `#D55E00`), `MODEL_COLOR = "#0072B2"`, `REFERENCE_COLORS` (majority, random, reference model in greys), `HEATMAP = "mako"`, `RISK = "YlOrRd"`; no third-party import (contracts/figures.md) (#10)
- [x] T006 Write `src/ride_classifier/viz/theme.py`: `apply_theme()` (white figure and axes background, slide font sizes, no top/right spines, category colour cycle) and `save_fig(fig, fig_id, slug) -> tuple[Path, Path]` writing `assets/figures/{fig_id}_{slug}.png` (200 dpi) and `.svg`, rejecting ids not matching `F\d{2}` or absent from the registry (contracts/figures.md) (#10)
- [x] T007 [P] Test `tests/unit/viz/test_theme.py`: both files written in `tmp_path`, unknown id raises `ValueError`, palette has one colour per category of `config.LABELS` (#10)
- [x] T008 Notebook `notebooks/00_visual_charter.ipynb`: category bars, a confusion-matrix-like heatmap in Mako vs viridis, a risk ramp in YlOrRd, model vs references; save F36; record the choice in `assets/figures/README.md` and freeze the palette (#10)

### Timeline core (part of A2 #14) — done

- [x] T009 [P] Write `src/ride_classifier/models/timeline.py`: `cumulative_mean(window_probas: np.ndarray) -> np.ndarray` (data-model.md `Timeline.cumulative`) and `f1_vs_elapsed(timelines, labels, times_s) -> pd.DataFrame` (data-model.md "Evaluation records", vs elapsed time) (#14)
- [x] T010 [P] Test `tests/unit/models/test_timeline.py`: running mean on a 3-window example, last row equals the mean of all windows, `f1_vs_elapsed` on a 4-ride toy set (#14)
- [x] T011 Add `CNNModel.predict_timeline(inputs) -> dict[str, Timeline]` in `src/ride_classifier/models/cnn.py` returning per-window probabilities with `end_s` (data-model.md `Window.window_index`, `Window.end_s`) and cumulative values; test in `tests/unit/models/test_cnn.py` that the last cumulative equals `predict_proba` within 1e-6 (data-model.md Timeline invariant) (#14)

### Danger detector (R1, #14) — reframed 21/11 after the course `data.txt`

- [x] T055 Write `src/ride_classifier/pipeline/falls.py`: seeded split of fall groups into development (70 %) and held-out (30 %) (data-model.md `Ride.fall_group_id`: a group never split), saved to `data/processed/fall_groups.csv` (committed, like `splits.csv`); `pre_fall_segment(ride, window_s, length_s, margin_s)` ending `margin_s` before the first topple (impact excluded), and the matching elapsed time on comparison rides; tests in `tests/unit/pipeline/test_falls.py` (#29)
- [x] T056 Write `src/ride_classifier/models/autoencoder.py`: `RideAutoencoder` (encoder = E3's 3 conv blocks, decoder = mirror with upsampling, input windows standardised per channel on train windows) and `DangerModel` following the model contract (`prepare` / `fit(inputs, labels=None)` / `predict_proba` / `predict_timeline`): `fit` trains the autoencoder on all given windows WITHOUT labels, then the 5-class head (logistic regression on the mean embedding per ride, class weights as E3) on the labelled rides only; danger score per window = reconstruction error z-scored on train windows; timeline carries both the danger score and the category probabilities (extend `models/timeline.Timeline` with an optional `danger` array); tests in `tests/unit/models/test_autoencoder.py` (shapes, no label used by the encoder, last cumulative danger = whole-ride danger) (#14)
- [ ] T057 (partly done: stage `prepare_srv` writes the clean SRV3490 table) Training data for the encoder in `pipeline/prepare.py` or a new stage: GBG500 rides + SRV3490 rides harmonised with `data.srv3490.harmonise_rotation`, cleaned like GBG500, minus held-out fall groups; params `train.model: danger`, `train.danger.*` in `params.yaml`; RAM check (17 M readings: build windows per ride, float32) (#14, #22)
- [ ] T058 Smoke-train `DangerModel` at 30 s on a subset, log to MLflow, check the danger score distribution of Safe train rides and the time per epoch; report before the full study (#14)

### A1b - Window-length study (#11), runs on the danger detector

- [x] T012 Pre-registration `docs/experiments/window_study.md` (revised 21/11 before any run for the danger detector) (#11)
- [x] T013 Time steps after pooling: 6 at 5 s, no adaptation (#11)
- [x] T014 Rewrite `src/ride_classifier/experiments/window_study.py` for the R15 rule: per candidate train `DangerModel` (T056) on the development data, compute (a) paired AUC pre-fall vs comparison on development fall groups with bootstrap CI, (b) ROC-AUC Reckless vs counterfactuals on GBG500 train rides, (c) false-alarm rate per counterfactual category at the train threshold, (d)-(f) as pre-registered; golden and held-out groups never read; one CSV row per candidate (#11)
- [x] T015 Rewrite `choose_window` and its tests in `tests/unit/experiments/test_window_study.py` for the new rule ((a) within the best's CI, then max (b), tie -> shorter) (#11)
- [ ] T016 Run the study on the laptop with MLflow on, `artifacts/window_study.csv` (#11) — IN PROGRESS: 30 s done (MLflow run 35c834ee), 60/15/10 s launched 21/11, 5 s after (may need Vertex if RAM is short); resume with the same command, finished candidates are skipped
- [ ] T017 F47 in `notebooks/03_live_classifier.ipynb` with `save_fig` (#11)
- [ ] T018 Apply the decision: `params.yaml` window, results in `docs/experiments/window_study.md` and `docs/PLAN.md` §6.5, R15 marked decided (#11)

### SRV3490 (#12)

- [x] T019 Obtain SRV3490 (course platform or teammate), place it in `data/raw/`, `uv run dvc add`, push to the S3 remote; commit only the `.dvc` file (#12)
- [x] T020 Write `src/ride_classifier/data/srv3490.py`: load, check no id overlaps GBG500 (data-model.md `Ride.ride_id`), derive `city`, `fall_time_index`, `fall_group_id` (data-model.md `Ride.fall_time_index`, `Ride.fall_group_id`) (#12)
- [x] T021 In `notebooks/02_units_and_cities.ipynb`, measure the per-ride rotation std ratio GBG500 / SRV3490 and compare with 180/pi; write the verified factor to `params.yaml` `data.srv3490.rotation_factor`; add `harmonise_rotation()` to `data/srv3490.py` (data-model.md `Ride.readings`, rotation in deg/s) with a test in `tests/unit/data/test_srv3490.py` (#12)

### Benchmark tool (#3)

- [x] T022 Write `ai-infrastructure/benchmarks/make_windows.py`: save a fixed set of golden-split windows from the reference model's preprocessing to `ai-infrastructure/benchmarks/windows.npy` (git-ignored, regenerated) — read-only use of golden, no choice made from it (#3)
- [x] T023 Write `ai-infrastructure/benchmarks/bench.py` subcommand `inference`: Python 3.10-compatible, warm-up, repeats, mean and p95, `torch.cuda.synchronize`, throughput, peak RAM (psutil) and VRAM, environment columns, `status = measured`; output CSV with exactly the columns of contracts/benchmark-results.md; optional `@@CSV`/`@@END` printing (#3)
- [x] T024 [P] Test `tests/unit/infra/test_bench.py`: CSV header equals the contract columns, p95 computed on a known list, runs on CPU with a 2-window input (#3)
- [ ] T025 Add subcommand `stages` (torch.profiler / timing of cleaning, windowing, network on whole rides) to `bench.py` (#3, #4)
- [ ] T026 Resource sizing in `ai-infrastructure/benchmarks/sizing.md` and `results/sizing.csv`: dataset sizes, rides, readings, input shape, parameter count, state-dict and MLflow model size, training duration and peak RAM on the laptop, theoretical training memory (16 bytes/parameter with Adam); single-machine conclusion (#3)
- [ ] T027 Notebook `ai-infrastructure/benchmarks/benchmark.ipynb` (Colab-runnable from a clone): read results CSVs, F23 and F24 with `save_fig` (#3)

**Checkpoint P0**: palette frozen (F36), window length decided (F47), SRV3490 harmonised or reported blocked,
`bench.py` producing contract CSVs locally.

---

## Phase 3: User Story 1 - MLOps jury (P1) 🎯 MVP

**Goal**: one ride followed through the whole system on kind, with monitoring and drift in action.
**Independent test**: quickstart Q1-Q6 on the production cluster, timed, every demo with a fallback.

- [ ] T028 [P] [US1] Data overview notebook `notebooks/01_data_overview.ipynb`: F01-F03 (#13)
- [ ] T029 [US1] `pipeline/evaluate.py` per research R16: danger score on all labelled rides (Reckless k / 6, false alarms per counterfactual category, ROC-AUC with bootstrap CIs, threshold from train rides), pre-fall vs comparison on held-out fall groups, 5-class head on golden, each vs E3 and random, and vs elapsed time; gate = better than random on Reckless detection and on the 5 categories; F06-F11, F48 in `notebooks/03_live_classifier.ipynb` and `04_evaluation.ipynb`, F49 in `07_falls_analysis.ipynb` (#14, #29)
- [x] T030 [US1] API per contracts/api.md: timeline in `/v1/predict`, stateless `/v1/predict/window` (data-model.md `Live window state.sum_proba`, `Live window state.n_windows`; narrowing: `n_windows` must equal `window_index`), `/v1/feedback`, `/health.drift_reference_loaded`; tests in `tests/unit/api/`; F19 screenshots (#15)
- [ ] T031 [US1] `features/window_stats.py` + DVC stage `reference` → `artifacts/drift_reference.json` (data-model.md "Drift reference"); `drift/psi.py` with tests (#17)
- [ ] T032 [US1] Layers 2-4 metrics in `api/metrics.py` per contracts/metrics-and-alerts.md; Prometheus + Grafana in `docker-compose.yml`, `monitoring/prometheus/{prometheus.yml,alerts.yml}`, provisioned dashboard `monitoring/grafana/dashboards/ride-api.json`; `docs/runbooks.md` (#16)
- [ ] T059 [US1] CI test of the unit-bug drift, `tests/integration/test_drift_unit_bug.py` (runs in `ci.yml`, no course data: tiny in-memory windows): the same windows sent once converted and once with the source conversion skipped (rotation / 57.3, what an upstream sender in rad/s produces) -> the model's own preprocessing is unchanged in both cases; the drift job returns exit 3 for the skipped conversion with the rotation features flagged and exit 0 for the converted ones; on the self-hosted runner, `drift.yml` replays the real `unit_bug` scenario (#17, #19)
- [ ] T033 [US1] `drift/job.py` (report JSON/MD, exit 3) and `drift/monitor.py` (live `drift_*` gauges) per contracts/drift-report.md; tests; `make drift` (#17)
- [ ] T034 [US1] `replay/client.py` and `drift/scenarios.py` (data-model.md "Drift scenario" `name`, `has_labels`); `make scenario NAME=...`; calibrate thresholds on `normal` in `params.yaml`; F22, F44 in `notebooks/06_drift_scenarios.ipynb` (#18)
- [ ] T035 [US1] Workflows per contracts/workflows.md: `.github/workflows/train.yml`, `cd.yml` (environment `production`), `drift.yml`; e2e test `tests/e2e/test_replay.py` added to `ci.yml`; `docs/retraining-policy.md` + F43 (data-model.md "Retraining trigger" `reason`) (#19)
- [ ] T036 [US1] Code-only Docker Hub images, model from `MODEL_URI` at startup; existing `-m2` tags handled with the account owner (#23)
- [ ] T037 [US1] kind: `k8s/monitoring/` manifests from the same configs, CPU-limited API replica, canary kept (#20)
- [ ] T038 [US1] SRV3490 in the pipeline: DVC stage `prepare_srv` (harmonised, cleaned), predicted category mix by city; F12, F13 (#22) — degradable
- [ ] T039 [US1] Drift demo in action on kind: run each scenario, record Grafana and alerts, apply the runbook (retrain vs upstream fix); F21, F40-F44 and recordings in `assets/demos/` (#21)

**Checkpoint A (23/11)**: quickstart Q1-Q6 pass.

---

## Phase 4: User Story 2 - AI infrastructure jury (P1, parallel to Phase 3)

**Goal**: every infrastructure choice backed by a measurement or a labelled estimate.
**Independent test**: every number in `ai-infrastructure/` traceable to a raw CSV and a procedure (SC-006).

- [ ] T040 [P] [US2] Current-system diagram `ai-infrastructure/architecture/current-system.png` (F14) and inventory in `ai-infrastructure/README.md` (#8)
- [ ] T041 [US2] Local CPU baseline with `bench.py` (batch 1-512 windows) → `benchmarks/baseline-results.csv`; F25, F26 (CPU) (#4)
- [ ] T042 [US2] `ai-infrastructure/vertex/submit.py` (SPOT, europe-west4, code tarball, CSV from logs, `--account=vlamy@albertschool.com`) and the data path of research R9; credit log before/after (#4)
- [ ] T043 [US2] Vertex runs: CPU, 1xA100, 2xA100 inference; refusals kept as evidence; F26, F27 (#4)
- [ ] T044 [US2] Optimization: `optimization/export_onnx.py`, ONNX FP32 and int8 vs PyTorch on CPU, `quality_agreement`; `optimization/comparison.ipynb`, F28 (#5)
- [ ] T045 [US2] `distributed-training/train_ddp.py` + runs x1/x2/x4/x8 on CPU, 1 GPU, 2 GPUs; `technical-analysis.md` with decision and flip point; F29, F30, F46 (#6)
- [ ] T046 [US2] Live load test: `loadtest/locustfile.py` (one user = one ride at real speed, speed factor), `loadtest/shape.py`, `loadtest/plot_capacity.py`; run on kind (needs T030, T037); F31, F32 (#7)
- [ ] T047 [US2] `cost-resilience/infrastructure-plan.md` + `capacity.ipynb`: instances and cost for ≥ 3 volumes, rollback timed; proposed architecture F15; F33-F35 (#7, #8)

**Checkpoint B (24/11)**: SC-006, SC-008, SC-009 met.

---

## Phase 5: User Story 3 - Live replay (P3, bonus)

- [ ] T048 [US3] Streamlit replay page in `webapp/app.py` (play, speed, cumulative probabilities, phases, charter colours); recording F37 (#25)

## Phase 6: User Story 4 - Fall warning (P4, bonus)

- [ ] T049 [US4] Fall forecast in `notebooks/07_fall_forecast.ipynb` with groups never split (data-model.md `Ride.fall_group_id`), held-out groups, horizon chosen by measurement; F38 (#26)

---

## Phase 7: Submission and cross-cutting

- [ ] T050 README: choices, results, screenshots, Docker Hub URLs, stated gaps, compliance table of both briefs (#24)
- [ ] T051 [P] Align `docs/PLAN.md` DEC-1/2/4, `docs/how_to_use.md` (new endpoints, make targets) with PROJECT_FRAMING (#24)
- [ ] T052 Story outline per course, demo fallbacks in `assets/demos/`, timed dry runs (#24)
- [ ] T053 Submission email before 25/11 20:00 (#24)
- [ ] T054 [P] Bonus D3 automatic retraining via `repository_dispatch` (#27); experiment D4 falls as weak supervision `notebooks/08_falls_weak_supervision.ipynb`, only if F49 shows a signal (#28) — only after checkpoints A and B

---

## Dependencies

- Phase 1 → Phase 2. Inside Phase 2: T005-T008 (charter) before any figure task; T055-T058 (danger detector) before T014; T012 committed **before** T016 runs; T018 fixes the window length used by T029, T030, T031, T046; T019-T021 gate T038 and the SRV3490 scenarios of T034; T022-T024 before T041-T045.
- US1 and US2 run in parallel after Phase 2; T046 needs T030 and T037.
- US3, US4, T054 only after checkpoints A and B.

## Parallel examples

- Phase 2 start: T005 (palette), T009 (timeline), T012 (pre-registration), T019 (SRV3490), T022-T023 (bench) — five people, five files.
- US1 after T018: T028, T031, T032 in parallel; US2: T040, T041, T042 in parallel.

## Implementation strategy

1. Phase 1-2 (P0) first: they remove the open choices (palette, window length) and the external blocker (SRV3490).
2. MVP = US1 on kind with the riskier-mix scenario (GBG500 only), then SRV3490 scenarios.
3. US2 in parallel; each measurement lands in its CSV and figure as it is made.
4. Submission tasks from 24/11; bonus only if both checkpoints passed.
