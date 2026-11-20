# Research and decisions: 001 final deliverable

Each entry: **Decision**, **Rationale**, **Alternatives considered**. Sources: the repository as of commit
5dc1ab0, `docs/PROJECT_FRAMING.md`, the course labs (`lectures/mlops-labs/06-monitoring`, `05-kubernetes`,
`retraining-location-notes.md`) and the AI infrastructure brief.

## R1 - Cumulative live classifier, version 1

- **Decision** (window length provisional until R15): v1 = the running mean of the per-window softmax over the
  windows seen so far, on the window length chosen by R15 (E3 if 30 s wins). Add
  `CNNModel.predict_timeline(readings) -> Timeline` next to `predict_proba` (the model contract is extended, not
  changed). The last cumulative value equals today's `predict_proba` (E3 already averages its windows), so the
  whole-ride label and the gate are unchanged. A model trained on ride beginnings (random prefixes) is a v2,
  built only if the F1-vs-time curve shows early answers are weak **and** time remains.
- **Rationale**: zero retraining, zero risk to the graded gate, and it produces the live story (F07, F08)
  immediately. The pure function (`cumulative_mean(window_probas)`) is trivially testable.
- **Alternatives**: a sequence model (GRU) over window embeddings: more to explain, needs retraining and a new
  gate pass; exponential moving average: forgets the start of the ride, which "Safe overall" needs.

## R15 - Window length: an open choice until tested (blocks R1, R4, R8)

- **Status**: **not admitted.** The data has one reading every 100 ms; 30 s (stride 15 s) is E3's modelling choice,
  never compared with other lengths (`docs/PLAN.md` §6.5 "Not tried"). Everything derived from it stays provisional
  until this test is done: minimum ride length (300 readings), window payload of `/v1/predict/window`, update rate
  of the live label, load-test request rate, F1-vs-time points.
- **Test (hypothesis and rules written before running)**:
  - Hypothesis: a shorter window gives a reliable cumulative label earlier with little loss on the whole-ride
    label; below some length a window carries too little context and the whole-ride score drops.
  - Candidates: 5, 10, 15, 30, 60 s; stride = half the window; same architecture, epochs, augmentation, class
    weights and seed as E3; one change only (window length). If the shortest windows leave too few time steps after
    pooling, the kernel stays 0.7 s and the number of pooling steps is the only adaptation, written down.
  - Data: the ~400 training rides, the same stratified 5-fold ride-level CV as E3. **Golden rides not used** to
    choose.
  - Measured per candidate (mean ± std over folds): whole-ride macro-F1 (OOF); cumulative macro-F1 at 30, 60, 90,
    120, 180, 240 s and end; time to first answer (= window length); per-window inference latency (laptop CPU,
    `bench.py`); training time.
  - Decision rule: keep the candidates whose whole-ride OOF macro-F1 is within one fold-std of the best; among
    them pick the one with the highest cumulative macro-F1 at 60 s; tie → the shorter window. All candidates logged
    in MLflow (`experiment_id` E4-window-*), results in `artifacts/window_study.csv`, figure F47.
  - Then only: the chosen length goes through `evaluate` (golden scored once, gate). If it is not 30 s, it is
    registered as a new version and replaces E3 after the gate; E3 stays the fallback.
- **Rationale**: Vincent's rule (20/11): a parameter that is neither a standard default nor the finest granularity
  of the data must be justified by a serious test before the constraints that follow from it are accepted.
- **Alternatives**: keep 30 s without test (rejected: unjustified); per-reading model (rejected a priori would also
  be unjustified: 5 s is the shortest candidate because a 1D CNN with kernel 0.7 s and pooling needs at least a few
  seconds of signal; the test shows whether even 5 s is too short).

## R2 - SRV3490 rotation units

- **Decision**: verify the factor before converting: compare per-ride std of `rx, ry, rz` between datasets
  (median ratio, histogram in F04) and check it against 180/π = 57.30. Convert SRV3490 rotation in
  `data/srv3490.py` (factor in `params.yaml`, `data.srv3490.rotation_factor`), never in the model.
- **Rationale**: Principle VI; the same unconverted data is reused as the "sensor unit bug" drift scenario.
- **Alternatives**: per-dataset standardisation: hides the unit problem the monitoring story needs.

## R3 - Getting SRV3490 (external blocker)

- **Decision**: ask the course platform/teammates first thing; track the file with DVC on the SeaweedFS remote
  like GBG500. Until it arrives, every SRV3490-dependent item (A9, two drift scenarios, x1-x8 training cost
  with 3,990 rides) runs on GBG500 only and is labelled so.
- **Rationale**: keeps Phases A and B unblocked.
- **Alternatives**: none acceptable (no synthetic data).

## R4 - Live window API: stateless

- **Decision**: `POST /v1/predict/window` receives one 30 s window (300 readings) plus the client's running state
  (`sum_proba`, `n_windows`) and returns the window probabilities and the updated cumulative probabilities. The
  server keeps no per-ride state. `POST /v1/predict` (whole ride) adds the full timeline to the existing answer.
- **Rationale**: any replica can answer any window, so a load balancer and horizontal scaling work without
  sticky sessions; this is the property the AI infra scaling plan relies on. The state is 6 numbers.
- **Alternatives**: server-side sessions keyed by ride id (sticky routing, memory per ride, lost on restart);
  WebSocket streaming (harder to load-test and to show in Swagger).

## R5 - Monitored input features and live PSI

- **Decision**: per window, compute a small feature set (`features/window_stats.py`): std of each of the 6
  channels and the norm of acceleration and rotation (8 features). A DVC stage `reference` writes their deciles on
  training windows (`artifacts/drift_reference.json`). The API creates one Prometheus histogram per feature with
  **the reference deciles as bucket edges**, so the reference share is 10 % per bucket by construction. The live
  exporter `drift/monitor.py` queries Prometheus (`increase(...[5m])` per bucket), computes PSI per feature and
  chi² on the predicted-category mix against the training mix, and exposes `drift_psi{feature}`,
  `drift_prediction_chi2_pvalue`, `drift_retrain_recommended` on its own `/metrics`. The offline job uses the same
  `psi()` on logged windows and adds KS per feature.
- **Rationale**: PSI moves live in Grafana (FR-005) without shared files between replicas; one `psi()` serves
  both paths. The unit bug multiplies rotation std by ~57: it lands entirely in the top bucket, PSI explodes,
  a clear demo.
- **Alternatives**: Pushgateway (one more component), PromQL-only PSI (unreadable), Evidently live (not a
  Prometheus source). Evidently stays as the optional HTML report of the offline job.

## R6 - Outcome layer (labels received later)

- **Decision**: `POST /v1/feedback {ride_id, predicted_label, true_label}` increments
  `feedback_total{true_label, correct}`. The replay client sends it for GBG500 rides after a delay. Alert on
  rolling accuracy over the last 15 min (enough samples required). Macro-F1 on feedback is computed by the
  offline drift job, not in PromQL.
- **Rationale**: stateless again (no prediction store to join on). Known simplification, written in the
  runbook: in production the join would be done on the prediction log by ride id.
- **Alternatives**: server-side cache of predictions (wrong across replicas), join in the drift job only (no
  live alert).

## R7 - Drift scenarios from real rides

- **Decision** (`drift/scenarios.py`, seeded):
  1. **normal**: training-split GBG500 rides resampled (thresholds are calibrated on this, never on golden);
  2. **riskier mix**: GBG500 rides resampled with Reckless, Tandem and Bad weather oversampled (labels known, so
     the outcome layer moves too) - available now;
  3. **unseen city**: SRV3490 rides from one city, units harmonised;
  4. **sensor unit bug**: the same SRV3490 rides sent **without** the conversion (a real upstream bug).
  3 vs 4 isolates the unit effect from the city effect. Runbook: 3 → retrain (with pseudo-labels, stated as such);
  4 → upstream fix, no retraining.
- **Rationale**: no synthetic data; each scenario shows a different trigger (input drift, prediction drift,
  performance).
- **Alternatives**: scaling GBG500 rotation by 1/57.3 (a transform, weaker story than the real file).

## R8 - Live load test

- **Decision**: Locust `RideUser`: on start picks a real ride and cuts its windows; every `stride_s / speed`
  seconds posts the next window to `/v1/predict/window` carrying its state; at the end of the ride starts
  another. A `LoadTestShape` steps users (e.g. +10 every 60 s). "One CPU instance" = one kind replica with a fixed
  CPU limit (1 vCPU, set in `params.yaml` `loadtest`). Result: max users before p95 > 200 ms, converted to live
  rides at real speed (`speed` factor reported). `plot_capacity.py` from lab 05 adapted for F31.
- **Rationale**: matches "one Locust user = one ride streaming at real speed"; the speed factor reaches
  saturation without thousands of Locust users on the laptop, and the conversion is exact for a stateless API.
- **Alternatives**: whole-ride requests (not the live use case).

## R9 - Where cloud jobs get code and data

- **Decision**: Vertex AI custom jobs in `advanced-cloud-ai-i2`, `europe-west4`, SPOT, prebuilt PyTorch GPU image,
  code sent as a base64 tarball in the command, results printed between `@@CSV` / `@@END` and parsed from the
  logs (pattern of the successful 12/11 job). Data: the prepared window tensors (a few hundred MB at most) are put
  in a private bucket in `agentics-workflows-2` with read granted to the Vertex service agent of
  `advanced-cloud-ai-i2`; if the cross-project grant fails, fall back to a private Artifact Registry image with the
  data baked in. `ai-infrastructure/vertex/submit.py` wraps submission with `--account=vlamy@albertschool.com`.
- **Rationale**: we cannot write buckets in the GPU project; both fallbacks keep data private.
- **Alternatives**: Colab/Kaggle (used as fallback for 1 GPU / 2x T4 if Vertex refuses, labelled).

## R10 - Python version in cloud jobs

- **Decision**: `bench.py`, `train_ddp.py` and the modules they import (`models/cnn.py`, `features/windows.py`)
  avoid 3.11+ only syntax beyond what 3.10 supports, checked by running `bench.py` once in the Vertex image before
  the real runs.
- **Rationale**: the prebuilt image is Python 3.10; rebuilding a custom GPU image costs time and disk.
- **Alternatives**: custom image with 3.14 (heavy on the laptop).

## R11 - Inference hardware and optimization

- **Decision**: same `bench.py` everywhere; targets: laptop CPU, Vertex CPU (n1-standard-8), 1 GPU
  (a2-highgpu-1g) and 2 GPUs (a2-highgpu-2g), A100 on both so only the GPU count changes; batch sizes 1, 8, 32,
  128, 512 windows; warm-up 20, 100 repetitions, mean and p95, `cuda.synchronize`. Optimization: ONNX Runtime CPU
  export, then dynamic int8 quantization, compared on latency, throughput, memory and agreement with PyTorch
  predictions plus golden macro-F1 (scored once, read-only).
- **Rationale**: the model is tiny (45 k parameters): the expected finding is that a GPU does not pay off for
  serving, which is a valid, measured conclusion.
- **Alternatives**: T4 (regional quota 1, cannot do 2); TensorRT (out of course scope).

## R12 - Publishing images without publishing the model

- **Decision**: Docker Hub images (required by the brief) contain code only. The API loads the model at startup
  from `MODEL_URI` (MLflow registry `models:/ride-classifier@production`, artifacts on the cluster's SeaweedFS).
  Existing tags with a baked-in model (`-m2`) are made private on Docker Hub or deleted - team decision, to
  confirm with Alex who owns the account.
- **Rationale**: constitution VII (no model trained on course data published) with the brief's Docker Hub
  requirement.
- **Alternatives**: private Docker Hub repositories only (free plan limits to one private repository).

## R13 - Retraining policy and where it runs

- **Decision**: hybrid triggers, all landing in one `train.yml` with an input `reason`
  (`schedule | input_drift | prediction_drift | performance | manual`): weekly cron; `drift.yml` runs the drift
  job (scheduled and on demand), writes the report to the job summary and, on exit 3, raises a warning flag and
  (core) tells the operator to dispatch `train.yml` with the reason; (bonus D3) dispatches it itself. `train.yml`
  runs on the self-hosted runner (data and MLflow are local), registers `@staging` only if the gate passes;
  `cd.yml` deploys only after approval on the `production` environment. Where to retrain: default CI
  self-hosted runner (training measured in minutes on CPU); Vertex AI recommended above the data volume where
  B5 shows the CPU time exceeding the 30 min target (SC-011).
- **Rationale**: matches the lab's retraining-location table and the measured cost; human approval keeps the
  hybrid policy honest.
- **Alternatives**: retrain on the inference machine (competes with serving, shown in the table only).

## R14 - Visual charter implementation

- **Decision**: `viz/palette.py` (pure constants: Okabe-Ito colour per category, Reckless = vermillion
  `#D55E00`, our model = blue `#0072B2`, references = greys), `viz/theme.py` (matplotlib rcParams, white
  background, Mako from seaborn, Heat = matplotlib `YlOrRd`), `save_fig(fig, fig_id, slug)` writing
  `assets/figures/{fig_id}_{slug}.png` (200 dpi) and `.svg`. The web app and Grafana use the same hex values.
  F36 decides Mako vs `viridis` before freezing.
- **Rationale**: one place for colours; decks built from files.
- **Alternatives**: plotly (interactive, but static PNG/SVG export needs kaleido and the slides are static).
