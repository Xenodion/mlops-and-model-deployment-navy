# Feature Specification: Final deliverable (two presentations on one shared base)

**Feature Branch**: none (work stays on `dev_vincent`; spec folder `specs/001-final-deliverable`)

**Created**: 2026-11-20

**Status**: Draft

**Input**: User description: "The final deliverable of the project, planned top-down from the two oral
presentations: the story each presentation tells, the catalogue of figures and demos, the features that produce
them (core or bonus), and a visual charter." Decisions it builds on: `docs/PROJECT_FRAMING.md`; rules:
`.specify/memory/constitution.md` v1.4.0. Final framing (21/11 evening): two distinct models, ride type
(cumulative) and fall risk (10 s sliding window), served together: see PROJECT_FRAMING §3 and ADR 001.

## Overview

The project is judged twice, on one shared system:

| | MLOps jury | AI infrastructure defence |
|---|---|---|
| When | 26/11/2026 (deliverable by email 25/11 20:00) | to confirm (see Assumptions) |
| Format | 10 min presentation + 10 min questions, group of 5 | presentation of the `ai-infrastructure/` contribution |
| Graded on | code quality, CI/CD automation, MLflow tracking and management, API usability (Swagger), consistency of the overall architecture, GitHub collaboration; model better than random | resource sizing, baseline benchmark, one optimization, distributed-training decision, scaling / cost / resilience; every choice justified by measurements |

**Shared base (the one sentence both stories start from):** e-scooter rides stream their motion-sensor data to
our deployed service, which tells, live, what kind of ride it is so far (Safe, Safe overall, Bad weather, Tandem,
Reckless; at the end, the ride's category) and raises a fall-risk alert when the last seconds of riding look like
those that precede a fall.

## User Scenarios & Testing *(mandatory)*

The users are the two juries. Each story is what a jury must see and understand; each one is a slice that can be
rehearsed and shown on its own.

### User Story 1 - MLOps jury: from raw rides to a deployed live classifier (Priority: P1)

The MLOps jury follows one ride through the whole system in under 10 minutes: the data and the problem (two
datasets, why the ride label is read live, window by window), the versioned pipeline (prepare, train, evaluate
with a quality gate), the tracked experiments and the model registry, the CI/CD that builds, tests, pushes and
deploys to the cluster with a manual approval, the API answering a ride with its category and its timeline, the
web app, and the monitoring dashboards. It ends with "the model beats random; the system is what we built".

**Why this priority**: it is the graded deliverable with a fixed date (25/11 20:00, jury 26/11) and it uses
every mandatory tool of the brief.

**Independent Test**: a dry run of the presentation on the production cluster, timed, where every demo works
from a clean start and every screenshot exists in the assets folder.

**Acceptance Scenarios**:

1. **Given** the cluster is running, **When** a presenter sends one golden ride to the API from its interactive
   documentation, **Then** the answer shows the ride's category, its confidence and its timeline (per-window
   and cumulative), and the jury sees the request on the monitoring dashboard within a minute.
2. **Given** a code change is pushed, **When** the pipeline runs, **Then** the jury can follow on GitHub the
   tests, the image build and push, the approval step and the deployment, and see the new version answering.
3. **Given** the experiment tracker is open, **When** the presenter shows the two registered models, **Then** model
   1 (ride type) shows its golden macro-F1 vs random and the gate result, and model 2 (fall risk) its held-out
   pre-fall vs comparison score with the 5 s and 10 s gaps.
4. **Given** a drift scenario is replayed (new city, changed sensor, riskier mix), **When** the dashboard is
   shown, **Then** the drift indicator moves past its alert threshold for that scenario and stays calm for
   normal traffic.
5. **Given** an alert fires, **When** the presenter follows the retraining policy, **Then** the jury sees which
   trigger fired (schedule, input drift, prediction drift or performance drop), the runbook decision (for the
   changed-sensor scenario: an upstream bug, fixed at the source, no retraining; for the new city: retrain), the
   retraining run, the gate, and the human approval before the new version reaches production.

---

### User Story 2 - AI infrastructure jury: the right infrastructure, proven by measurements (Priority: P1)

The AI infrastructure jury sees the model's real needs (data volume, model size, memory, training time), the
baseline performance measured under fixed conditions, one optimization compared fairly, the decision on
distributed training with the threshold where it would change, and a serving architecture sized from a live
load test, with costs and a rollback plan. The thread: "how many live rides can we follow, on what hardware,
and when would we need a GPU or several?"

**Why this priority**: graded deliverable; its five parts and folder layout are imposed by the brief.

**Independent Test**: every number in the `ai-infrastructure/` summary is traceable to a saved raw result and a
re-runnable procedure, or is explicitly marked as an estimate with its assumptions.

**Acceptance Scenarios**:

1. **Given** the baseline results file, **When** the jury asks how a latency was obtained, **Then** the
   hardware, batch size, concurrency, repetitions and warm-up are written next to it.
2. **Given** the hardware comparison (CPU, 1 GPU, 2 GPUs) and the training-cost curve (data x1 to x8), **When**
   the presenter states the distributed-training decision, **Then** it is backed by the measured speedup and
   efficiency and names the data or model size at which the decision would flip.
3. **Given** the live load test, **When** the presenter states the number of live rides one CPU instance
   follows, **Then** the jury sees the point where p95 latency crosses the limit, and the proposed architecture
   and cost table derive from that number.

---

### User Story 3 - Both juries: a ride replayed live (Priority: P3, bonus)

A presenter picks a ride, presses play with a chosen speed, and the web app shows the cumulative category
probabilities and the per-window phases evolving until the final label.

**Why this priority**: the most memorable demo and the clearest picture of the live design, but not required by
either brief; built only once stories 1 and 2 work.

**Independent Test**: replaying any golden ride end to end at x10 speed without error, final label equal to the
model's whole-ride prediction.

**Acceptance Scenarios**:

1. **Given** the web app, **When** a ride is chosen and played at x10, **Then** the curves update smoothly and
   end on the ride's predicted category, with the true label shown for golden rides.

---

### User Story 4 - Both juries: an early warning before a fall (Priority: P4, bonus)

The presenter shows, on rides that ended in a fall, the estimated probability of a fall in the next few seconds
rising (or not) before the fall, compared with the same moment of comparison rides from the same city.

**Why this priority**: the real-world use case juries remember, but outside both briefs; built only if the core
is validated and time is left.

**Independent Test**: a held-out set of falls and their comparison rides, never used to build or tune the
forecast, gives a score and an alert trade-off that are reported as measured, even if close to chance.

**Acceptance Scenarios**:

1. **Given** a fall ride cut a minimum warning time before the fall, **When** the forecast is run, **Then** the
   probability and the alert decision are shown, with the comparison rides' values next to it.

---

### Edge Cases

- A ride shorter than one window: the cumulative label is still returned, marked low confidence.
- A ride sent with missing readings or wrong units: the API rejects malformed input with a clear error; unit
  drift on valid input is caught by monitoring, not silently absorbed.
- Golden rides must never be used to choose anything; if a choice needed them, the result is reported as
  optimistic.
- A GPU job is not granted (quota or SPOT capacity): the result is reported as blocked with the refusal message
  kept as evidence; the presentation uses the measurements that exist.
- The cumulative classifier fails the quality gate by the deadline: the reference model stays in production and
  the comparison becomes the result.
- A demo fails live: every demo has a recording or screenshots in the assets folder.

## Requirements *(mandatory)*

### Functional Requirements

**Presentation and assets**

- **FR-001**: The deliverable MUST include, per course, a story outline (sections, timing, demos, figures) that
  maps every graded criterion of that course's brief to at least one slide, demo or figure.
- **FR-002**: Every figure and demo listed in the Figure Catalogue MUST state the story it serves, core or
  bonus, and the notebook or procedure that produces it; core items MUST exist before the deadline.
- **FR-003**: All figures MUST follow the Visual Charter and be saved, as they are produced, as PNG and SVG into
  one assets folder, so that the decks are assembled from files, not redrawn.
- **FR-004**: Every demo MUST have a fallback (recording or screenshots) in the assets folder.
- **FR-005**: The drift and detection story MUST be shown in action: each drift scenario is replayed against the
  deployed service while the dashboards and the alert list are recorded, so the slide shows the drift metrics
  moving, the alerts firing and the runbook decision, from a real run.

**MLOps submission (brief "Format")**

- **FR-006**: The repository MUST contain all the code (preprocessing, training, operational API, interactive web
  app for testing the model, monitoring), the pipeline DAG, the CI/CD workflows, all tests (unit, integration and
  end-to-end, each kind present and run in CI where possible), the documentation, and a README explaining the
  technical choices and the results with screenshots of the production environment and the Docker Hub image URLs.
- **FR-007**: The submission email (subject `[ALBERT][MLOPS]`, to both teachers) MUST contain the repository link,
  the Docker Hub URLs, the list of group members, and, if deployed remotely, the URLs and access to them; it is sent
  before 25/11/2026 20:00.
- **FR-008**: Gaps with the brief MUST be stated in the README with the reason: the API receives a ride's sensor
  recording, not an image (the data has no images); GCP instead of AWS; feature store (bonus) not built unless time
  allows, with what it would bring.

**AI infrastructure submission (brief "Expected contribution" and checkpoint)**

- **FR-009**: The AI infrastructure deliverable MUST reproduce the brief's templates as they are: the inventory of
  the current system (model, framework, implemented components, where it runs) with the current-system diagram, the
  baseline results table, the optimization comparison table, the distributed-training rationale, the proposed
  architecture (API, model runtime, scaling, monitoring, failure recovery), costs with explicit assumptions, the
  rollback procedure, and named responsibilities with next actions; files at the paths of the suggested layout.

**Core system (MLOps)**

- **FR-010**: Model 1 (ride type) MUST read a ride window by window (30 s, ADR 002) and give, at every window, the
  ride's category so far, cumulated since the start of the ride; at the end, the whole-ride category. It is E3.
- **FR-011**: Model 1 MUST be evaluated on the frozen golden rides (quality gate: better than random and above the
  floor), with Reckless and the counterfactual categories reported explicitly, and on macro-F1 as a function of
  elapsed ride time.
- **FR-011b**: Model 2 (fall risk) MUST score short sliding windows (10 s, ADR 002), not cumulated, trained on
  SRV3490 pre-fall windows vs the same moments of comparison rides, never seeing the last 5 s before a fall
  (anti-leakage gap, 10 s as sanity check, ADR 004), and MUST be evaluated on held-out fall groups.
- **FR-012**: The second dataset (3,490 rides) MUST enter the versioned pipeline with its rotation units
  harmonised with the first after the conversion factor is verified on the data; it is the training and
  evaluation data of model 2.
- **FR-013**: One API image MUST serve both models (ADR 010): for a ride, model 1's category, confidence and
  timeline, and model 2's fall risk per window (marked retrospective on a recorded ride); for a live window, model
  1's cumulative category and model 2's current fall risk.
- **FR-014**: Training MUST run through the pipeline, register a version only when the gate passes, and deploy
  through CI/CD (build, test, push to the image registry, manual approval, rollout, automatic rollback on a
  failed smoke test) on the local cluster, with a self-hosted runner.
- **FR-015**: Monitoring MUST cover the four layers taught in the course, each with a dashboard row, an alert
  and a runbook entry: infrastructure (latency, errors, throughput), input distribution (sensor channels),
  prediction distribution (category mix, confidence), business outcome (performance on labels received later).
- **FR-016**: Three drift scenarios MUST be produced by resampling real rides (a city never seen, the real
  sensor unit difference, a riskier ride mix) and each MUST be visible on the drift dashboard.
- **FR-018**: Drift MUST be measured by a job comparing the inputs logged by the service with the training
  reference (per-feature PSI with an alert and a retraining threshold, a statistical test per feature, a
  readable drift report), runnable on demand, on a schedule and in CI/CD, where it raises a visible flag.
- **FR-019**: Labels received after the prediction MUST be accepted by the service (outcome layer). They come
  from real labelled rides replayed after their prediction, and feed a performance metric with an alert.
- **FR-034**: Retraining MUST follow a hybrid policy: a schedule, an input-drift trigger (PSI above the
  retraining threshold), a prediction-drift trigger and a performance trigger (outcome metric below its floor).
  A trigger starts the training workflow with its reason; the new version is registered only if the gate passes
  and reaches production only after a human approval (human in the loop). The runbook tells retraining apart
  from an upstream bug (unit change: fixed at the source, no retraining).
- **FR-035**: Where retraining runs MUST be justified against the options taught (local, CI/CD runner,
  inference machine, dedicated cloud such as Vertex AI) using the measured training cost; the recommendation is
  part of both presentations.
- **FR-036**: If any part of the retraining loop cannot be fully automated before the deadline, the measurements
  and the CI/CD flags MUST still exist, and the presentation MUST state the recommended policy and what is
  missing.
- **FR-017**: The reference supervised model (E3) MUST stay registered and serve as production fallback until the
  stays model 1; model 2 is registered separately under its own name and gate.

**Core measurements (AI infrastructure)**

- **FR-020**: The `ai-infrastructure/` folder MUST follow the imposed layout and contain the five parts, each with
  a before/after view.
- **FR-021**: One benchmark procedure MUST be reused on every hardware target, recording environment, batch size,
  concurrency, repetitions, warm-up, mean and p95 latency, throughput, memory and a quality metric, into raw
  result files.
- **FR-022**: Inference MUST be measured on CPU, 1 GPU and 2 GPUs on the course's cloud project, with jobs that
  stop by themselves.
- **FR-023**: Training cost MUST be measured against data volume (all real rides, replicated x2, x4, x8) on CPU,
  1 GPU and 2 GPUs (data-parallel), reporting time, memory, speedup and parallel efficiency; replication MUST be
  labelled as a cost measurement, not new information.
- **FR-024**: One optimization (ONNX Runtime and/or quantization, or an inference setting) MUST be compared with
  the baseline on the same workload and hardware, including prediction quality.
- **FR-025**: A live load test MUST simulate concurrent rides, each streaming its windows at real speed, and find
  how many live rides one CPU instance follows before p95 latency exceeds 200 ms; dashboards MUST show it live.
- **FR-026**: The scaling, cost and resilience plan MUST derive instance counts and costs from the measured
  capacity for stated request volumes, and describe a rollback tested on the local cluster.

**Bonus**

- **FR-030** (bonus): Live replay of a ride in the web app (play, speed, cumulative probabilities, phases).
- **FR-031** (bonus): Fall forecast (probability of a fall in the next X seconds, X chosen by measurement among
  5 / 10 / 30 s, with a minimum warning time of 1-2 s, a fall and its comparison rides always on the same side of
  the split, held-out evaluation).
- **FR-032** (bonus): a richer model 2 (model 1's output as a feature, lags, longer context).
- **FR-033** (bonus): Fully automatic retraining on a drift alert, end to end without a manual start (the core
  loop of FR-034 may start the workflow by hand from the alert).

**Constraints**

- **FR-040**: Cloud resources MUST be GCP (the school's provider; the only allowed deviation from the MLOps
  brief's AWS wording); the local stack keeps S3-compatible storage. GPUs are used only in the course project
  `advanced-cloud-ai-i2`.
- **FR-041**: Course data and models trained on it MUST NOT be published; images holding them go to a private
  registry, files to a private bucket.
- **FR-042**: No synthetic data; every load, scaling or drift input is real data replayed, resampled or
  replicated, and says which.

### Figure Catalogue

Exhaustive on purpose; the decks pick from it. Notebook names are proposals for the plan.

| ID | What it shows | Story | Core / bonus | Source |
|---|---|---|---|---|
| F01 | The two datasets side by side (rides, readings, cities, labels) | MLOps, infra | core | `01_data_overview` |
| F02 | Category counts (500 rides) and imbalance | MLOps | core | `01_data_overview` |
| F03 | One ride per category: the 6 sensor channels over time | MLOps | core | `01_data_overview` |
| F04 | Rotation units: GBG500 vs SRV3490 distributions before and after conversion | MLOps | core | `02_units_and_cities` |
| F05 | Rides and falls per city (SRV3490) | MLOps | core | `02_units_and_cities` |
| F06 | A ride cut into 30 s windows (diagram) | both | core | `03_live_classifier` |
| F07 | Per-window scores vs cumulative label along one ride | both | core | `03_live_classifier` |
| F08 | Macro-F1 vs elapsed ride time (cumulative classifier vs reference vs random) | MLOps | core | `03_live_classifier` |
| F09 | Golden confusion matrix of the whole-ride label | MLOps | core | `04_evaluation` |
| F10 | Gate: golden macro-F1 of candidates vs floor, majority and random references | MLOps | core | `04_evaluation` |
| F11 | Model 1 on golden rides: Reckless and counterfactual categories, vs majority and random | MLOps | core | `04_evaluation` |
| F12 | Model 1 category mix and model 2 alert rate on SRV3490 by city vs GBG500 | MLOps | core | `05_srv3490_by_city` |
| F13 | Model 1 confidence on GBG500 vs SRV3490 | MLOps | core | `05_srv3490_by_city` |
| F14 | Current system architecture diagram | both | core | `ai-infrastructure/architecture` |
| F15 | Proposed (target) architecture diagram | infra | core | `ai-infrastructure/architecture` |
| F16 | Pipeline graph (prepare, train, evaluate) | MLOps | core | pipeline tool output |
| F17 | Experiment tracker: runs compared, registry versions and aliases (screenshots) | MLOps | core | screenshots |
| F18 | CI/CD run: jobs, approval, deployment (screenshots) | MLOps | core | screenshots |
| F19 | API documentation page and one answer with timeline (screenshots) | MLOps | core | screenshots |
| F20 | Monitoring dashboards: service metrics during normal traffic (screenshot) | MLOps | core | screenshots |
| F21 | Drift dashboard during each of the three scenarios (screenshots) | MLOps | core | `06_drift_scenarios` + screenshots |
| F22 | Drift indicator per feature and scenario vs alert thresholds | MLOps | core | `06_drift_scenarios` |
| F23 | Resource sizing table: data, model size, memory, training time | infra | core | `ai-infrastructure/benchmarks` |
| F24 | Theoretical training memory vs available memory (log scale) | infra | core | `ai-infrastructure/benchmarks` |
| F25 | Latency (mean, p95) by stage: cleaning, windows, network | infra | core | `ai-infrastructure/benchmarks` |
| F26 | Latency and throughput vs batch size, CPU vs 1 GPU vs 2 GPUs | infra | core | `ai-infrastructure/benchmarks` |
| F27 | Memory (RAM / VRAM) by hardware | infra | core | `ai-infrastructure/benchmarks` |
| F28 | Optimization before/after: latency, throughput, memory, quality | infra | core | `ai-infrastructure/optimization` |
| F29 | Training time and memory vs data volume (x1-x8), CPU / 1 GPU / 2 GPUs | infra | core | `ai-infrastructure/distributed-training` |
| F30 | Data-parallel speedup and efficiency; small vs large batches (communication-bound case) | infra | core | `ai-infrastructure/distributed-training` |
| F31 | Live load test: p95 latency vs number of live rides, limit line, capacity per instance | both | core | `ai-infrastructure/cost-resilience` |
| F32 | Load test seen live on the dashboard (screenshot or recording) | both | core | screenshots |
| F33 | Instances and monthly cost vs request volume, CPU vs GPU | infra | core | `ai-infrastructure/cost-resilience` |
| F34 | Rollback timeline measured on the cluster | both | core | `ai-infrastructure/cost-resilience` |
| F35 | Decision table: recommended infrastructure by volume, model size and latency target | infra | core | `ai-infrastructure/README` |
| F36 | Palette test (categories, heatmap, risk scales) | none (charter) | core | `00_visual_charter` |
| F37 | Live replay of a ride (recording) | both | bonus | web app |
| F38 | Fall forecast: probability before the fall vs comparison rides, alert trade-off | both | bonus | `09_fall_forecast` |
| F39 | Richer model 2 vs minimal model 2 (bonus) | MLOps | bonus | `08_fall_risk_features` |
| F40 | Drift alert triggering retraining end to end (screenshot of the run) | MLOps | bonus | screenshots |
| F41 | Four-layer monitoring dashboard: baseline and each incident (screenshots) | MLOps | core | screenshots |
| F42 | Alerts going inactive → pending → firing during an incident (screenshot) | MLOps | core | screenshots |
| F43 | Retraining policy diagram: triggers → workflow → gate → human approval → production, and the runbook fork (retrain vs fix upstream) | both | core | `docs/` diagram |
| F44 | Outcome layer: performance on labels received later, before and after a drift scenario | MLOps | core | `06_drift_scenarios` |
| F45 | Drift report (per-feature PSI and test) for each scenario | MLOps | core | drift job output |
| F46 | Where to retrain: options vs measured training cost (local, CI runner, inference machine, Vertex AI) | both | core | `ai-infrastructure/distributed-training` |
| F48 | Cumulative (model 1) vs sliding-window (model 2) scan of one ride: the pedagogical diagram | both | core | `03_live_classifier` |
| F49 | Model 2 fall risk before a fall vs the same moment of comparison rides, 5 s and 10 s gaps (held-out groups) | both | core | `07_fall_risk` |
| F47 | Window comparison behind ADR 002: pre-fall vs comparison AUC per window length (5-60 s) | both | core | `03_live_classifier` |

### Visual Charter

- **Background and mode**: white background, light mode for every figure, web app page and dashboard screenshot.
- **Categories**: Okabe-Ito colour-blind-safe palette, one fixed colour per category used everywhere (notebooks,
  web app, dashboards when possible, slides); Reckless gets the strongest warm colour so alerts stand out.
- **Model vs references**: one fixed colour for "our model", neutral greys for the references (majority, random,
  reference model).
- **Heatmaps** (confusion matrices, probability timelines): the Mako sequential scale.
- **Risk** (fall forecast, drift intensity): the Heat scale, yellow to red.
- **Validation**: the palette test figure (F36) is reviewed before the charter is frozen.
- **Production**: one shared theme definition and one save step that writes PNG and SVG into the single assets
  folder; figures are produced in notebooks that also show the exploration that led to them.

### Key Entities

- **Ride**: one sensor recording (about 3-8 min, 10 readings per second, 6 channels), from GBG500 (with a
  category) or SRV3490 (with city, fall moment if any, link to its comparison rides).
- **Window**: a 30 s slice of a ride; carries a per-window score (a hint, never a validated label).
- **Timeline**: for one ride, the sequence of per-window scores and cumulative probabilities; its last cumulative
  value is the ride's predicted category.
- **Golden ride**: one of the 100 frozen GBG500 rides used only for the final evaluation of model 1 (it holds 1
  Reckless ride, stated wherever Reckless results are shown).
- **Fall risk**: model 2's value for one 10 s window, not cumulated; an alert when it passes a threshold chosen on
  development fall groups.
- **Fall group**: a SRV3490 ride with a fall and its 4 comparison rides from the same city; never split.
- **Measurement**: one benchmark result with its conditions (hardware, batch size, concurrency, repetitions,
  warm-up, versions) and its status (measured or estimated).
- **Drift scenario**: a resampled stream of real rides (unseen city, sensor unit change, riskier mix).
- **Figure / asset**: a saved PNG and SVG with its catalogue ID, story and source.
- **Retraining trigger**: what started a training run (schedule, input drift, prediction drift, performance drop), with its value and time; linked to the run, the gate result and the approver.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Each presentation fits its time slot (10 min for MLOps) in a timed dry run, and every graded
  criterion of each brief is covered by at least one slide, demo or figure.
- **SC-002**: 100 % of core catalogue items (F01-F36, F41-F49) exist in the assets folder before 25/11 20:00, with a
  fallback for each demo.
- **SC-003**: Model 1 beats random on the golden rides and passes the gate, with macro-F1 vs elapsed time for at
  least 6 time points; model 2's held-out pre-fall score is reported with the 5 s gap and the 10 s check.
- **SC-004**: A ride sent to the deployed service gets its category and timeline in under 1 second for a whole
  ride, and each live window is answered in under 200 ms at p95 up to the measured capacity.
- **SC-005**: The capacity of one CPU instance in live rides is measured, and the proposed architecture's instance
  counts and costs are derived from it for at least 3 request volumes.
- **SC-006**: 100 % of numbers in `ai-infrastructure/` are labelled measured or estimated, and every measured one
  points to a raw result file and a re-runnable procedure.
- **SC-007**: Each of the three drift scenarios moves its drift indicator past the alert threshold on the
  dashboard, while normal traffic stays below it.
- **SC-010**: Each of the four monitoring layers has an alert that is seen firing in at least one rehearsed
  incident, with a runbook entry giving the decision.
- **SC-011**: From a retraining trigger to a new version waiting for approval takes under 30 minutes, and every
  production change in the period is traceable to a trigger, a gate result and an approver.
- **SC-008**: Distributed training is decided with a measured speedup and efficiency on 2 GPUs (or the documented
  refusal if GPUs were not granted) and the volume at which the decision flips.
- **SC-009**: No billable cloud resource is left running after a measurement (checked after each session), and
  the remaining GCP credits are recorded before and after each cloud session.
- **SC-012**: Every line of both briefs (MLOps objectives, format and bonus; AI infrastructure steps 1-5 and
  checkpoint) is ticked in a compliance table in the README, pointing to where it is met, or marked as a stated gap.

## Assumptions

- The AI infrastructure defence date and format are not known yet; the deliverable is built for the same week as
  the MLOps jury, with the `ai-infrastructure/` folder complete by 25/11. To confirm with the teacher.
- Vertex AI GPU jobs in SPOT mode remain available in `advanced-cloud-ai-i2` (a 2x A100 job ran on 12/11);
  if a GPU type is refused, the refusal is kept as evidence and the teacher is emailed.
- Files that GPU jobs need or produce live in a private bucket of `agentics-workflows-2` (storage admin there),
  readable by the Vertex jobs' service account; otherwise results are printed to job logs.
- The second dataset (SRV3490) is obtained from the course platform; its rotation is very likely in rad/s (to be
  verified on the data before conversion).
- The current production model (E3, registered version 2) is the reference for all comparisons until replaced.
- The team of 5 shares the work; owners are set when tasks become issues.
- Bonus items (stories 3 and 4, FR-030 to FR-033) start only after stories 1 and 2 are validated.
