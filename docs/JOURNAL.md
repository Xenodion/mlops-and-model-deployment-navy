# Project journal

What was done, why, and where the evidence is, newest first. Decisions are detailed in
[decisions/](decisions/README.md); this log is the story behind them, written to be reread when preparing the
presentations. One entry per working session; mistakes found and corrected are logged too.

## Where we are (resume here) - updated 21/11 end of session

This block is rewritten at each stop; the dated entries below are the history.

- **Resume, notebook:** `notebooks/07_fall_risk.ipynb` now plots F49 from the existing held-out metrics; the 10 s
  check shifts the entire 60 s segment, so the difference cannot be attributed only to its last five seconds.
  `fall-risk` v1 was registered at `@staging` on the shared MLflow server on 22/11: run
  `a970ab9cb9994727a3ceb7c885916491`, clean commit `f8ba66e`, gate passed, held-out paired AUC 0.888 / 0.756
  (5 s / 10 s gap). The MLflow interface accepts window statistics; the API still loads its joblib artifact
  separately. Training ran directly; its outputs were recorded in `dvc.lock` with `dvc commit`.
- **Model 1 figures:** notebooks `03_live_classifier` and `04_evaluation` now produce F06-F11 and F47-F48 from
  existing outputs. The golden split contains only one Reckless ride; F11 shows the actual counts. F08 compares
  elapsed-time E3 results to whole-ride reference scores, not references recomputed at each time.
- **Drift reference (T031):** eight window standard deviations, PSI and the reference DVC stage are implemented
  and tested. DVC reproduced the reference from 7,568 training windows and OOF predictions. Repeated quantiles are
  collapsed and actual reference bin shares retained. Integration with the API and drift job remains; the model
  field currently identifies the MLflow model URI rather than a registry version.
- **API metrics:** live model-2 scores now increment `fall_risk` and `fall_alerts_total` Prometheus metrics;
  29 API tests pass. T063 remains open for removal of old danger fields and the image/model loading decision.
- **Offline drift job (T033, in progress):** PSI, prediction-mix and delayed-label checks produce JSON/Markdown
  reports and exit 0 / 3 / 1. The job accepts feature rows or grouped sensor windows. A tiny integration test
  detects the skipped rotation conversion, while the same converted windows stay normal. KS, live monitor,
  production log/scenario inputs and normal-scenario threshold calibration remain open; the prediction reference is
  per ride, so current prediction sampling must be comparable.

- **Design (final):** two distinct models served by one API (ADR 001, ADR 010). Model 1 = ride type, E3 read
  cumulatively. Model 2 = fall risk on 10 s windows (ADR 002, ADR 004).
- **Done:**
  - Shared MLflow (Cloud Run `ride-mlflow`) + shared DVC remote (`gs://albert-navy-ride-mlops/dvc`), ADR 011. 11 runs
    migrated, data and pipeline outputs pushed (`make shared-data-push`). #30 can be closed.
  - Model 2 (T060, T061): `models/fall_risk.py`, DVC stage `train_fall_risk`, logistic regression on window
    statistics. Held-out paired AUC 0.89 with the 5 s gap, 0.76 with the 10 s check (window AUC 0.71 / 0.69, ROC curves for both gaps in `artifacts/fall_risk_roc.png`, F49): gate
    passed; the drop at 10 s means part of the signal is very close to the fall; say so when presenting.
  - API (T063, mostly): `fall_risk` on `/v1/predict` (retrospective) and `/v1/predict/window`, `/health` fields;
    model 2 optional (`FALL_RISK_MODEL_PATH`, default `artifacts/fall_risk_model.joblib`), threshold 0.6 (contract, not
    tuned). 148 tests pass.
  - Model 1 vs elapsed time (T029, code): `artifacts/golden_f1_vs_elapsed.csv` and MLflow; golden macro-F1 0.21 at
    30 s, 0.23 at 60 s, 0.51 from 120 s, 0.516 at the end.
- **Left on those tasks:** T063 remove the old `danger_*` fields and settle API model loading with Alex (PR #32).
- **Next, in order:** monitoring, drift job, scenarios, workflows (#16-#21, CI unit-bug test T059); AI-infrastructure
  measurements on the fixed models (#3-#8, Vertex, GPUs only in `advanced-cloud-ai-i2`); README, compliance table,
  submission (#24).
- **PRs:** #32 (Alex -> main: Docker, web app, E3, kind) is open and must be merged **before** #9; then rebase #9. #9
  (dev_vincent -> main) stays a draft until #32 is merged; then mark it ready and ask @Xenodion for review.
  `dev_alex` is merged into `dev_vincent` (21/11).
- **Known limits (ADR 011):** one MLflow instance, SQLite snapshot (a hard kill can lose the last seconds of registry
  writes); E3 v2 is registered only on Alex's machine and must be republished to the shared registry.
- **To-do list:** `specs/001-final-deliverable/tasks.md` and GitHub issues (P0-P3, milestones 23/11, 24/11, 25/11).
- **Rules:** MLflow always on (`make dev-up`, scripts stop without it); no data, weights or secrets in the repo;
  English in everything committed; work on `dev_vincent`, PR to `main`; ask before any push.

## 21/11/2026 (night)

**Shared MLflow deployed** (ADR 011). Image from `docker/mlflow-gcp/` pushed to the private Artifact Registry,
Cloud Run service `ride-mlflow` (private, one instance, scale to zero). First attempt failed as feared: SQLite on a
Cloud Storage volume cannot work (out-of-order journal writes, 429 rate limits, container never healthy); fixed by
keeping the database on the container disk with a snapshot copied to the bucket after each change and at shutdown.
Smoke test through `gcloud run services proxy`: run, artifact in the bucket, database backup, anonymous call refused
(403). Teammates' school accounts granted on the bucket (object admin) and on the service (invoker, viewer).
DVC remote `shared` added; the first push was refused because this laptop's default Google credentials belong to
another project (OSPRI): no course data left for it. The README documents an isolated credentials file for that case.

## 21/11/2026 (evening)

**Debrief: two use cases had been mixed** (ADR 001). Rereading the results with Vincent showed that the "danger
score" design answered two different questions with one score: the ride's type (a whole-ride label, so
cumulative) and the risk of a fall (a property of a moment, so a short sliding window). The window had been tuned
on the fall metric to be applied to the ride type, and the study grew into a grid search out of scope for a course
graded on the system. Final design: model 1 = E3 cumulative (ride type, the brief's task); model 2 = fall risk on
10 s windows, may use model 1 as a feature; one service serves both. PR #9 put back to draft until the docs and the
shared infrastructure are clean.

**Window study closed** (ADR 002). Results of the 5 candidates, all tracked in MLflow: shorter windows separate
pre-fall riding from comparison riding better (AUC 0.72 at 60 s to 0.88 at 10 s and 0.96 at 5 s), while the
Reckless-vs-counterfactual separation stays flat (0.82-0.84). The 5 s value is suspected to include the start of the
fall, hence 10 s for model 2 with a 5 s anti-leakage gap and a 10 s sanity check (ADR 004). Model 1 keeps E3's 30 s.

**Vocabulary fixed:** "false alarms" measured on GBG500 were alarms on rides labelled non-reckless, not alarms
without a fall (GBG500 has no fall information). Model 2's alarms are measured on SRV3490, where an alarm before a
fall is a good alarm and an alarm on a comparison ride is a false one.

**Shared infrastructure started** (#30): private bucket `gs://albert-navy-ride-mlops` (europe-west1, project
`agentics-workflows-2`) and service account `ride-mlflow` (project `advanced-cloud-ai-i2`) with write access to it.
Cloud SQL is disabled in the course project, so the shared MLflow server will use SQLite on the bucket, a stated
limit. Access will be granted to the four team members' school accounts (not committed).

**Model 1 clarified.** E3 classifies each 30 s window on its own and averages the window probabilities: the
network has no memory, the cumulative mean is the memory. With 400 train rides, windows are what makes learning
possible (~7,500 examples). Weak point: the plain mean dilutes short episodes. Recorded as a planned improvement, not
scheduled (issue #31): top-k mean of the windows, then a learned aggregator, both recomputed at every live step.

**Model 2 output on a recorded ride:** a table (ride id, end of window, risk, alert), one row every 5 s; it is the
same computation as live, done at once; it serves the evaluation (does the risk rise before the gap?) and a
per-ride summary (when the ride was risky).

**Learned:** the "S3" of the brief is the storage protocol, served locally by SeaweedFS (as MinIO in the labs); no
AWS account is used. GCP is our cloud (ADR 006); the local S3-compatible storage stays for the dev stack.

## 21/11/2026 (afternoon)

**MLflow mandatory.** The first full 30 s run went untracked because the local MLflow server was down. Rule
(Vincent): every run is tracked; the study now stops when MLflow is unreachable (smoke runs included, tagged
`run_type=smoke`) and logs per candidate the parameters, metrics, code and data versions, the trained weights and
the fall-group table (commit ff243a9). The 30 s candidate was rerun with MLflow on a clean tree (run 35c834ee): the
metrics are identical to the untracked run, a reproducibility check for free.

**First result, 30 s window (measured, laptop CPU, 10 epochs, 83,610 training windows):** danger before a fall vs
comparison rides, paired AUC 0.75 (95 % CI 0.73-0.78) on 485 development groups (0.75 on the 445 complete groups);
Reckless vs counterfactual categories AUC 0.84 (5 Reckless train rides, wide CI); 4 of 5 Reckless train rides above
the threshold; false alarms Safe 5 %, Bad weather 6 %, Tandem 21 %, Safe overall 27 %. The label-free score rises
before falls; the counterfactual trap of `data.txt` is real for Tandem and Safe overall. Training: 13 min, peak
memory 6.7 GB.

**Memory bug fixed** (commit 13df52d): the reconstruction errors were computed on all windows in one forward pass
(about 3 GB of activations for the first layer alone); now batched, standardised in place, timelines built by
chunks of rides; a test checks batching does not change the scores. Treated as a development bug, not as an
AI-infrastructure optimisation (the infrastructure work starts on the fixed model). The 30 s run keeps the memory
measured before the fix; the other candidates run after it (memory only, scores unaffected).

**Team sharing problem** (issue #30): each teammate's MLflow and DVC remote live in their own Docker volumes, so
runs are not shared. Proposed: export run metadata to the repo, a private GCS bucket for artifacts and data, a
hosted MLflow server later if time allows.

## 21/11/2026 (morning)

**Reframing after reading the course `data.txt`** (ADR 001). The archive shipped with SRV3490 contains a data
description: GBG500 labels are "used only for evaluation", Reckless is "the positive class we aim to detect", and
Safe overall, Bad weather and Tandem are counterfactuals that "should not be flagged". Together with the teacher's
briefing (option 2: unsupervised, validated with labels, linked to falls), this showed that our 20/11 decision
(supervised classifier) was option 1. We also found that the golden set holds a single Reckless ride. New model: a
danger score learned without labels on all ~4,000 rides, cumulated live, plus a 5-class head for the brief; E3
kept as reference. Spec, plan, research, contracts and issues updated (commits f5bde04, d816e61).

**Falls as a label-free check** (ADR 004). Vincent's idea: with only 6 Reckless rides, use the 698 SRV3490 falls to
ask whether danger precedes a fall. Fall groups split 70/30 by city (489 development, 209 held-out groups, a group
never split); pre-fall segment = 60 s ending 2 s before the fall; comparison rides scored at the same elapsed time
(commit e4a02b8). Using falls as training examples ("weak supervision") stays an experiment, only if this check
shows a signal (issue #28).

**Built, by five parallel agents then reviewed** (commits 37310be to 958fc87, then c163a67 to 8248e55):

- visual charter: palette, theme, `save_fig`, palette test F36 (Mako kept over viridis);
- timeline: cumulative values window by window, `predict_timeline` on E3;
- SRV3490 loader: 3,490 rides, 698 crash rides with exactly 4 comparison rides each, 8 cities, 100 ms sampling,
  no id overlap with GBG500; notebook 02 with F04 and F05;
- `bench.py`: one benchmark tool, CSV schema fixed by contract, Python 3.10-compatible for Vertex AI; first laptop
  CPU baseline (measured): 0.9 ms per window at batch 1, ~9,300 windows/s at batch 512;
- `DangerModel`: label-free convolutional autoencoder + logistic 5-class head; a test checks the encoder never reads
  labels;
- API: danger score and timeline in `/v1/predict`, stateless `/v1/predict/window` (ADR 005), `/v1/feedback`,
  metrics of monitoring layers 3 and 4;
- window study rewritten for the danger detector; pre-registration revised before any full run (ADR 002).

**Corrected during review:**

- the SRV3490 rotation factor had been set to the measured ratio (60.04); replaced by 180/pi (ADR 003);
- the timeline returned the first window's value before that window was complete, which uses future data
  (a 60 s window would have "answered" at 30 s); it now returns no answer;
- the pre-fall score was the cumulative danger since the start of the ride; replaced by the mean over the windows
  ending in the pre-fall segment;
- 1.6 % of SRV3490 sensor values are missing: cleaning moved into one loader and a DVC stage `prepare_srv`
  (commit 648fe25); the served model still cleans every ride it receives, so live data goes through the same step.

**Data facts:** SRV3490 tracked with DVC from the course archive (the archive itself is git-ignored). 334 of 2,792
comparison rides are shorter than their fall's pre-fall segment and give no pair. No fall happens before 120 s of
riding.

**Open at that point:** full window study; CI test of the
unit-bug drift (task T059); fall analysis on held-out groups (issue #29).

## 20/11/2026

**Teacher's briefing.** Two options (supervised on 500 rides, or unsupervised on all data validated with labels);
not a plain k-means; explore features predictive of falls. Written up in `docs/PROJECT_FRAMING.md` (commit beb6e1f).

**First framing: a cumulative live classifier** (superseded on 21/11). A ride read window by window, its category
"so far" at every moment. Fall forecast and live replay as bonus (commit c1f043c).

**One design for both courses** (commit 75869b5): live load test (one Locust user = one ride streaming at real
speed), inference hardware CPU / 1 GPU / 2 GPUs, training cost vs data volume (real data replicated x1-x8), drift
scenarios built by resampling real rides (new city, real sensor unit difference, riskier mix).

**Planning with Spec Kit** (commits aa72d7f to 01df461): constitution (8 principles, local), spec with 4 user
stories and a 46-figure catalogue, plan in phases (core MLOps first, AI infrastructure in parallel, bonus last),
research decisions, contracts, 54 tasks, 19 GitHub issues with colour-coded priority labels and checkpoint
milestones (23/11, 24/11, 25/11 20:00). Four-layer monitoring, drift job and hybrid retraining with human approval
made core (ADR 008). GCP confirmed as the only allowed deviation from the brief (ADR 006). Visual charter fixed
(ADR 009).

**Window length questioned** (ADR 002): 30 s is not the data's granularity (10 Hz) but E3's untested choice; made an
open choice to be decided by a pre-registered study.

## 16/11 - 19/11/2026 (before this journal)

System built from the course labs: DVC pipeline (prepare, train, evaluate with a quality gate), frozen golden split,
statistics baseline then 1D CNN (E1-E3, E3 golden macro-F1 0.51), MLflow tracking and registry with aliases,
FastAPI service, Streamlit web app, Docker images on Docker Hub, CI, deployment on kind with a canary. Details in
`docs/PLAN.md` and the git history (commits 5bd852a to 0d35241).
