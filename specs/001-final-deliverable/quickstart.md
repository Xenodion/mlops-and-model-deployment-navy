# Quickstart: validation scenarios

Each scenario proves one slice of the deliverable end to end. Commands use `make` targets listed in the plan
(new ones are created by the tasks); expected outcomes refer to the contracts. Run on the laptop, **one stack at a
time** (compose or kind), Docker closed otherwise.

## Prerequisites

`make install`, `make dev-up`, `make data` (GBG500; SRV3490 once available), `make pipeline` (passes the gate),
`make register`, `make export`.

## Q1 - Live classifier and evaluation (Phase A2)

`make pipeline` → `artifacts/golden_metrics.json` holds macro-F1 at 7 elapsed times and the gate result;
`uv run pytest tests/unit/models` passes the invariant "last cumulative = whole-ride prediction".
Expected: gate passes (unchanged E3 result); F07, F08 saved by `03_live_classifier`.

## Q2 - API timeline and streaming (A3)

`make api`, open `/docs`: `POST /v1/predict` with an example ride returns `timeline` (contracts/api.md);
`POST /v1/predict/window` twice, feeding the returned `state`, gives `n_windows = 2`; `POST /v1/feedback` returns
204. Expected: `/metrics` shows `input_window_feature_bucket`, `prediction_confidence`, `feedback_total`.

## Q3 - Monitoring and drift in action (A4-A6, compose)

`make app-up monitoring-up`, then `make scenario NAME=normal` (5 min) and `make scenario NAME=riskier_mix`.
Expected: Grafana (light theme) shows all four rows moving; PSI stays < 0.1 under `normal`;
`riskier_mix` raises `PredictionMixShift` and, with feedback, `AccuracyDrop` to firing in Prometheus > Alerts.
With SRV3490: `unseen_city` and `unit_bug` push `drift_psi` ≥ 0.25 (`InputDriftHigh`); in `unit_bug` the rotation
features carry the jump (runbook fork).

## Q4 - Drift job and retraining flag (A5, A7)

`make drift CURRENT=reports/scenarios/unit_bug.parquet` → exit 3 and `drift_report.json` with
`status = retrain_recommended`. On GitHub: dispatch `drift.yml` → yellow warning flag and report in the summary;
dispatch `train.yml` with `reason=input_drift` → registered `@staging` only if the gate passes, MLflow run tagged.

## Q5 - Deployment with approval and rollback (A7, A8, kind)

`make kind-up`, push a `v*` tag → `cd.yml` waits on the `production` environment; approve → rollout, smoke
test green, `@production` moved. Break the smoke test on purpose (bad `MODEL_URI` in a test tag) → automatic
`rollout undo`, previous version still answering. Time the rollback (F34).

## Q6 - Live load test (B6, kind)

`make loadtest SPEED=10` against the CPU-limited API (1 replica) → `reports/load/*_stats.csv` and
`loadtest/plot_capacity.py` → F31: the step where p95 crosses 200 ms and the live-ride capacity. Grafana row 1
recorded during the run (F32).

## Q7 - Benchmarks (B2-B5)

Local: `uv run python ai-infrastructure/benchmarks/bench.py inference --device cpu ...` → CSV with every column of
contracts/benchmark-results.md. Cloud: `uv run python ai-infrastructure/vertex/submit.py --target 1xa100` →
SPOT job ends by itself; the CSV is extracted from the logs. After each cloud session: no running job or
endpoint (`gcloud ai custom-jobs list --account=vlamy@albertschool.com`), credits logged (SC-009).

## Q8 - Submission readiness (Phase C)

README compliance table: every brief line ticked or stated as a gap; `assets/figures/README.md`: every core ID
`final`; each demo has a fallback in `assets/demos/`; timed dry run under 10 min.
