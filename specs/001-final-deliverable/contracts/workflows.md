# Contract: GitHub Actions workflows

| Workflow | Triggers | Runner | Does | Gate / flag |
|---|---|---|---|---|
| `ci.yml` (exists, extended) | push, pull request | hosted | lint, format, unit + integration tests, image build + container smoke test, **e2e replay test** on the compose stack | fails the run |
| `train.yml` (new) | `schedule` (weekly cron), `workflow_dispatch` with inputs `reason` (`schedule \| input_drift \| prediction_drift \| performance \| manual`) and `evidence`, `repository_dispatch` type `retrain` (bonus D3) | self-hosted | `dvc pull`, `dvc repro`, tag the MLflow run with `trigger_reason` / `trigger_evidence`, register `@staging` if the gate passed, summary with gate table and F1-vs-time | `evaluate` exit 1 stops it: nothing registered |
| `cd.yml` (new) | push of a tag `v*` or `workflow_dispatch` with `model_version` | hosted (build, push) + self-hosted (deploy) | build and push code-only images to Docker Hub, then job `deploy` in environment **`production`** (required reviewer: human approval), `kubectl apply`, rollout status, smoke test, **automatic `kubectl rollout undo`** on failure, move `@production` alias after success | approval + smoke test |
| `drift.yml` (new) | `schedule` (daily), `workflow_dispatch` with `current` (log or scenario) | self-hosted | run the drift job, append `drift_report.md` to the summary, upload the report as an artifact | exit 3 → `::warning::` flag "retraining recommended: <reasons>" and the command to dispatch `train.yml`; bonus D3 dispatches it |

Secrets (repository settings, never in files): `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`. MLflow and S3 endpoints
on the self-hosted runner come from its local environment.

Traceability (SC-011): every production change = a `cd.yml` run linking the model version, its MLflow run
(trigger tags, gate result) and the approver.
