# ADR 011 - Shared MLflow and DVC on GCP, local dev stack kept

**Status:** accepted, 21/11/2026.

## Context

Each teammate ran MLflow and the S3-compatible storage (SeaweedFS) in their own Docker volumes, so nobody saw the
others' experiments, registered models or versioned data. The MLflow lab of the course shows a shared server
(`MLFLOW_TRACKING_URI=http://<host>:5000`) as the way to share runs. "S3" in the brief is the storage protocol
(MinIO in the labs, SeaweedFS here), not an AWS account; our cloud is GCP (ADR 006).

## Decision

- **Shared MLflow tracking server** on Cloud Run (`ride-mlflow`, project `advanced-cloud-ai-i2`, europe-west1):
  private (no anonymous access), one instance, scale to zero, 1 vCPU / 2 GiB (1 GiB was killed for memory during the migration: `/tmp`, which holds the registry database, is in RAM on Cloud Run). Image built from
  `docker/mlflow-gcp/` (same MLflow 3.17.2 as the local stack) and stored in the private Artifact Registry
  repository `cloud-ai`.
- **Artifacts and DVC data** in one private bucket, `gs://albert-navy-ride-mlops` (project `agentics-workflows-2`,
  europe-west1, public access prevented): `mlflow-artifacts/` (written by the server, so clients need no bucket
  access for MLflow), `dvc/` (DVC remote `shared`), `mlflow-state/` (registry backup).
- **Registry database:** SQLite on the container disk, restored from the bucket at start and copied back after
  every change and at shutdown (`docker/mlflow-gcp/serve.py`). Tested on 21/11: SQLite directly on the Cloud
  Storage volume fails (out-of-order journal writes, rate limits).
- **Access:** teammates' school accounts, `roles/storage.objectAdmin` on the bucket, `roles/run.invoker` and
  `roles/run.viewer` on the service only. Developers reach the server with `make shared-mlflow` (authenticated
  proxy on `localhost:5001`); no public URL.
- **Local dev stack unchanged** (Docker Compose: MLflow, Postgres, SeaweedFS), as the brief asks for a dev
  environment; the default DVC remote stays the local `storage`, the shared one is opt-in (`make shared-data-*`).
- Course data and models are never published: the bucket and the registry are private.

## Alternatives rejected

- Export / import of run metadata through the repo: a workaround, not the documented way.
- Cloud SQL for the registry: the API is disabled in the course project and has a fixed monthly cost.
- Public MLflow URL with a password: more exposure for no benefit.

## Consequences

- Limits, stated: one instance; a hard kill can lose the last seconds of registry writes (artifacts are safe);
  SQLite suits a team of five, not concurrent heavy use.
- Cost: near zero (scale to zero, a few cents of storage). **Teardown:**
  `gcloud run services delete ride-mlflow --project advanced-cloud-ai-i2 --region europe-west1` and
  `gcloud storage rm -r gs://albert-navy-ride-mlops`.
- CI/CD: the self-hosted runner logs to the shared server and pushes to the shared DVC remote with the runner
  owner's credentials (to wire in the workflows, issue #19).
- Vertex AI jobs log to a file store in the job and import their runs into the shared server (research R9).
- On a machine where the default Google credentials belong to another project, DVC must use an isolated
  credentials file (see the README), never the default one.

## Migration of existing runs

21/11: the 11 local runs (baselines, E1-E3, window candidates) were copied to the shared server with parameters, metric history, tags and artifacts, plus the tag `migrated_from_run_id` (verified: 11 runs, all FINISHED, all with artifacts). No model was registered on the local server; E3 version 2 lives on Alex's machine and is republished from there.

## Evidence

Smoke test 21/11: run and artifact written through the proxy, database backup present in the bucket, anonymous call
refused (403). Issue #30; `docker/mlflow-gcp/`, `.dvc/config`, `Makefile` targets `shared-*`.
