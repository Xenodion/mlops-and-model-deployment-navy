# ADR 006 - GCP instead of AWS; GPU runs as self-terminating Vertex AI SPOT jobs

**Status:** accepted, 20/11/2026.

## Context

The MLOps brief mentions AWS, but the school gives us access to GCP only. GPUs are available only in the project
`advanced-cloud-ai-i2`, where we cannot write to Cloud Storage nor read quotas; Compute Engine GPU quota is 0. A
2xA100 Vertex AI job in SPOT mode succeeded there on 12/11.

## Decision

- Cloud = GCP. Local and cluster storage stays S3-compatible (SeaweedFS), as the course teaches.
- GPU measurements run as Vertex AI custom jobs in SPOT mode (they stop by themselves), code sent with the job,
  results printed between `@@CSV` / `@@END` in the logs. Data goes in a private bucket of a project where we can
  write, or in a private Artifact Registry image.
- Free resources first (laptop, Colab, Kaggle); every billable run has a planned teardown and a credit log.

## Alternatives rejected

- AWS: no access. Compute Engine GPU VMs: quota 0 and a teardown risk.

## Consequences

The README states the deviation from the brief. GPU results depend on SPOT capacity; a refused job is kept as
evidence and reported.

## Evidence

`docs/PROJECT_FRAMING.md` §4, research R9-R11, issues #4 and #6.
