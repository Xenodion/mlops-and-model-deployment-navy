# ADR 008 - Hybrid retraining policy with a human approval before production

**Status:** accepted, 20/11/2026.

## Context

The MLOps course teaches four monitoring layers (infrastructure, inputs, predictions, outcome) and asks when to
retrain. Retraining on every alert is wrong: some alerts are upstream bugs (ADR 003), and an automatic promotion
of a worse model is a risk.

## Decision

- Four triggers start one training workflow (`train.yml`) with a recorded reason: a weekly schedule, input drift
  (PSI above 0.25 on a monitored feature), prediction drift (category mix test), performance drop (accuracy on
  labels received later).
- The quality gate decides registration (`@staging`); a human approves production in `cd.yml` (GitHub
  environment `production`); a failed smoke test rolls back automatically.
- The runbook separates retraining from an upstream bug: rotation features moving by a constant factor near 57 =
  unit bug, fixed at the source, no retraining; a new population = retrain.
- Where to retrain is decided from the measured training cost (CI self-hosted runner by default; Vertex AI above
  the volume where it exceeds the time budget).

## Alternatives rejected

- Fully automatic retrain-and-promote: kept as a bonus (issue #27), always with the human approval.
- Schedule only: misses drift between two runs.

## Consequences

If a part cannot be automated before the deadline, the measurements and CI flags still exist and the
presentation states the recommended policy and what is missing.

## Evidence

`docs/PROJECT_FRAMING.md` §4, research R13, contracts `workflows.md`, issues #17 and #19.
