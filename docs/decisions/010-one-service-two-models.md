# ADR 010 - One inference service serving both models

**Status:** accepted, 21/11/2026.

## Context

Model 1 (ride type) and model 2 (fall risk) are trained, versioned and gated separately (ADR 001), but a live ride
needs both answers at once, and model 2 may use model 1's output as a feature.

## Decision

- Two registered MLflow models (`ride-classifier` = model 1, `fall-risk` = model 2), each with its own alias
  `@production`.
- One API image that loads both at startup. On each live window: model 1 updates the cumulative category, model 2
  gives the current fall risk. On a whole recorded ride: model 1's category; model 2's per-window risk returned for
  analysis and marked retrospective.
- The API answers model 1 alone if model 2 is missing (degraded mode), never the reverse.

## Alternatives rejected

- Two services: two deployments, a network hop between them when model 2 uses model 1, double monitoring.
- One merged model: couples two release cycles and two gates.

## Consequences

For the AI-infrastructure course: latency and memory are measured for both models per window in one pod, and the
scaling plan compares one pod serving both with two separate services.

## Evidence

`docs/PROJECT_FRAMING.md` §3, `specs/001-final-deliverable/contracts/api.md`.
