# ADR 007 - Published Docker Hub images carry neither the model nor data

**Status:** accepted, 20/11/2026; existing tags to handle with the account owner.

## Context

The brief requires images on Docker Hub. The GitHub repository is public, and course data and models trained on it
must not be published. Earlier image tags (`-m2`) had the model baked in.

## Decision

Images contain code only. The API loads the model at startup from the MLflow registry
(`MODEL_URI=models:/ride-classifier@production`, artifacts on the cluster's storage). Tags with a baked-in model
are made private or deleted.

## Alternatives rejected

- Private Docker Hub repositories only: the free plan allows one.
- Keep the model in the image: simpler start-up, but publishes a model trained on course data.

## Consequences

The pod needs the registry reachable at startup; the smoke test and the rollback cover that path.

## Evidence

Research R12, issue #23.
