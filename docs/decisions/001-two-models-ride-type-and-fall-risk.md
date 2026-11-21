# ADR 001 - Two distinct models: ride type (cumulative) and fall risk (sliding window)

**Status:** accepted, 21/11/2026 (evening). Supersedes two earlier readings of the same day: a supervised cumulative
classifier only (20/11), then one label-free "danger score" used for everything (21/11 morning).

## Context

- The MLOps brief asks for a deep learning model that classifies rides into 5 categories and beats random.
- The teacher's briefing (20/11) asked to use all the data, validate with the labels, and look at falls.
- The course `data.txt`: labels "used only for evaluation"; Reckless is "the positive class we aim to detect";
  Safe overall, Bad weather and Tandem are counterfactuals that "should not be flagged"; SRV3490 has 698 rides with
  a confirmed fall, each with 4 comparison rides.
- The morning version mixed two use cases into one score: a ride's category describes the whole ride, while a fall
  risk describes a moment. It also tuned the window on the fall metric to apply it to the ride type.

## Decision

1. **Model 1 - ride type**: E3 (the registered CNN, 30 s windows every 15 s), read cumulatively: at every step,
   the category of the ride from its start; at the end, the ride's category. Evaluated as the brief asks (golden
   macro-F1 vs random, gate) and vs elapsed time; Reckless and the counterfactual categories reported explicitly.
2. **Model 2 - fall risk**: a short sliding window (10 s) scored continuously, not cumulative; trained on SRV3490
   (pre-fall windows of fall rides vs the same moments of their comparison rides) with an anti-leakage gap (ADR
   004); may use model 1's current output as a feature. Evaluated on held-out fall groups.
3. Two registered models in MLflow, each with its own pipeline, gate and version; served together (ADR 010).

## Alternatives rejected

- One model for both: a whole-ride label and a momentary risk need different aggregations and evaluations.
- A label-free score as the ride-type model: more ML work, not graded; E3 already passes the gate.
- Fall detection after the fact: no value (the sensor and the rider already know).

## Consequences

- Model 1 is ready (E3 + `predict_timeline`); model 2 is new work, second in priority after CI/CD and cloud.
- The label-free autoencoder built on 21/11 is kept as a candidate feature extractor for model 2.
- Model quality is not graded: no further parameter search; numbers are chosen and explained (ADR 002).

## Evidence

`docs/PROJECT_FRAMING.md` §3, journal 21/11 evening, issues #14 and #29.
