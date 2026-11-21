# ADR 002 - Window lengths: 30 s for model 1, 10 s for model 2

**Status:** accepted, 21/11/2026. The window study is closed.

## Context

The sensors give one reading every 100 ms. E3 uses 30 s windows every 15 s, a choice never compared with others. A
number that is neither a default nor the data's granularity must be explained before the constraints that follow
from it (API minimum length, request size, update rate, load-test rate) are accepted. Model quality is not graded,
so the explanation must be short, not a search.

## Decision

- **Model 1 (ride type): 30 s, stride 15 s**, as validated in E3. Its output is a cumulative mean over the ride,
  which smooths the window choice; 30 s gives each window enough riding to carry a style.
- **Model 2 (fall risk): 10 s.** A short window reacts to a period, which is what a fall risk is about. Measured on
  21/11 with the label-free score (laptop CPU, 489 development fall groups, 2 s margin before the fall): paired AUC
  pre-fall vs comparison rides 0.72 (60 s), 0.75 (30 s), 0.84 (15 s), 0.88 (10 s), 0.96 (5 s). 5 s was not kept:
  the jump to 0.96 while the Reckless-vs-counterfactual separation stayed flat (0.82-0.84) suggests the start of
  the fall leaking into the last windows; 10 s with a 5 s anti-leakage gap (ADR 004) is the cautious choice.

## Alternatives rejected

- Keep 30 s for both: worse for a momentary risk (measured above).
- A full pre-registered grid search with bootstrap CIs (what we started on 21/11): out of scope for a course
  graded on the system; its runs are kept in MLflow (`E4-window-*`) and in `docs/experiments/window_study.md` as the
  measurement behind the 10 s choice.

## Consequences

The API keeps 300 readings (30 s) as the minimum for model 1; model 2 scores 100-reading windows. The load test
streams 10 s windows every 5 s for model 2 and updates model 1 every 15 s.

## Evidence

`artifacts/window_study.csv`, MLflow experiment `ride-classifier`, runs `E4-window-*` (commit 76a7db3), issue #11.
