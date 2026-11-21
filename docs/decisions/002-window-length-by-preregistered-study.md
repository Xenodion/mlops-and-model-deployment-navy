# ADR 002 - The window length is decided by a pre-registered study

**Status:** accepted 20/11/2026 (rule revised 21/11 before any full run); study running.

## Context

The sensors give one reading every 100 ms. The model cuts rides into 30 s windows every 15 s, a choice inherited
from experiment E3 and never compared with other lengths. Many things follow from it: the minimum ride length the
API accepts, the size of a live request, how often the live score is updated, the load-test request rate, and the
earliest moment an alert can be raised. A parameter that is neither a standard default nor the finest granularity
of the data must be justified before the constraints that follow from it are accepted.

## Decision

Compare 5, 10, 15, 30 and 60 s windows (stride = half the window) with everything else fixed, under a rule written
before running (`docs/experiments/window_study.md`):

- main signal: paired AUC of the danger before a fall vs the comparison rides of the same fall group, on the
  development fall groups (489 groups), with a bootstrap CI;
- tie-breaker: ROC-AUC Reckless vs the counterfactual categories on the GBG500 train rides;
- keep the candidates within the best one's CI, take the best tie-breaker, then the shorter window.

Golden rides and held-out fall groups are never used. Every number derived from the window stays provisional
until the study decides.

## Alternatives rejected

- Keep 30 s without testing: unjustified.
- The shortest possible window (one reading): a reading carries no riding style; 5 s is the shortest candidate
  that keeps 6 time steps after the network's pooling (computed and measured).
- Choose on golden rides: they would no longer be a fair final test.

## Consequences

- If a window shorter than 30 s wins, the API's 300-reading minimum is relaxed accordingly.
- The first version of the rule (supervised macro-F1) was never run; it was rewritten after ADR 001.
- The study also measures training time and memory per window length (AI-infrastructure sizing).

## Evidence

`docs/experiments/window_study.md`, research R15, `src/ride_classifier/experiments/window_study.py`, issue #11,
results in `artifacts/window_study.csv` (to come), figure F47.
