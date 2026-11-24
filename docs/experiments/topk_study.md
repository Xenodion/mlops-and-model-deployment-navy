# E5 - Top-k aggregation of model 1's windows (issue #31, phase A)

## Pre-registration (written 24/11, before any run)

**Question.** Model 1 (E3, the registered CNN, v2) labels a ride with the **plain mean** of its window
probabilities. The mean dilutes short episodes: a ride labelled Reckless may hold only a few reckless moments. Does a
**top-k mean** (per category, the mean of its k highest window probabilities; the label is the category with the
highest value) do better?

**What is fixed.** E3 exactly: its settings in `params.yaml` (`train.cnn`), its seed, its 5 cross-validation folds
(`data/processed/splits.csv`). The network is not changed; only the way its window probabilities are combined.

**Data.** Out-of-fold window probabilities of the 400 train rides: each ride is scored by the fold model that never
saw it. **The 100 golden rides are not read** by this study.

**Candidates** (9, plus the reference):
- reference: plain mean (E3);
- top-k, k ∈ {1, 2, 3, 5, 8} windows;
- top-share, the highest {10, 20, 30, 50} % of the ride's windows (at least 1): train rides have 11 to 30 windows
  (median 18), so a fixed k is a different share of each ride.

**Sanity check, first.** The plain mean must reproduce E3's cross-validated macro-F1 (0.357, `artifacts/cv_metrics.json`)
within 0.005. If not, the study is void (the fold models differ from E3's).

**Metric.** Cross-validated macro-F1 (pooled over the 400 rides, as for E3), with the per-fold values; also the
recall of each category and how many rides change label.

**Decision rule.** The best candidate is adopted for phase B (a new model version, scored once on the golden rides,
through the gate) **only if both**:
1. its cross-validated macro-F1 is at least **0.03** above the plain mean's;
2. it beats the plain mean on **at least 4 of the 5 folds**.

Otherwise the plain mean stays. Choosing the best of 9 candidates on the same folds flatters its score a little:
phase B's single golden score is the unbiased check.

**Tracking.** One MLflow run `E5-topk-study` (experiment `ride-classifier`, tag `run_type=study`): the results table
and the out-of-fold window probabilities as artifacts.

```bash
uv run python -m ride_classifier.experiments.topk_study    # needs MLflow (make dev-up), ~10 min on a laptop CPU
```

## Results (24/11, laptop CPU; MLflow run `E5-topk-study`)

**Sanity check passed:** the plain mean reproduces E3 exactly (CV macro-F1 0.357). **Decision: not adopted.** No
candidate beats the plain mean; the best, top-3, is 0.004 below it and better on 3 folds out of 5.

| Candidate | CV macro-F1 | Std over folds | Folds better than the mean | Rides changed | Recall Bad weather | Recall Reckless |
|---|---|---|---|---|---|---|
| **mean (E3)** | **0.357** | 0.071 | – | – | 0.111 | 0.2 |
| top-1 | 0.324 | 0.068 | 1 | 28 | 0.111 | 0.2 |
| top-2 | 0.348 | 0.075 | 3 | 22 | 0.167 | 0.2 |
| top-3 | 0.353 | 0.073 | 3 | 16 | 0.167 | 0.2 |
| top-5 | 0.343 | 0.070 | 1 | 11 | 0.111 | 0.2 |
| top-8 | 0.349 | 0.069 | 1 | 9 | 0.111 | 0.2 |
| top-10 % | 0.352 | 0.073 | 3 | 19 | 0.167 | 0.2 |
| top-20 % | 0.341 | 0.070 | 1 | 12 | 0.111 | 0.2 |
| top-30 % | 0.343 | 0.072 | 1 | 9 | 0.111 | 0.2 |
| top-50 % | 0.342 | 0.070 | 0 | 8 | 0.111 | 0.2 |

Reading: top-k does not rescue the Reckless rides (the same 1 of 5 train Reckless rides is found by every
candidate): on the rides it misses, the CNN's windows are not strongly Reckless anywhere, so there is no diluted
episode to recover. It slightly helps Bad weather (2 rides more for top-2/3) but costs Safe rides. The limit is the
network's window scores (weak labels: every window inherits its ride's label), not the way they are combined.
Phase B is not started; the golden rides were not read.
