# E4 - Window length study (pre-registration)

Written before any run. Rule source: `specs/001-final-deliverable/research.md` R15. Tracked by GitHub issue #11.
Code: `src/ride_classifier/experiments/window_study.py`. Results go in `artifacts/window_study.csv`, figure F47.

## Question

Which window length should the ride classifier use: does a shorter window give a reliable live label earlier,
at little cost on the whole-ride label?

## Why

The data has one reading every 100 ms (10 Hz). The 30 s window (stride 15 s) is E3's modelling choice and was
never compared with other lengths (`docs/PLAN.md` section 6.5, "Not tried"). The minimum ride length, the
payload of `/v1/predict/window`, the update rate of the live label and the load-test request rate all follow from
it, so none is accepted until this test is done.

## Hypothesis

A shorter window gives a reliable cumulative label earlier, with little loss on the whole-ride label. Below some
length a window carries too little context and the whole-ride score drops.

## Candidates

Window 5, 10, 15, 30, 60 s (50, 100, 150, 300, 600 readings); stride = window / 2.

## Fixed conditions

- Everything else from `params.yaml` `train.cnn` (architecture, filters, kernel 7, dropout, epochs 10, batch
  size, learning rate, weight decay, augmentation, `class_weight: sqrt`). One change only: window length (and the
  stride that follows it).
- Same seed (`data.seed`) for every candidate; same 5 CV folds, read from `data/processed/splits.csv`.
- Train rides only. The golden rides are excluded: never read, never used to choose.

## Metrics

Mean and std over folds, unless stated:

- Whole-ride macro-F1 on out-of-fold predictions (per fold, then mean +/- std).
- Cumulative macro-F1 at 30, 60, 90, 120, 180, 240 s and at the end of the ride (all OOF rides pooled).
- Time to first answer = window length.
- Per-window CPU latency: median of 200 single-window calls after 20 warm-up calls, default torch threads,
  laptop CPU (recorded with the CSV).
- Training time (total over the folds).

## Time steps after pooling (T013)

The network is 3 blocks of convolution (kernel 7, padding 3: length unchanged) and `MaxPool1d(2)`
(length -> floor(length / 2)). So the number of steps is floor(floor(floor(n / 2) / 2) / 2). Checked by
instantiating `RideCNN` on a dummy tensor.

| Window | Readings | After pool 1 | After pool 2 | After pool 3 (measured) |
| --- | --- | --- | --- | --- |
| 5 s | 50 | 25 | 12 | 6 |
| 10 s | 100 | 50 | 25 | 12 |
| 15 s | 150 | 75 | 37 | 18 |
| 30 s | 300 | 150 | 75 | 37 |
| 60 s | 600 | 300 | 150 | 75 |

The shortest window keeps 6 steps (at least 3), each with a receptive field far larger than the kernel, and the
final layer averages over time anyway. No adaptation is needed: the architecture is identical for all candidates.

## Decision rule (verbatim from R15)

Keep the candidates whose whole-ride OOF macro-F1 is within one fold-std of the best; among them pick the one
with the highest cumulative macro-F1 at 60 s; tie -> the shorter window. Implemented as `choose_window`
(the fold-std is the std of the best candidate). All candidates are logged in MLflow
(`experiment_id` E4-window-*), results in `artifacts/window_study.csv`, figure F47.

## After the study

Only then: the chosen length goes through `evaluate` (golden scored once, gate). If it is not 30 s, it is
registered as a new version and replaces E3 after the gate; E3 stays the fallback.

## Threats to validity

- Very few Reckless rides (6 in total): per-fold macro-F1 for that class is close to noise, and some folds hold
  1 or 0 of them.
- Fold variance is large (E3 CV 0.36); differences between candidates may be inside it, which is why the rule
  uses one fold-std.
- Known confound: a shorter window with stride = window / 2 gives more windows per ride, hence more training
  steps per epoch. Epochs are kept fixed (10) as the pre-registered choice; the number of training windows per
  candidate is reported (`n_windows_train`) so the confound can be read, not removed.
- One seed per candidate: seed-to-seed variance is not measured.
- Cumulative F1 at a given time only counts rides that reached that time (`n_rides`, `n_rides_complete` are
  reported); the pool changes with the time point.

## Results

To be filled after the run.
