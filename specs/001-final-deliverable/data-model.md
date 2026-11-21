# Data model: 001 final deliverable

Entities from the spec (Key Entities), with fields, rules and states. Types are logical; the storage is given
per entity.

## Ride

| Field | Type | Rule |
|---|---|---|
| `ride_id` | str | unique within its dataset; GBG500 and SRV3490 ids never overlap (checked at load) |
| `dataset` | `GBG500` \| `SRV3490` | |
| `readings` | table `time_index` (ms, non-decreasing), `ax ay az` (m/s²), `rx ry rz` (deg/s) | 10 Hz; SRV3490 rotation multiplied by the verified factor (R2) before any shared use |
| `label` | one of 5 categories \| null | GBG500 only |
| `split` | `train` \| `golden` | GBG500 only; from the committed `data/processed/splits.csv`, never recomputed |
| `cv_fold` | int | train rides only |
| `city` | str \| null | SRV3490 only |
| `fall_time_index` | int \| null | SRV3490: first reading with `topple = 1` |
| `fall_group_id` | str \| null | SRV3490: shared by a fall ride and its 4 comparison rides |

Storage: `data/raw/*.parquet` (DVC), `data/processed/*` (DVC outputs).

## Window

| Field | Type | Rule |
|---|---|---|
| `ride_id`, `window_index` | str, int | window `k` covers `[k·stride_s, k·stride_s + window_s)`; `window_s` = 30, `stride_s` = 15 (`params.yaml`) |
| `end_s` | float | elapsed ride time when the window is complete; used for "label after t seconds" |
| `probabilities` | 5 floats, sum 1 | per-window category probabilities from the 5-class head: a **hint**, never compared with a label |
| `danger` | float | per-window danger score: reconstruction error of the label-free autoencoder, z-scored on train windows |
| `stats` | 8 floats | monitored features (`features/window_stats.py`): std of 6 channels, norm of acc and rot |

## Timeline

| Field | Type | Rule |
|---|---|---|
| `ride_id` | str | |
| `windows` | list of Window (probabilities, end_s) | ordered by `window_index` |
| `cumulative` | list of 5-float vectors | `cumulative[k]` = mean of `windows[0..k].probabilities` (R1); uses only data up to `end_s[k]` |
| `cumulative_danger` | list of floats or null | running mean of `windows[0..k].danger`; null for E3 |
| `label`, `confidence` | category, float | argmax and max of `cumulative[-1]`; equals the whole-ride prediction |

Invariants (tested): `cumulative[-1] == predict_proba(ride)` within 1e-6; before the first window ends there is no
answer (no future data).

## Live window state (API streaming)

| Field | Type | Rule |
|---|---|---|
| `sum_proba` | 5 floats ≥ 0 | sum of the per-window probabilities so far |
| `sum_danger` | float | sum of the per-window danger scores so far (0 for E3) |
| `n_windows` | int ≥ 0 | windows seen so far; `0` with zero sums at the first window |

Held by the client; the server is stateless (R4). Transition: `(sum, n) + window p → (sum + p, n + 1)`,
cumulative = `(sum + p) / (n + 1)`.

## Evaluation records (R16, SC-003)

- **Danger on labelled rides**: one row per GBG500 ride (all 500): `ride_id`, `label`, `split`, whole-ride
  `danger_score`, `is_dangerous`; summary: Reckless detected k / 6, false-alarm rate per category, ROC-AUC Reckless
  vs counterfactuals, bootstrap CIs. Threshold from train rides only.
- **Pre-fall comparison**: one row per held-out fall group: pre-fall danger of the fall ride, danger of each of the
  4 comparison rides at the same elapsed time, `exceeds_all`, `exceeds_median`; summary: shares and paired AUC.
- **5-class head on golden**: macro-F1 and confusion matrix (as today).
- **Vs elapsed time**: one row per (model, metric, `t_s`), `t_s` in {30, 60, 90, 120, 180, 240, end}; NaN before
  the first window ends.

## Golden ride

A GBG500 ride with `split = golden` (100 rides, frozen, **1 Reckless**). Scores the 5-class head only (and the
read-only quality check of the optimization, R11). The label-free danger score is evaluated on all labelled
rides. Replaying golden rides to the API for a demo is allowed; calibrating a threshold on them is not.

## Fall group

A fall ride + its 4 comparison rides, same city (`fall_group_id`). Split once, seeded, saved in
`data/processed/fall_groups.csv`: `development` (70 %, may be in the encoder training data and used for choices)
or `held_out` (30 %, never trained on nor used to choose). All 5 rides of a group share the same side.

## Drift reference

`artifacts/drift_reference.json`: for each of the 8 window features, its 9 inner decile edges on training
windows; the training predicted-category mix (from out-of-fold predictions); the model version. Written by the
DVC `reference` stage, loaded by the API (histogram buckets) and by the drift job and monitor.

## Drift scenario

| Field | Type | Rule |
|---|---|---|
| `name` | `normal` \| `riskier_mix` \| `unseen_city` \| `unit_bug` | |
| `source` | dataset and split used | never synthetic (R7) |
| `rides` | list of ride ids, with repetitions | seeded resampling; seed and weights in `params.yaml` `drift.scenarios` |
| `has_labels` | bool | true for GBG500-based scenarios (feedback sent) |

## Drift report

See [contracts/drift-report.md](contracts/drift-report.md). One per run; states: `ok` (exit 0), `alert`
(any PSI ≥ 0.1, exit 0, warning), `retrain_recommended` (any PSI ≥ 0.25 or prediction-mix p < 0.01 or
performance below floor, exit 3).

## Retraining trigger

| Field | Type | Rule |
|---|---|---|
| `reason` | `schedule` \| `input_drift` \| `prediction_drift` \| `performance` \| `manual` | input of `train.yml` |
| `evidence` | str | drift report path or alert name and value |
| `triggered_at` | datetime | workflow run time |
| `run_id` | GitHub run + MLflow run | MLflow tags `trigger_reason`, `trigger_evidence` |
| `gate_passed` | bool | from `evaluate` |
| `approver` | GitHub user | from the `production` environment approval in `cd.yml` |

States: `triggered → trained → gate_failed | registered(@staging) → approved → deployed(@production)` or
`rejected`. An upstream-bug alert (unit change) ends at the runbook: `no_retrain` with the fix reference.

## Measurement (AI infra)

One row of a benchmark CSV, schema in [contracts/benchmark-results.md](contracts/benchmark-results.md); `status`
is `measured` or `estimated` (estimates carry `assumptions`).

## Figure / asset

`assets/figures/{id}_{slug}.png` and `.svg`; registry in [contracts/figures.md](contracts/figures.md): `id`
(F01-F46), `slug`, `story`, `core|bonus`, `source` (notebook or procedure), `status` (`planned | draft | final`).
