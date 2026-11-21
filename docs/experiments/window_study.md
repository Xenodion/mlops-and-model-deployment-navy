# E4 - Window length study (closed 21/11)

**Closed.** This study was set up for the superseded single "danger score" design. Its runs are kept as the
measurement behind ADR 002 (30 s for model 1, 10 s for model 2); no further search is planned (ADR 001: model
quality is not graded). Original pre-registration below, unchanged.

## Results (measured, laptop CPU, 10 epochs, 2 s margin, development fall groups; MLflow `E4-window-*`)

| Window | Pre-fall vs comparison AUC (95 % CI) | Reckless vs counterfactual AUC | Training time | Peak memory |
|---|---|---|---|---|
| 5 s | 0.957 (0.945-0.968) | 0.83 | 21 min | 3.9 GB |
| 10 s | 0.878 (0.856-0.896) | 0.83 | 16 min | 3.6 GB |
| 15 s | 0.835 (0.813-0.858) | 0.82 | 14 min | 3.6 GB |
| 30 s | 0.752 (0.725-0.781) | 0.84 | 13 min | 6.7 GB (before the memory fix) |
| 60 s | 0.717 (0.689-0.746) | 0.82 | 11 min | 3.6 GB |

Reading: shorter windows separate pre-fall riding better; the 5 s value is suspected to include the start of the
fall (flat Reckless separation, last window ending 2 s before `topple`), hence 10 s with a 5 s anti-leakage gap for
model 2 (ADR 004).

---

# Original pre-registration

Written before any run. Rule source: `specs/001-final-deliverable/research.md` R15 (model: R1, evaluation: R16).
Tracked by GitHub issue #11. Results go in `artifacts/window_study.csv`, figure F47.

Revisions 21/11, still before any full run (only 1-epoch smoke runs on a small subset): the problem was reframed after the course `data.txt` (labels for evaluation
only; Reckless = the class to detect; Safe overall, Bad weather and Tandem = counterfactuals that must not be
flagged). The study now compares window lengths for the label-free danger detector, not for a supervised
5-class classifier. The first version (supervised, macro-F1 rule) was never run.

## Question

Which window length should the danger detector use: does a shorter window warn earlier without losing the
separation between dangerous and normal riding, and without flagging the counterfactual categories?

## Why

The data has one reading every 100 ms (10 Hz). The 30 s window (stride 15 s) is E3's modelling choice and was
never compared with other lengths (`docs/PLAN.md` section 6.5, "Not tried"). The minimum ride length, the payload
of `/v1/predict/window`, the update rate of the live score and the load-test request rate all follow from it, so
none is accepted until this test is done.

## Hypothesis

Shorter windows answer earlier but carry less context; below some length the danger score no longer separates
pre-fall riding from comparison riding, nor Reckless rides from the counterfactual categories.

## Candidates

Window 5, 10, 15, 30, 60 s (50, 100, 150, 300, 600 readings); stride = window / 2.

## Fixed conditions

- The model of R1: a convolutional autoencoder (encoder = E3's three convolution blocks), trained without labels;
  danger score = reconstruction error standardised on train windows. Same architecture, epochs, batch size,
  learning rate and seed for every candidate. One change only: window length (and the stride that follows it).
- Training data: windows of all GBG500 rides and all SRV3490 rides **except the held-out fall groups** (30 % of fall
  groups, seeded split, each fall ride with its 4 comparison rides; R16).
- **Golden rides are not used to choose.** Held-out fall groups are not used at all.

## Metrics

- (a) **Main signal**: paired AUC of the pre-fall danger vs the 4 comparison rides, on the development fall groups
  (489 groups), with a bootstrap 95 % CI over groups. Pre-fall danger = mean danger of the windows ending inside
  the pre-fall segment (the 60 s ending 2 s before the first `topple`: impact excluded; `params.yaml` `falls`); the
  comparison rides are scored on the same elapsed span. A comparison ride too short to cover the span gives no
  pair (334 of 2,792 comparison rides are shorter than their fall's segment end); a group with no valid pair is
  dropped; the number of groups kept is reported per candidate, and (a) is also reported on the groups whose 4
  comparison rides all have a score (`auc_fall_full_groups`), to check that dropped pairs do not drive the result.
- Why 60 s and 2 s: 60 s is the smallest segment that gives every candidate (up to 60 s windows, stride 30 s) at
  least two window ends, so the candidates are compared on the same span; 2 s is the spec's minimum warning time
  and keeps the impact out. With 60 s windows the windows ending in the segment start up to 60 s earlier, so
  "the state before the fall" covers up to 2 minutes for that candidate (stated, not corrected). Segments of 30
  and 120 s and a 5 s margin are reported later as a sensitivity check in the fall analysis (#29), never used to
  choose.
- The run is resumable: one CSV row per finished candidate; each row records the epoch time and the process
  memory after training (also an AI-infrastructure sizing measurement).
- SRV3490 readings are cleaned with the pipeline's `clean_rides` (1.6 % missing sensor values) and rotation is
  converted with 180/pi before any use.
- (b) Secondary: ROC-AUC of the whole-ride score, Reckless vs counterfactual categories, on the GBG500 train rides
  (5 Reckless: wide CI, reported).
- (c) False-alarm rate per counterfactual category (train rides) at the threshold = 95th percentile of Safe train
  rides.
- (d) Time to first answer = window length. (e) Per-window CPU latency (median of 200 single-window calls after 20
  warm-up). (f) Training time.

## Time steps after pooling

The encoder is 3 blocks of convolution (kernel 7, padding 3: length unchanged) and `MaxPool1d(2)`. Measured on a
dummy tensor:

| Window | Readings | After pool 1 | After pool 2 | After pool 3 |
| --- | --- | --- | --- | --- |
| 5 s | 50 | 25 | 12 | 6 |
| 10 s | 100 | 50 | 25 | 12 |
| 15 s | 150 | 75 | 37 | 18 |
| 30 s | 300 | 150 | 75 | 37 |
| 60 s | 600 | 300 | 150 | 75 |

The shortest window keeps 6 steps: no adaptation is needed.

## Decision rule

Keep the candidates whose (a) is within the bootstrap CI of the best candidate's (a); among them pick the highest
(b); tie -> the shorter window. All candidates logged in MLflow (`experiment_id` E4-window-*).

## After the study

The chosen length is used for the final model, which is then evaluated once (R16: all labelled rides for the
danger score, golden rides for the 5-class head, held-out fall groups for the pre-fall comparison).

## Threats to validity

- Pre-fall riding is a proxy for danger, not a label: a fall can follow a pothole or wet ground. The metric
  measures association, not recklessness.
- SRV3490 comes from other cities and devices: differences may be partly "other population"; the comparison rides
  are from the same city, which controls for it in (a).
- 5 Reckless train rides: (b) has a wide CI and is only a tie-breaker.
- Shorter windows give more windows per ride, hence more training steps per epoch; epochs are fixed and the number
  of training windows is reported, so the confound can be read, not removed.
- One seed per candidate: seed-to-seed variance is not measured.

