# E4 - Window length study (pre-registration)

Written before any run. Rule source: `specs/001-final-deliverable/research.md` R15 (model: R1, evaluation: R16).
Tracked by GitHub issue #11. Results go in `artifacts/window_study.csv`, figure F47.

Revision 21/11, still before any run: the problem was reframed after the course `data.txt` (labels for evaluation
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

- (a) **Main signal**: paired AUC of the cumulative danger score in the pre-fall segment (last 30-60 s ending 2 s
  before the first `topple`) vs the same elapsed time on the 4 comparison rides, on the development fall groups
  (~490 groups), with a bootstrap 95 % CI over groups.
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

## Results

To be filled after the run.
