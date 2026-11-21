# ADR 001 - A live danger detector learned without labels, plus a 5-class head

**Status:** accepted, 21/11/2026. Replaces the 20/11 decision "supervised cumulative classifier".

## Context

- The MLOps brief asks for a deep learning model that classifies rides into 5 categories and beats random.
- The teacher (briefing 20/11) offered two options: supervised on the 500 labelled rides (simple, throws most data
  away) or an unsupervised method on all rides, validated with the labels, linked to falls; she preferred the second.
- The course `data.txt`, shipped with the data and read on 21/11, says: GBG500 labels are "used only for
  evaluation"; "Reckless rides are the positive class we aim to detect"; Safe overall, Bad weather and Tandem "are
  counterfactual: they lie near the decision boundary and should not be flagged as anomalies".
- `data/processed/splits.csv`: the frozen golden set holds 1 Reckless ride (6 in total). A Reckless detection rate
  on golden alone means nothing.
- Our 20/11 decision trained on the labels: it was option 1, honestly named but not what the course asks.

## Decision

1. An encoder (convolutional autoencoder on sensor windows) is trained **without labels** on all rides of both
   datasets. Its reconstruction error, standardised on train windows, is the **danger score** of a window; the
   score is cumulated as the ride goes (live).
2. A small **5-class head** (logistic regression on the encoder's mean embedding) is trained on the GBG500 train
   labels, for the brief's 5 categories. It is named as the supervised part.
3. The supervised CNN (E3) stays registered as the reference and the production fallback.
4. Evaluation with labels only: the danger score on all 500 labelled rides (Reckless detected k / 6, false-alarm
   rate per counterfactual category, threshold chosen on train rides), the danger before falls vs comparison rides
   on SRV3490 (ADR 004), the 5-class head on golden rides. Gate: better than random on both.

## Alternatives rejected

- Keep the supervised classifier: meets the brief's wording but contradicts `data.txt` and the briefing.
- Plain clustering (k-means) to recover 5 classes: the teacher ruled it out; 6 Reckless and 18 Tandem rides
  cannot form clusters.
- Train on falls straight away: falls are a noisy proxy of danger (a pothole is not reckless riding); kept as an
  experiment only if the pre-fall analysis shows a signal (issue #28).

## Consequences

- SRV3490's 3,490 unlabelled rides become training data; the falls give a label-free validation.
- The infrastructure (pipeline, API, monitoring, CI/CD, benchmarks) is unchanged; the API gains a danger score.
- The result may be that the score also flags Bad weather or Tandem: that is measured and reported, not hidden.

## Evidence

`docs/PROJECT_FRAMING.md` §3, research R1 and R16 in `specs/001-final-deliverable/research.md`, commit f5bde04,
issues #14 and #29.
