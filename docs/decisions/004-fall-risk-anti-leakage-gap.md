# ADR 004 - Fall risk: 60 s before the fall, 5 s anti-leakage gap, same moments on comparison rides

**Status:** accepted, 21/11/2026 (gap raised from 2 s to 5 s with the two-model decision).

## Context

Model 2 learns "this period is abnormal, a fall may follow". SRV3490 marks the fall with `topple`; the comparison
rides of the same fall group (same city) have no fall. If the training windows reach the fall itself, the model
learns to recognise a crash in progress, which is detection after the fact.

## Decision

- **Positive windows:** windows of a fall ride that end within the 60 s before the gap.
- **Anti-leakage gap: 5 s.** Training and evaluation stop at least 5 s before the first `topple` reading, so the
  start of the fall is never seen. **Sanity check at 10 s:** the result must not collapse when the gap grows; if it
  does, the model was reading the fall.
- **Negative windows:** the same elapsed moments on the 4 comparison rides (controls for city and moment of the
  ride); a comparison ride too short for that moment gives no pair (334 of 2,792).
- **Split:** fall groups split once, 70 % development / 30 % held-out, by city, a group never split
  (`data/processed/fall_groups.csv`).
- **Measure:** share of held-out falls whose pre-fall risk is above their comparison rides (paired AUC).

## Why these values

60 s: long enough to hold several 10 s windows, short enough to describe "the moment before". 5 s: the first
seconds of a fall (loss of balance) are plausibly within that span; 10 s as the check. They are explained choices,
not tuned.

## Consequences

SRV3490 has 20 % falls by design: alert rates measured on it are not real-world rates. A fall can follow a pothole,
so model 2 measures a risk associated with falls, not recklessness.

## Evidence

`params.yaml` `falls`, `src/ride_classifier/pipeline/falls.py`, issue #29.
