# ADR 004 - Pre-fall danger: 60 s segment, 2 s margin, same elapsed time on comparison rides

**Status:** accepted, 21/11/2026.

## Context

SRV3490 has 698 rides with a confirmed fall, each with 4 comparison rides from the same city. Without any label we
can ask: "before a fall, is the rider in a dangerous state?", by comparing the danger score just before the fall
with the same moment of the comparison rides. Three parameters must be fixed: how long before the fall, how much to
cut just before it, and how to score that span.

## Decision

- **Segment:** the 60 s ending 2 s before the first `topple` reading.
- **Score:** mean danger of the windows ending inside the segment (not the cumulative score since the start of
  the ride, which would dilute 2 to 17 minutes of riding into the measure).
- **Comparison:** the same elapsed span on each comparison ride; a comparison ride too short for it gives no pair
  (334 of 2,792); a group with no valid pair is dropped; results are also reported on complete groups only.
- **Split:** fall groups split once, 70 % development / 30 % held-out, stratified by city, a group never split
  (`data/processed/fall_groups.csv`).

## Why these values

- 60 s is the smallest segment that gives every window candidate of ADR 002 (up to 60 s windows, stride 30 s) at
  least two window ends, so the candidates are compared on the same span.
- 2 s is the spec's minimum warning time and keeps the impact itself out: otherwise the model would detect the
  crash, not the riding before it.
- Same elapsed time, same city: controls for the city and for the moment of the ride (warm-up, fatigue).

## Alternatives rejected

- Comparison at the same relative position (e.g. 80 % of the ride): breaks the elapsed-time control.
- Choosing the segment length on the results: 30 s and 120 s segments and a 5 s margin are reported as a
  sensitivity check in the fall analysis (issue #29), never used to choose.

## Consequences

With 60 s windows the windows ending in the segment start up to 60 s earlier: "the state before the fall" covers
up to 2 minutes for that candidate (stated, not corrected). A fall can follow a pothole, so the measure is an
association with danger, not a proof of recklessness.

## Evidence

`params.yaml` `falls`, `src/ride_classifier/pipeline/falls.py`, `window_study.segment_danger`, commits e4a02b8,
8248e55 and 648fe25, issue #29.
