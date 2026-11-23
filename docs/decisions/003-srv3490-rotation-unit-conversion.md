# ADR 003 - SRV3490 rotation is converted with 180/pi

**Status:** accepted, 21/11/2026.

## Context

GBG500 rotation is in deg/s (the API contract says so). SRV3490 rotation looked about 57 times smaller, the
signature of rad/s. We measured it before converting anything (notebook `02_units_and_cities`): the ratio of
per-ride standard deviations GBG500 / SRV3490 is about 60 on the three rotation channels (95 % CI 58-62), and
0.77-0.97 on the three acceleration channels, which share the same unit in both datasets.

## Decision

Convert SRV3490 rotation with the exact unit factor 180/pi = 57.2958, in one place
(`data.srv3490.load_clean_srv3490`, stage `prepare_srv`), never inside the model.

## Alternatives rejected

- The measured ratio (~60): a unit change is a physical constant, not a parameter to fit. The ~5 % gap is of the
  same kind as the 3-23 % differences seen on accelerations: the two datasets hold different riders, cities and
  devices.
- Standardising each dataset separately: it would hide the unit problem that the monitoring story needs.

## Consequences

- The conversion is a fix of this data source. A sender that skips it is the "sensor unit bug" drift scenario:
  the model's own preprocessing is unchanged, the input distribution moves, the drift detector must catch it, and
  the runbook says "fix upstream, do not retrain" (ADR 008).
- A CI test reproduces this with tiny in-memory windows (task T059).

## Evidence

`notebooks/02_units_and_cities.ipynb` (F04), `params.yaml` `data.srv3490.rotation_factor`, commits 0291843 and
d816e61, issue #12.
