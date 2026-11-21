# Contract: ride API (FastAPI, Swagger at `/docs`)

Existing routes keep their behaviour: `GET /health`, `GET /metrics`, `POST /v1/predict/file`. Changes are
additive (no field removed), so the web app and the smoke test keep working. One image serves two models (ADR 010): model 1
(ride type, cumulative, E3) and model 2 (fall risk, 10 s windows). If model 2 is not loaded, its fields are `null`
and model 1 still answers.

## `POST /v1/predict` - whole ride, with its timeline

Request: `Ride` (unchanged: columnar `time_index`, `ax..rz`, 300 to 20,000 readings).

Response: `Prediction` (unchanged fields: `label`, `probabilities`, `reckless_score` from model 1) **plus**

```json
"timeline": {
  "window_s": 30, "stride_s": 15,
  "windows": [ {"end_s": 30.0, "probabilities": {"Safe": 0.7, "...": 0.0}} ],
  "cumulative": [ {"end_s": 30.0, "probabilities": {"Safe": 0.7, "...": 0.0}} ],
  "note": "per-window probabilities are hints: only whole rides are labelled"
},
"fall_risk": {
  "window_s": 10, "stride_s": 5,
  "windows": [ {"end_s": 10.0, "risk": 0.12, "alert": false} ],
  "threshold": 0.6,
  "retrospective": true,
  "note": "computed on a recorded ride: for analysis, not a forecast"
}
```

- Rule: the last `cumulative` entry of `timeline` equals `probabilities`.

## `POST /v1/predict/window` - one live window (stateless)

Request:

```json
{
  "ride_id": "r-123",
  "window_index": 4,
  "readings": { "time_index": [...300], "ax": [...], "ay": [...], "az": [...], "rx": [...], "ry": [...], "rz": [...] },
  "state": { "sum_proba": {"Safe": 2.9, "Safe overall": 0.4, "Bad weather": 0.3, "Tandem": 0.2, "Reckless": 0.2}, "n_windows": 4 }
}
```

- `readings`: the last 30 s (300 readings), same validation as `Ride`: model 1 scores the whole 30 s window, model
  2 the last 10 s of it (ADR 002).
- `state`: optional; absent = first window (zeros). `n_windows` must equal `window_index`.

Response:

```json
{
  "ride_id": "r-123", "window_index": 4, "end_s": 90.0,
  "window_probabilities": {"...": 0.0},
  "cumulative_probabilities": {"...": 0.0},
  "fall_risk": 0.12, "fall_alert": false,
  "label": "Safe", "confidence": 0.62,
  "state": { "sum_proba": {"...": 0.0}, "n_windows": 5 },
  "model_version": "2"
}
```

Errors: 422 malformed (wrong length, out-of-range values, inconsistent state); 503 no model loaded.
Latency budget: 200 ms p95 (SC-004).

Edge case recorded: a ride shorter than one window is rejected with 422, as today: a window cannot be scored on
less signal than it was trained on. The minimum follows the window length chosen by R15 (provisional 300
readings). The spec edge case ("returned, marked low confidence") is amended
accordingly at `/speckit-analyze`.

## `POST /v1/feedback` - label received later (outcome layer)

Request: `{"ride_id": "r-123", "predicted_label": "Safe", "true_label": "Reckless"}` (both from the 5
categories). Response: `204`. Effect: `feedback_total{true_label, correct}` +1. Simplification stated in the
README: the caller sends the predicted label (no server-side join, R6).

## `GET /health`

Unchanged fields, plus `drift_reference_loaded: bool` (layer-2 histograms need it), `fall_risk_loaded: bool` and
`fall_risk_threshold: float | null`.
