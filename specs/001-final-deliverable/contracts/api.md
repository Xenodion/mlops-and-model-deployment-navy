# Contract: ride API (FastAPI, Swagger at `/docs`)

Existing routes keep their behaviour: `GET /health`, `GET /metrics`, `POST /v1/predict/file`. Changes are
additive (no field removed), so the web app and the smoke test keep working. Model (research R1): a danger score
learned without labels + a 5-class head. When the served model is E3 (fallback), the danger fields are `null`.

## `POST /v1/predict` - whole ride, with its timeline

Request: `Ride` (unchanged: columnar `time_index`, `ax..rz`, 300 to 20,000 readings).

Response: `Prediction` (unchanged fields: `label` and `probabilities` from the 5-class head, `reckless_score` =
P(Reckless)) **plus**

```json
"danger_score": 1.8,
"is_dangerous": true,
"danger_threshold": 1.5,
"timeline": {
  "window_s": 30, "stride_s": 15,
  "windows": [ {"end_s": 30.0, "danger": 2.4, "probabilities": {"Safe": 0.7, "...": 0.0}} ],
  "cumulative": [ {"end_s": 30.0, "danger": 2.4, "probabilities": {"Safe": 0.7, "...": 0.0}} ],
  "note": "per-window values are hints: only whole rides are labelled"
}
```

- `danger_score`: whole-ride cumulative danger (reconstruction error z-scored on train windows), learned without
  labels; `is_dangerous` = above `danger_threshold` (chosen on train rides, also in `/health`).
- Rule: the last `cumulative` entry equals `danger_score` and `probabilities`.

## `POST /v1/predict/window` - one live window (stateless)

Request:

```json
{
  "ride_id": "r-123",
  "window_index": 4,
  "readings": { "time_index": [...300], "ax": [...], "ay": [...], "az": [...], "rx": [...], "ry": [...], "rz": [...] },
  "state": { "sum_proba": {"Safe": 2.9, "Safe overall": 0.4, "Bad weather": 0.3, "Tandem": 0.2, "Reckless": 0.2}, "sum_danger": 3.1, "n_windows": 4 }
}
```

- `readings`: exactly `window_s × 10` readings, same validation as `Ride`. `window_s` comes from the served
  model's metadata (provisional 30 s → 300 readings until the window study R15 decides).
- `state`: optional; absent = first window (zeros). `n_windows` must equal `window_index`.

Response:

```json
{
  "ride_id": "r-123", "window_index": 4, "end_s": 90.0,
  "window_probabilities": {"...": 0.0},
  "cumulative_probabilities": {"...": 0.0},
  "window_danger": 2.4, "cumulative_danger": 1.1, "is_dangerous": false,
  "label": "Safe", "confidence": 0.62,
  "state": { "sum_proba": {"...": 0.0}, "sum_danger": 5.5, "n_windows": 5 },
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

Unchanged fields, plus `drift_reference_loaded: bool` (layer-2 histograms need it) and `danger_threshold: float |
null`.
