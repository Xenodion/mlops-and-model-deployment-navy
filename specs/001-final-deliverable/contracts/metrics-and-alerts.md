# Contract: Prometheus metrics, alerts and runbooks

Sources: the API `/metrics` (layers 1-4) and the drift monitor `/metrics` (`drift_*`). Thresholds live in
`params.yaml` `monitoring` and are calibrated on the **normal** scenario (training rides), never on golden rides.
Config files: `monitoring/prometheus/{prometheus.yml,alerts.yml}`, mounted as-is in `k8s/monitoring/`.

## Metrics

| Layer | Metric | Type | Labels |
|---|---|---|---|
| 1 infra | `api_requests_total` | counter | `endpoint`, `status` (exists) |
| 1 infra | `api_request_duration_seconds` | histogram | `endpoint` (exists; 0.2 s bucket = budget) |
| 2 input | `input_window_feature` | histogram, buckets = reference deciles | `feature` (8 window features) |
| 2 input | `drift_psi` | gauge (drift monitor) | `feature` |
| 3 prediction | `predictions_total` | counter | `label`, `endpoint` (exists, + endpoint) |
| 3 prediction | `prediction_confidence` | histogram | `endpoint` |
| 3 prediction | `fall_risk` | histogram (model 2 risk at each live window) | `endpoint` |
| 3 prediction | `fall_alerts_total` | counter (live windows with a fall alert) | `endpoint` |
| 3 prediction | `drift_prediction_chi2_pvalue` | gauge (drift monitor) | |
| 4 outcome | `feedback_total` | counter | `true_label`, `correct` |
| all | `model_info` | gauge | `version` (exists) |
| policy | `drift_retrain_recommended` | gauge 0/1 (drift monitor) | `reason` |

## Alerts (one per layer at least, each with `runbook` annotation)

| Alert | Expression (sketch) | For | Layer | Runbook |
|---|---|---|---|---|
| `LatencyP95High` | p95 of `/v1/predict/window` > 0.2 s | 1m | infra | `#latency` |
| `ErrorRateHigh` | 5xx share > 5 % | 1m | infra | `#errors` |
| `InputDriftHigh` | `drift_retrain_recommended{reason="input_drift"} == 1` | 2m | input | `#input-drift` |
| `PredictionMixShift` | `drift_prediction_chi2_pvalue < 0.01` | 2m | prediction | `#prediction-drift` |
| `AccuracyDrop` | correct share of feedback over 15 m < floor, with ≥ 20 labels | 5m | outcome | `#performance` |

Warning level (dashboard colour, no alert): PSI ≥ 0.1.

Live input alerts are disabled by default (`drift.live_alerts_enabled: false`): PSI remains visible, but the
recommendation gauge is omitted and `InputDriftHigh` cannot fire. `drift_monitor_alerts_enabled` exposes this
configuration. Enable only after validating ride sampling and thresholds; a 20-window minimum alone produced
large PSI on a normal two-ride smoke test. Offline flags remain provisional and require investigation.

## Grafana dashboard (`monitoring/grafana/dashboards/ride-api.json`)

One row per layer, light theme, category colours from `viz/palette.py`: (1) req/s, p95 vs 200 ms line, 5xx share;
(2) PSI per feature with 0.1 / 0.25 lines, rotation-std distribution; (3) category mix stacked, confidence, fall-risk distribution and alert rate;
(4) rolling accuracy, labels received; plus an alert list panel and the model version.

## Runbooks (`docs/runbooks.md`)

Each anchor: symptom, first checks, decision. The input-drift entry holds the fork (F43): if rotation features
moved by a constant factor near 57 → **upstream unit bug**: fix the sender, no retraining; if several features
moved without a unit pattern → **new population**: dispatch `train.yml` with `reason=input_drift`, gate, approve.
