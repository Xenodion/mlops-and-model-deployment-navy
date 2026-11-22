# Contract: drift job

```text
uv run python -m ride_classifier.drift.job \
    --reference artifacts/drift_reference.json \
    --current <prediction log .jsonl | scenario windows .parquet> \
    --out reports/drift/<run>/ [--evidently]
```

Exit codes: `0` ok or alert (warning only), `3` retrain recommended, `1` error (bad input). CI reads the exit code
(`drift.yml`), never parses text.

Outputs in `--out`:

- `drift_report.json`:

```json
{
  "model_version": "2", "reference": "artifacts/drift_reference.json", "n_windows": 1234,
  "features": { "rot_norm_std": {"psi": 3.2, "ks_pvalue": 0.0, "level": "retrain"} },
  "prediction_mix": { "chi2_pvalue": 0.003, "current": {"Safe": 0.6, "...": 0.0}, "reference": {"...": 0.0} },
  "performance": { "n_labels": 80, "macro_f1": 0.41, "floor": 0.30 } ,
  "status": "retrain_recommended",
  "reasons": ["input_drift", "prediction_drift"]
}
```

- `drift_report.md`: the same as a table, appended to the GitHub job summary.
- `evidently.html` (optional).

Thresholds (`params.yaml` `drift`): `psi_alert: 0.1`, `psi_retrain: 0.25`, `chi2_pvalue: 0.01`,
`min_labels: 20`. `psi()` uses the reference deciles (10 bins) with a small epsilon for empty bins; the same function
serves `drift/monitor.py`.

Current implementation: JSONL or parquet holds one row per window with the eight `*_std` feature columns, or
sensor rows with `window_id`, `time_index` and the six sensor columns. Optional `predicted_label` and `true_label`
columns provide prediction and outcome checks; on raw sensor rows each must be constant within a window. The
prediction reference is the out-of-fold **ride** mix, so the current prediction sample must use a comparable unit;
repeated live windows from the same ride are not an independent ride sample. Chi-square p-values are approximate
with small expected counts. KS is reported as null until individual reference values are retained; a histogram
alone does not support an exact two-sample KS test. Thresholds have not yet been calibrated on the normal scenario.
The API prediction log contains these features on `window_prediction` events; other events are ignored. The
scenario exporter still needs to produce this format. Optional Evidently output is not implemented.

The live exporter queries Prometheus histogram increases over a rolling 15-minute window. It exports PSI only
for complete reference-compatible buckets with at least 20 windows; `drift_monitor_up` is zero if any feature is
unavailable, and the retraining gauge is omitted until all features are available. It does not calculate the
prediction-mix p-value: the current API counter is per window, not an independent per-ride sample.
