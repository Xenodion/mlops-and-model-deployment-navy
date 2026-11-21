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
