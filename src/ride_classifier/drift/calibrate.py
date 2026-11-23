"""Measure normal-scenario PSI variability by resampling whole training rides.

This uses the same training population as the reference: it estimates within-population resampling variability,
not performance on independent rides or production false-alarm rates. It never changes configured thresholds.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from ride_classifier.config import ID_COLUMN, MLFLOW_TRACKING_URI, ROOT, SAMPLING_HZ, load_params
from ride_classifier.data.clean import clean_rides
from ride_classifier.drift.psi import psi
from ride_classifier.features.window_stats import FEATURES, window_stats
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.tracking import check_server, run_tags


def training_features(readings: pd.DataFrame, window_s: float, stride_s: float) -> dict[str, pd.DataFrame]:
    """Calculate each training window once, using the reference stage's cleaning and windowing."""
    size, step = round(window_s * SAMPLING_HZ), round(stride_s * SAMPLING_HZ)
    if size <= 0 or step <= 0:
        raise ValueError("Window and stride must be positive")
    rides = {}
    for ride_id, ride in clean_rides(readings).groupby(ID_COLUMN, sort=True):
        rows = [window_stats(ride.iloc[start : start + size]) for start in range(0, len(ride) - size + 1, step)]
        if rows:
            rides[ride_id] = pd.DataFrame(rows)
    if not rides:
        raise ValueError("No complete training windows")
    return rides


def sample_psi(rides: dict[str, pd.DataFrame], reference: dict, n_rides: int, seeds: list[int]) -> pd.DataFrame:
    """Resample at ride level; repetitions retain their full contribution to window PSI."""
    if not rides or n_rides < 1 or not seeds:
        raise ValueError("Rides, sample size and seeds must be non-empty")
    ids = sorted(rides)
    rows = []
    for seed in seeds:
        selected = np.random.default_rng(seed).choice(ids, size=n_rides, replace=True)
        windows = pd.concat([rides[ride_id] for ride_id in selected], ignore_index=True)
        result = {"seed": seed, "n_rides": n_rides, "n_windows": len(windows)}
        for feature in FEATURES:
            ref = reference["features"][feature]
            result[feature] = psi(ref["shares"], windows[feature].to_numpy(), ref["edges"])
        result["max_psi"] = max(result[feature] for feature in FEATURES)
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--rides", type=int, nargs="+", default=[1, 5, 20, 100])
    parser.add_argument("--out", type=Path, default=Path("reports/drift/calibration"))
    args = parser.parse_args()
    if args.samples < 1 or any(n < 1 for n in args.rides) or len(set(args.rides)) != len(args.rides):
        parser.error("samples and distinct ride counts must be positive")
    check_server(MLFLOW_TRACKING_URI)
    params = load_params()
    reference_path = ROOT / "artifacts/drift_reference.json"
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    cnn = params["train"]["cnn"]
    reference_sha256 = hashlib.sha256(reference_bytes).hexdigest()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("drift-calibration")
    with mlflow.start_run(run_name="normal-ride-resampling"):
        mlflow.set_tags(
            {
                **run_tags({"gbg500_md5": ROOT / "data/raw/GBG500.parquet.dvc"}),
                "reference_sha256": reference_sha256,
                "population": "GBG500 training; shared with reference",
            }
        )
        mlflow.log_params(
            {
                "samples": args.samples,
                "first_seed": args.first_seed,
                "ride_counts": ",".join(map(str, args.rides)),
                "window_s": cnn["window_s"],
                "stride_s": cnn["stride_s"],
            }
        )
        training = pd.read_parquet(ROOT / params["data"]["train_path"])
        rides = training_features(training, cnn["window_s"], cnn["stride_s"])
        seeds = list(range(args.first_seed, args.first_seed + args.samples))
        result = pd.concat([sample_psi(rides, reference, count, seeds) for count in args.rides], ignore_index=True)
        args.out.mkdir(parents=True, exist_ok=True)
        raw_path = args.out / "normal_psi_samples.csv"
        result.to_csv(raw_path, index=False)
        summary = {
            "samples": args.samples,
            "first_seed": args.first_seed,
            "ride_counts": args.rides,
            "reference_sha256": reference_sha256,
            "by_ride_count": {},
        }
        for count, group in result.groupby("n_rides", sort=True):
            summary["by_ride_count"][str(count)] = {
                "max_psi_p95": float(group["max_psi"].quantile(0.95)),
                "max_psi_p99": float(group["max_psi"].quantile(0.99)),
                "max_psi_max": float(group["max_psi"].max()),
                "existing_alert_fraction": float((group["max_psi"] >= params["drift"]["psi_alert"]).mean()),
                "existing_retrain_fraction": float((group["max_psi"] >= params["drift"]["psi_retrain"]).mean()),
            }
        summary_path = args.out / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics(
            {
                f"rides_{count}_{key}": value
                for count, values in summary["by_ride_count"].items()
                for key, value in values.items()
            }
        )
        mlflow.log_artifact(str(raw_path))
        mlflow.log_artifact(str(summary_path))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
