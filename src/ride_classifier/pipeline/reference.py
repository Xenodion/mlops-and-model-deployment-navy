"""Build drift reference distributions from training windows and out-of-fold predictions only.

The reference is also stored with model 1's training run in MLflow: the API images fetch it from the run of the model
they serve, so it always matches the model loaded (docs/PLAN.md INF-5).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow import MlflowClient

from ride_classifier.config import (
    DRIFT_REFERENCE_ARTIFACT,
    ID_COLUMN,
    LABELS,
    MLFLOW_TRACKING_URI,
    ROOT,
    SAMPLING_HZ,
    load_params,
)
from ride_classifier.data.clean import clean_rides
from ride_classifier.features.window_stats import FEATURES, window_stats
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.tracking import check_server


def build_reference(readings: pd.DataFrame, predictions: pd.DataFrame, window_s: float, stride_s: float) -> dict:
    """Use complete windows only; repeated quantiles are collapsed and actual bin shares are retained."""
    size, step = round(window_s * SAMPLING_HZ), round(stride_s * SAMPLING_HZ)
    if size <= 0 or step <= 0:
        raise ValueError("Window and stride must be positive")
    rows = []
    for _, ride in clean_rides(readings).groupby(ID_COLUMN, sort=True):
        rows.extend(window_stats(ride.iloc[start : start + size]) for start in range(0, len(ride) - size + 1, step))
    if not rows:
        raise ValueError("No complete training windows")
    if predictions.empty or not predictions["predicted"].isin(LABELS).all():
        raise ValueError("OOF predictions must contain known categories")
    if predictions[ID_COLUMN].duplicated().any() or set(predictions[ID_COLUMN]) != set(readings[ID_COLUMN]):
        raise ValueError("OOF predictions must contain exactly one prediction per training ride")
    counts = predictions["predicted"].value_counts(normalize=True)
    features = {}
    for name in FEATURES:
        values = np.array([row[name] for row in rows])
        edges = np.unique(np.quantile(values, np.arange(1, 10) / 10))
        shares = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1) / len(values)
        features[name] = {"edges": edges.tolist(), "shares": shares.tolist()}
    return {
        "n_windows": len(rows),
        "features": features,
        "prediction_mix": {label: float(counts.get(label, 0.0)) for label in LABELS},
    }


def log_to_training_run(path: Path, run_id: str, tracking_uri: str = MLFLOW_TRACKING_URI) -> None:
    """Store the reference with the training run of the model it describes (the API fetches it from there)."""
    check_server(tracking_uri)
    MlflowClient(tracking_uri).log_artifact(run_id, str(path))


def main() -> None:
    setup()
    params = load_params()
    readings = pd.read_parquet(ROOT / params["data"]["train_path"])
    predictions = pd.read_csv(ROOT / params["train"]["oof_path"])
    config = params["train"]["cnn"]
    reference = build_reference(readings, predictions, config["window_s"], config["stride_s"])
    run = json.loads((ROOT / params["train"]["run_path"]).read_text(encoding="utf-8"))
    reference["model_version"] = run["model_uri"]
    path = ROOT / "artifacts" / DRIFT_REFERENCE_ARTIFACT
    Path(path).write_text(json.dumps(reference, indent=2), encoding="utf-8")
    log_to_training_run(path, run["run_id"])


if __name__ == "__main__":
    main()
