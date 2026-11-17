"""Project-wide configuration: ride schema, label order, params.yaml and environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", ROOT / "params.yaml"))

# Ride schema (GBG500): one row per sensor reading, sampled at 10 Hz
ID_COLUMN = "ride_id"
TIME_COLUMN = "time_index"  # ms since the start of the ride
SAMPLING_PERIOD_MS = 100
ACCEL_COLUMNS = ["ax", "ay", "az"]  # m/s², gravity included
GYRO_COLUMNS = ["rx", "ry", "rz"]  # deg/s
SENSOR_COLUMNS = ACCEL_COLUMNS + GYRO_COLUMNS
LABEL_COLUMN = "label"

# GBG500_labels.csv uses upper-case column names
RAW_LABEL_COLUMNS = {"RIDE_ID": ID_COLUMN, "LABEL": LABEL_COLUMN}

# Fixed label order: position i of every probability vector is LABELS[i]
LABELS = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]

SAMPLING_HZ = 1000 / SAMPLING_PERIOD_MS

# MLflow (docs/PLAN.md §8): dev server started by `make dev-up`
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ride-classifier")
MODEL_NAME = os.getenv("MODEL_NAME", "ride-classifier")


@lru_cache
def load_params(path: Path = PARAMS_PATH) -> dict:
    """Read params.yaml once: pipeline settings tracked by DVC and logged to MLflow."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
