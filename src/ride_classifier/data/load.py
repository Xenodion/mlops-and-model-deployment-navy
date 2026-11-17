"""Read the raw GBG500 files and check their shape before anything else touches them."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ride_classifier.config import ID_COLUMN, LABEL_COLUMN, LABELS, RAW_LABEL_COLUMNS, SENSOR_COLUMNS, TIME_COLUMN


class DataValidationError(ValueError):
    """The raw data does not have the expected shape: stop the pipeline with a clear message."""


def load_rides(path: str | Path) -> pd.DataFrame:
    """Sensor readings, one row per reading: ride_id, time_index, ax, ay, az, rx, ry, rz."""
    rides = pd.read_parquet(path)
    validate_rides(rides)
    return rides


def load_labels(path: str | Path) -> pd.DataFrame:
    """One label per ride, with the project's column names (ride_id, label)."""
    labels = pd.read_csv(path).rename(columns=RAW_LABEL_COLUMNS)
    validate_labels(labels)
    return labels


def validate_rides(rides: pd.DataFrame) -> None:
    missing = {ID_COLUMN, TIME_COLUMN, *SENSOR_COLUMNS} - set(rides.columns)
    if missing:
        raise DataValidationError(f"rides: missing columns {sorted(missing)}")
    non_numeric = [c for c in (TIME_COLUMN, *SENSOR_COLUMNS) if not pd.api.types.is_numeric_dtype(rides[c])]
    if non_numeric:
        raise DataValidationError(f"rides: non-numeric columns {non_numeric}")
    if rides[ID_COLUMN].isna().any() or rides[TIME_COLUMN].isna().any():
        raise DataValidationError(f"rides: empty {ID_COLUMN} or {TIME_COLUMN}")


def validate_labels(labels: pd.DataFrame) -> None:
    missing = {ID_COLUMN, LABEL_COLUMN} - set(labels.columns)
    if missing:
        raise DataValidationError(f"labels: missing columns {sorted(missing)}")
    duplicated = labels.loc[labels[ID_COLUMN].duplicated(), ID_COLUMN]
    if not duplicated.empty:
        raise DataValidationError(
            f"labels: {duplicated.nunique()} rides have several labels, e.g. {duplicated.iloc[0]}"
        )
    unknown = sorted(set(labels[LABEL_COLUMN].dropna()) - set(LABELS))
    if unknown or labels[LABEL_COLUMN].isna().any():
        raise DataValidationError(f"labels: empty or unknown labels {unknown}, expected one of {LABELS}")


def check_rides_match_labels(rides: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Every ride must have exactly one label, and every label a ride."""
    ride_ids, label_ids = set(rides[ID_COLUMN].unique()), set(labels[ID_COLUMN])
    without_label, without_ride = ride_ids - label_ids, label_ids - ride_ids
    if without_label or without_ride:
        raise DataValidationError(
            f"rides without a label: {len(without_label)}, labels without a ride: {len(without_ride)}"
        )
