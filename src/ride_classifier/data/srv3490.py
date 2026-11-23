"""SRV3490: crash and baseline rides from eight cities, with per-ride tables and unit harmonisation."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ride_classifier.config import GYRO_COLUMNS, ID_COLUMN, SENSOR_COLUMNS, TIME_COLUMN
from ride_classifier.data.clean import clean_rides

logger = logging.getLogger(__name__)

CRASH_REF_COLUMN = "crash_ride_id"
TOPPLE_COLUMN = "topple"


def load_srv3490(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read the SRV3490 readings, optionally only some columns (the file has 17 million rows)."""
    readings = pd.read_parquet(path, columns=columns)
    logger.info("SRV3490: %d readings loaded from %s", len(readings), path)
    return readings


def ride_table(readings: pd.DataFrame) -> pd.DataFrame:
    """One row per ride: city, n_readings, duration_s, fall_time_index, fall_group_id.

    Needs ride_id, crash_ride_id, city, time_index and topple. A crash ride is its own fall group;
    a baseline ride belongs to the group of the crash ride it points to.
    """
    grouped = readings.groupby(ID_COLUMN, sort=True)
    table = grouped.agg(
        city=("city", "first"),
        n_readings=(TIME_COLUMN, "size"),
        t_min=(TIME_COLUMN, "min"),
        t_max=(TIME_COLUMN, "max"),
        crash_ref=(CRASH_REF_COLUMN, "first"),
    )
    table["duration_s"] = (table["t_max"] - table["t_min"]) / 1000.0
    fall_rows = readings.loc[readings[TOPPLE_COLUMN] == 1]
    first_fall = fall_rows.groupby(ID_COLUMN)[TIME_COLUMN].min()
    table["fall_time_index"] = first_fall.reindex(table.index).astype("Int64")
    table["fall_group_id"] = table["crash_ref"].fillna(table.index.to_series())
    return table.drop(columns=["t_min", "t_max", "crash_ref"]).reset_index()


def check_no_overlap(srv_ids: Iterable[str], gbg_ids: Iterable[str]) -> None:
    """Raise if a ride id appears in both datasets."""
    common = set(srv_ids) & set(gbg_ids)
    if common:
        raise ValueError(f"{len(common)} ride ids appear in both SRV3490 and GBG500, e.g. {sorted(common)[0]}")


def harmonise_rotation(readings: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Return a copy with rx, ry, rz multiplied by factor (rad/s to deg/s); accelerations untouched."""
    out = readings.copy()
    for column in GYRO_COLUMNS:
        out[column] = out[column] * factor
    return out


def load_clean_srv3490(path: Path, rotation_factor: float, ride_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """SRV3490 sensor readings ready for the model: rotation converted to deg/s, cleaned, float32.

    The one entry point for SRV3490 sensor data (the dataset has ~1.6 % missing sensor values): conversion is a
    fix of this source, applied here; cleaning is the same `clean_rides` the model applies to every ride it serves.
    """
    filters = [(ID_COLUMN, "in", list(ride_ids))] if ride_ids is not None else None
    readings = pd.read_parquet(path, columns=[ID_COLUMN, TIME_COLUMN, *SENSOR_COLUMNS], filters=filters)
    readings = clean_rides(harmonise_rotation(readings, rotation_factor))
    readings[SENSOR_COLUMNS] = readings[SENSOR_COLUMNS].astype("float32")
    logger.info("SRV3490: %d clean readings, %d rides", len(readings), readings[ID_COLUMN].nunique())
    return readings
