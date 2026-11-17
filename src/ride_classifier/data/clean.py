"""Clean raw readings: time order, no duplicate timestamps, no missing sensor values (docs/PLAN.md §4.2)."""

from __future__ import annotations

import logging

import pandas as pd

from ride_classifier.config import ID_COLUMN, SENSOR_COLUMNS, TIME_COLUMN
from ride_classifier.data.load import DataValidationError

logger = logging.getLogger(__name__)


def clean_rides(rides: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: one row per (ride, timestamp), in time order, with every sensor value filled.

    Missing values are filled by linear interpolation inside each ride, never from another ride.
    At the very start or end of a ride (never seen in GBG500) the nearest reading of the same ride is copied.
    The same function cleans the rides sent to the API, so training and serving see the same data.
    """
    out = rides.sort_values([ID_COLUMN, TIME_COLUMN], kind="stable")
    n_before = len(out)
    out = out.drop_duplicates([ID_COLUMN, TIME_COLUMN], keep="first")
    n_gaps = int(out[SENSOR_COLUMNS].isna().any(axis=1).sum())

    out[SENSOR_COLUMNS] = out.groupby(ID_COLUMN, sort=False)[SENSOR_COLUMNS].transform(
        lambda column: column.interpolate(method="linear", limit_direction="both")
    )
    if out[SENSOR_COLUMNS].isna().any().any():
        raise DataValidationError("cleaning: a ride has a sensor channel with no value at all")

    logger.info("cleaning: %d duplicate readings dropped, %d readings with gaps filled", n_before - len(out), n_gaps)
    return out.reset_index(drop=True)
