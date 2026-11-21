"""Eight input drift features: per-window standard deviations of the six channels and two norms."""

import numpy as np
import pandas as pd

from ride_classifier.config import ACCEL_COLUMNS, GYRO_COLUMNS, SENSOR_COLUMNS

FEATURES = tuple(f"{name}_std" for name in SENSOR_COLUMNS) + ("acc_norm_std", "rot_norm_std")


def window_stats(readings: pd.DataFrame) -> dict[str, float]:
    """Summarize one already-cleaned window; rotation must be in the API's deg/s units."""
    if readings.empty:
        raise ValueError("A window must contain at least one reading")
    values = readings[SENSOR_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Window readings must be finite")
    spreads = values.std(axis=0)
    acc = np.linalg.norm(readings[ACCEL_COLUMNS].to_numpy(dtype=float), axis=1).std()
    rot = np.linalg.norm(readings[GYRO_COLUMNS].to_numpy(dtype=float), axis=1).std()
    return dict(zip(FEATURES, [*spreads, acc, rot], strict=True))
