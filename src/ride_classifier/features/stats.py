"""Per-ride statistics for the baseline model: one row of ~170 numbers per ride (docs/PLAN.md §6.2).

For each sensor channel, plus the overall acceleration and rotation magnitudes:
level and spread (mean, std, percentiles), shape (skewness, kurtosis), jerkiness (changes between consecutive
readings) and frequency content (share of the signal's energy in each frequency band).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

from ride_classifier.config import ACCEL_COLUMNS, GYRO_COLUMNS, ID_COLUMN, SAMPLING_HZ, SENSOR_COLUMNS

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
BANDS_HZ = ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 3.5), (3.5, 5))


def ride_features(rides: pd.DataFrame) -> pd.DataFrame:
    """One row per ride (index = ride_id, sorted), one column per statistic. Expects clean rides (no gaps)."""
    rows = {ride_id: _one_ride(ride) for ride_id, ride in rides.groupby(ID_COLUMN, sort=True)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(ID_COLUMN)


def _one_ride(ride: pd.DataFrame) -> dict[str, float]:
    channels = {column: ride[column].to_numpy(dtype=float) for column in SENSOR_COLUMNS}
    channels["acc_norm"] = np.linalg.norm(ride[ACCEL_COLUMNS].to_numpy(dtype=float), axis=1)
    channels["gyro_norm"] = np.linalg.norm(ride[GYRO_COLUMNS].to_numpy(dtype=float), axis=1)

    features = {"duration_s": len(ride) / SAMPLING_HZ}
    for name, x in channels.items():
        change = np.abs(np.diff(x)) if len(x) > 1 else np.zeros(1)
        q = np.percentile(x, PERCENTILES)
        flat = np.ptp(x) < 1e-9  # skewness/kurtosis are undefined for a constant signal: report 0
        features.update(
            {
                f"{name}_mean": x.mean(),
                f"{name}_std": x.std(),
                **{f"{name}_p{p}": value for p, value in zip(PERCENTILES, q, strict=True)},
                f"{name}_iqr": q[4] - q[2],
                f"{name}_skew": 0.0 if flat else stats.skew(x),
                f"{name}_kurtosis": 0.0 if flat else stats.kurtosis(x),
                f"{name}_jerk_mean": change.mean(),
                f"{name}_jerk_std": change.std(),
                f"{name}_jerk_p99": np.percentile(change, 99),
            }
        )
        freqs, power = signal.welch(x - x.mean(), fs=SAMPLING_HZ, nperseg=min(256, len(x)))
        total = power.sum() or 1.0
        for low, high in BANDS_HZ:
            features[f"{name}_energy_{low}_{high}hz"] = power[(freqs >= low) & (freqs < high)].sum() / total
    return features
