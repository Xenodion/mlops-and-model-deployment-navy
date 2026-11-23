"""Fixed-length windows of a ride: the CNN's input (docs/PLAN.md §6.2).

A 30 s window every 15 s turns one ride of ~3,000 readings into ~19 overlapping windows of 300 readings x 6 channels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ride_classifier.config import ID_COLUMN, SAMPLING_HZ, SENSOR_COLUMNS


def ride_windows(readings: np.ndarray, window: int, stride: int) -> np.ndarray:
    """(n_readings, channels) -> (n_windows, channels, window), channels first as PyTorch expects.

    A ride shorter than one window is padded by repeating its last reading.
    """
    if len(readings) < window:
        readings = np.pad(readings, ((0, window - len(readings)), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(readings, window, axis=0)[::stride]
    return np.ascontiguousarray(windows, dtype=np.float32)


def windows_by_ride(rides: pd.DataFrame, window_s: float, stride_s: float) -> pd.Series:
    """One entry per ride (index = ride_id, sorted): its windows. Expects clean rides (no gaps)."""
    window, stride = round(window_s * SAMPLING_HZ), round(stride_s * SAMPLING_HZ)
    windows = {
        ride_id: ride_windows(ride[SENSOR_COLUMNS].to_numpy(dtype=np.float32), window, stride)
        for ride_id, ride in rides.groupby(ID_COLUMN, sort=True)
    }
    return pd.Series(windows, name="windows", dtype=object).rename_axis(ID_COLUMN)
