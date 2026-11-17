"""Fixtures shared by all tests: synthetic rides, so tests never need the real data."""

import numpy as np
import pandas as pd
import pytest

from ride_classifier.config import ID_COLUMN, LABEL_COLUMN, LABELS, SAMPLING_PERIOD_MS, SENSOR_COLUMNS, TIME_COLUMN


def make_ride(n_samples: int = 600, ride_id: str = "ride-test", seed: int = 0) -> pd.DataFrame:
    """One fake ride in the GBG500 format: a reading every 100 ms, gravity on the y axis."""
    rng = np.random.default_rng(seed)
    ride = pd.DataFrame(rng.normal(0.0, 1.0, size=(n_samples, len(SENSOR_COLUMNS))), columns=SENSOR_COLUMNS)
    ride["ay"] -= 9.2  # the sensor's y axis is roughly vertical
    ride.insert(0, TIME_COLUMN, np.arange(n_samples) * SAMPLING_PERIOD_MS)
    ride.insert(0, ID_COLUMN, ride_id)
    return ride


@pytest.fixture
def ride() -> pd.DataFrame:
    """A 60-second fake ride."""
    return make_ride()


@pytest.fixture
def ride_factory():
    """Build custom fake rides: ride_factory(n_samples=3000, ride_id="r1", seed=1)."""
    return make_ride


@pytest.fixture(scope="session")
def labelled_rides() -> tuple[pd.DataFrame, pd.DataFrame]:
    """10 fake 60-second rides per category, each category shaking more than the previous one.

    Returns (readings with a label column, one row per ride: ride_id, label).
    Built once for the whole test session: tests must copy before modifying.
    """
    readings, labels = [], []
    for k, label in enumerate(LABELS):
        for i in range(10):
            ride_id = f"{label}-{i}"
            ride = make_ride(n_samples=600, ride_id=ride_id, seed=100 * k + i)
            ride[SENSOR_COLUMNS] = ride[SENSOR_COLUMNS] * (1 + k)
            readings.append(ride.assign(**{LABEL_COLUMN: label}))
            labels.append((ride_id, label))
    return pd.concat(readings, ignore_index=True), pd.DataFrame(labels, columns=[ID_COLUMN, LABEL_COLUMN])
