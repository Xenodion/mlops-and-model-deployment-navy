"""Fixtures shared by all tests: synthetic rides, so tests never need the real data."""

import numpy as np
import pandas as pd
import pytest

from ride_classifier.config import ID_COLUMN, SAMPLING_PERIOD_MS, SENSOR_COLUMNS, TIME_COLUMN


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
