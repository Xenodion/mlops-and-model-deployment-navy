import numpy as np
import pandas as pd
import pytest

from ride_classifier.config import SENSOR_COLUMNS
from ride_classifier.data.clean import clean_rides


def test_rows_are_sorted_by_ride_then_time(ride_factory):
    shuffled = pd.concat([ride_factory(ride_id="b"), ride_factory(ride_id="a")]).sample(frac=1, random_state=0)
    clean = clean_rides(shuffled)
    assert clean["ride_id"].is_monotonic_increasing
    assert all(ride["time_index"].is_monotonic_increasing for _, ride in clean.groupby("ride_id"))


def test_duplicate_timestamp_keeps_the_first_reading(ride):
    duplicate = ride.iloc[[5]].assign(ax=999.0)
    clean = clean_rides(pd.concat([ride, duplicate]))
    assert len(clean) == len(ride)
    assert clean.loc[5, "ax"] == ride.loc[5, "ax"]


def test_gap_is_filled_by_linear_interpolation(ride):
    ride.loc[10:12, SENSOR_COLUMNS] = np.nan  # readings 10, 11, 12 missing
    clean = clean_rides(ride)
    assert not clean[SENSOR_COLUMNS].isna().any().any()
    halfway = (ride.loc[9, "ax"] + ride.loc[13, "ax"]) / 2  # reading 11 sits halfway between 9 and 13
    assert clean.loc[11, "ax"] == pytest.approx(halfway)


def test_gap_is_never_filled_from_another_ride(ride_factory):
    a, b = ride_factory(ride_id="a", seed=1), ride_factory(ride_id="b", seed=2)
    a.loc[a.index[-1], SENSOR_COLUMNS] = np.nan  # last reading of ride a is missing
    clean = clean_rides(pd.concat([a, b], ignore_index=True))
    last_of_a = clean[clean["ride_id"] == "a"].iloc[-1]
    assert last_of_a["ax"] == pytest.approx(a["ax"].iloc[-2])  # copied from ride a itself, not from ride b
