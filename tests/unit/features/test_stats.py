import numpy as np
import pandas as pd

from ride_classifier.config import SENSOR_COLUMNS
from ride_classifier.features.stats import BANDS_HZ, ride_features


def test_one_row_per_ride_with_the_same_columns(ride_factory):
    rides = pd.concat([ride_factory(ride_id="b", n_samples=900), ride_factory(ride_id="a", n_samples=600)])
    features = ride_features(rides)
    assert list(features.index) == ["a", "b"]
    assert features.loc[["a", "b"], "duration_s"].tolist() == [60.0, 90.0]
    assert np.isfinite(features.to_numpy()).all()


def test_a_perfectly_still_sensor_gives_finite_numbers(ride):
    ride[SENSOR_COLUMNS] = 1.0
    assert np.isfinite(ride_features(ride).to_numpy()).all()


def test_frequency_bands_split_the_energy(ride):
    features = ride_features(ride).iloc[0]
    shares = [features[f"ax_energy_{low}_{high}hz"] for low, high in BANDS_HZ]
    assert 0.9 < sum(shares) <= 1.0 + 1e-9
