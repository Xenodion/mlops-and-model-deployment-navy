import pandas as pd
import pytest

from ride_classifier.drift.calibrate import sample_psi
from ride_classifier.drift.psi import psi
from ride_classifier.features.window_stats import FEATURES


def test_resamples_rides_reproducibly_and_uses_all_windows():
    rides = {
        "a": pd.DataFrame({name: [0.0, 0.0] for name in FEATURES}),
        "b": pd.DataFrame({name: [1.0] for name in FEATURES}),
    }
    reference = {"features": {name: {"edges": [0.5], "shares": [0.5, 0.5]} for name in FEATURES}}
    first = sample_psi(rides, reference, 5, [1, 2])
    pd.testing.assert_frame_equal(first, sample_psi(rides, reference, 5, [1, 2]))
    assert first["n_windows"].between(5, 10).all()
    assert (first["max_psi"] == first["rx_std"]).all()
    one = sample_psi({"a": rides["a"]}, reference, 1, [42])
    assert one.loc[0, "rx_std"] == pytest.approx(psi([0.5, 0.5], [0, 0], [0.5]))


def test_requires_nonempty_samples():
    with pytest.raises(ValueError):
        sample_psi({}, {}, 1, [1])
