import numpy as np
import pandas as pd

from ride_classifier.config import SENSOR_COLUMNS
from ride_classifier.features.windows import ride_windows, windows_by_ride


def test_a_60_second_ride_gives_3_windows_of_30_seconds(ride):
    windows = windows_by_ride(ride, window_s=30, stride_s=15).loc["ride-test"]
    assert windows.shape == (3, len(SENSOR_COLUMNS), 300)  # starts at 0, 15 and 30 s
    assert windows.dtype == np.float32


def test_windows_keep_the_readings_in_time_order_channels_first(ride):
    windows = windows_by_ride(ride, window_s=30, stride_s=15).loc["ride-test"]
    np.testing.assert_allclose(windows[1, 0], ride["ax"].to_numpy()[150:450], rtol=1e-6)  # 2nd window, channel ax


def test_a_ride_shorter_than_a_window_is_padded_with_its_last_reading():
    readings = np.arange(12, dtype=np.float32).reshape(2, 6)  # 2 readings, 6 channels
    windows = ride_windows(readings, window=4, stride=2)
    assert windows.shape == (1, 6, 4)
    np.testing.assert_array_equal(windows[0, :, -1], readings[-1])


def test_one_entry_per_ride_sorted(ride_factory):
    rides = pd.concat([ride_factory(ride_id="b"), ride_factory(ride_id="a")])
    assert list(windows_by_ride(rides, window_s=30, stride_s=15).index) == ["a", "b"]
