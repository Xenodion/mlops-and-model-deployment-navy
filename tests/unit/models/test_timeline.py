import math

import numpy as np
import pandas as pd
import pytest

from ride_classifier.config import LABELS
from ride_classifier.models.timeline import Timeline, cumulative_at, cumulative_mean, f1_vs_elapsed, window_end_times


def one_hot(*indices: int) -> np.ndarray:
    return np.eye(len(LABELS))[list(indices)]


def make_timeline(ride_id: str, indices: list[int], end_s: list[float]) -> Timeline:
    probas = one_hot(*indices)
    return Timeline(ride_id, np.array(end_s, dtype=float), probas, cumulative_mean(probas))


def test_cumulative_mean_is_running_mean_on_3x5_example():
    probas = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=float)
    expected = np.array([[1, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0], [1 / 3, 1 / 3, 1 / 3, 0, 0]])
    np.testing.assert_allclose(cumulative_mean(probas), expected)


def test_cumulative_mean_last_row_equals_mean_of_all_rows():
    probas = np.random.default_rng(0).random((7, 5))
    np.testing.assert_allclose(cumulative_mean(probas)[-1], probas.mean(axis=0))


def test_cumulative_mean_raises_on_empty_input():
    with pytest.raises(ValueError):
        cumulative_mean(np.empty((0, 5)))


def test_window_end_times_30_15():
    np.testing.assert_array_equal(window_end_times(4, 30, 15), [30, 45, 60, 75])


def test_cumulative_at_uses_only_completed_windows_no_future_leakage():
    timeline = make_timeline("a", [0, 0, 4], [30, 45, 60])
    at_50 = cumulative_at(timeline, 50)  # windows ending at 30 and 45 only
    np.testing.assert_allclose(at_50, one_hot(0)[0])
    assert at_50[4] == 0  # the third window (Reckless) must not leak in
    np.testing.assert_allclose(cumulative_at(timeline, 60), timeline.cumulative[2])


def test_cumulative_at_returns_none_before_any_window_is_complete():
    timeline = make_timeline("a", [1, 0], [30, 45])
    assert cumulative_at(timeline, 10) is None


def test_f1_vs_elapsed_is_nan_before_the_first_window_ends():
    timelines = {"a": make_timeline("a", [0, 0], [30, 45])}
    curve = f1_vs_elapsed(timelines, pd.Series({"a": LABELS[0]}), [10, math.inf])
    assert math.isnan(curve["macro_f1"].iloc[0]) and not math.isnan(curve["macro_f1"].iloc[1])


def test_cumulative_at_returns_last_row_when_time_is_infinite():
    timeline = make_timeline("a", [1, 0], [30, 45])
    np.testing.assert_allclose(cumulative_at(timeline, math.inf), timeline.cumulative[-1])


def test_timeline_label_and_confidence_come_from_last_cumulative_row():
    timeline = make_timeline("a", [1, 1, 0], [30, 45, 60])
    assert timeline.label == LABELS[1]
    assert timeline.confidence == pytest.approx(2 / 3)


def test_f1_vs_elapsed_matches_known_answers_with_a_short_ride():
    # r2 is wrong until its third window; r4 is short (one window, ends at 30 s).
    timelines = {
        "r1": make_timeline("r1", [0, 0, 0], [30, 45, 60]),
        "r2": make_timeline("r2", [0, 1, 1], [30, 45, 60]),
        "r3": make_timeline("r3", [0, 0, 0], [30, 45, 60]),
        "r4": make_timeline("r4", [4], [30]),
    }
    labels = pd.Series({"r1": LABELS[0], "r2": LABELS[1], "r3": LABELS[4], "r4": LABELS[4]})
    out = f1_vs_elapsed(timelines, labels, [30, 60, math.inf])
    assert out["t_label"].tolist() == ["30", "60", "end"]
    assert math.isinf(out["t_s"].iloc[2])
    assert out["n_rides"].tolist() == [4, 4, 4]
    assert out["n_rides_complete"].tolist() == [4, 3, 4]  # r4 is shorter than 60 s but still counted; "end" = all
    # t=30: Safe F1 1/2, Safe overall 0, Reckless 2/3. t=60 and end: Safe 2/3, Safe overall 1, Reckless 2/3.
    np.testing.assert_allclose(out["macro_f1"].tolist(), [(0.5 + 2 / 3) / 5, (4 / 3 + 1) / 5, (4 / 3 + 1) / 5])
