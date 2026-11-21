import numpy as np
import pandas as pd
import pytest

from ride_classifier.experiments.window_study import (
    bootstrap_ci,
    choose_window,
    false_alarm_rates,
    group_wins,
    paired_fall_auc,
    safe_threshold,
    segment_danger,
)
from ride_classifier.models.timeline import Timeline


def results(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["window_s", "auc_fall", "auc_fall_ci_low", "auc_fall_ci_high", "auc_reckless"])


def test_choose_window_picks_clear_winner():
    table = results([(5, 0.55, 0.50, 0.60, 0.9), (30, 0.80, 0.75, 0.85, 0.6), (60, 0.60, 0.55, 0.65, 0.9)])
    assert choose_window(table) == 30


def test_choose_window_prefers_auc_reckless_among_candidates_inside_best_ci():
    table = results([(10, 0.78, 0.70, 0.86, 0.9), (30, 0.80, 0.75, 0.85, 0.6)])
    assert choose_window(table) == 10


def test_choose_window_ties_go_to_shorter_window():
    table = results([(30, 0.80, 0.75, 0.85, 0.7), (15, 0.79, 0.74, 0.84, 0.7)])
    assert choose_window(table) == 15


def test_choose_window_excludes_candidate_far_below_best():
    table = results([(5, 0.60, 0.55, 0.65, 0.99), (30, 0.80, 0.75, 0.85, 0.5)])
    assert choose_window(table) == 30


def test_paired_fall_auc_counts_wins_and_ties():
    fall = pd.Series({"g1": 2.0, "g2": 1.0})
    comparison = pd.DataFrame({"a": [1.0, 1.0], "b": [3.0, 0.0]}, index=["g1", "g2"])
    # g1: win, loss; g2: tie, win -> (1 + 0 + 0.5 + 1) / 4
    assert paired_fall_auc(fall, comparison) == pytest.approx(0.625)
    assert group_wins(fall, comparison).to_dict() == {"g1": 0.5, "g2": 0.75}


def test_paired_fall_auc_ignores_nan_pairs():
    fall = pd.Series({"g1": 2.0, "g2": np.nan})
    comparison = pd.DataFrame({"a": [1.0, 1.0]}, index=["g1", "g2"])
    assert paired_fall_auc(fall, comparison) == 1.0


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    values = np.random.default_rng(0).random(200)
    low, high = bootstrap_ci(values, 500, seed=1)
    assert low < values.mean() < high
    assert (low, high) == bootstrap_ci(values, 500, seed=1)


def test_false_alarm_rates_per_category_at_safe_p95_threshold():
    scores = pd.Series([0.0, 1.0, 2.0, 3.0, 10.0, 0.5, 20.0], index=list("abcdefg"))
    labels = pd.Series(["Safe"] * 4 + ["Tandem", "Tandem", "Reckless"], index=scores.index)
    threshold = safe_threshold(scores, labels)
    assert threshold == pytest.approx(2.85)
    rates = false_alarm_rates(scores, labels, threshold)
    assert rates.to_dict() == {"Safe": 0.25, "Tandem": 0.5, "Reckless": 1.0}


def test_segment_danger_averages_only_windows_ending_inside_the_segment():
    timeline = Timeline(
        ride_id="a",
        end_s=np.array([30.0, 45.0, 60.0, 75.0]),
        window_probas=np.zeros((4, 5)),
        cumulative=np.zeros((4, 5)),
        danger=np.array([10.0, 1.0, 3.0, 99.0]),
    )
    assert segment_danger(timeline, 30.0, 60.0) == 2.0  # windows ending at 45 and 60; 75 is after the segment
    assert segment_danger(timeline, 80.0, 90.0) is None
