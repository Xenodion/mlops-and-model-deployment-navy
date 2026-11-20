import pandas as pd

from ride_classifier.experiments.window_study import choose_window


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["window_s", "f1_mean", "f1_std", "f1_at_60"])


def test_choose_window_picks_best_f1_at_60_among_windows_within_one_std():
    results = _frame([(5, 0.30, 0.05, 0.25), (10, 0.34, 0.05, 0.33), (30, 0.36, 0.05, 0.28)])
    assert choose_window(results) == 10


def test_choose_window_prefers_shorter_window_on_tie():
    results = _frame([(30, 0.36, 0.05, 0.30), (10, 0.35, 0.05, 0.30)])
    assert choose_window(results) == 10


def test_choose_window_excludes_window_outside_one_std_even_if_best_at_60():
    results = _frame([(5, 0.20, 0.03, 0.50), (30, 0.36, 0.03, 0.30)])
    assert choose_window(results) == 30


def test_choose_window_uses_std_of_the_best_row_not_of_the_candidate():
    results = _frame([(5, 0.34, 0.20, 0.40), (30, 0.36, 0.01, 0.30)])  # threshold 0.35 excludes 5
    assert choose_window(results) == 30


def test_choose_window_single_candidate_returns_it():
    assert choose_window(_frame([(30, 0.36, 0.05, 0.3)])) == 30


def test_choose_window_long_window_best_overall_but_shorter_wins_at_60():
    results = _frame([(60, 0.40, 0.05, 0.31), (15, 0.37, 0.06, 0.38), (5, 0.30, 0.05, 0.45)])
    assert choose_window(results) == 15
