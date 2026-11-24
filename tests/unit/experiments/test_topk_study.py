"""E5: the aggregations and the pre-registered decision rule (no training here)."""

import numpy as np
import pandas as pd
import pytest

from ride_classifier.config import LABELS
from ride_classifier.experiments.topk_study import aggregate, candidates, decide

SAFE, RECKLESS = LABELS.index("Safe"), LABELS.index("Reckless")


def windows(n_calm: int, n_reckless: int) -> np.ndarray:
    """A ride: calm windows (Safe 0.7) and a few strongly reckless ones (Reckless 0.9)."""
    calm = np.full((n_calm, 5), 0.075)
    calm[:, SAFE] = 0.7
    reckless = np.full((n_reckless, 5), 0.025)
    reckless[:, RECKLESS] = 0.9
    return np.vstack([calm, reckless])


def test_a_short_reckless_episode_is_diluted_by_the_mean_but_not_by_top_k():
    ride = windows(n_calm=16, n_reckless=2)
    assert LABELS[aggregate(ride, "mean").argmax()] == "Safe"
    assert LABELS[aggregate(ride, "top-k", 2).argmax()] == "Reckless"


def test_k_is_capped_by_the_ride_and_a_share_keeps_at_least_one_window():
    ride = windows(n_calm=2, n_reckless=1)
    assert aggregate(ride, "top-k", 8) == pytest.approx(aggregate(ride, "mean"))  # 8 > 3 windows: all of them
    assert aggregate(ride, "top-share", 0.1) == pytest.approx(ride.max(axis=0))  # ceil(0.3) = 1 window


def test_the_reference_and_the_nine_pre_registered_candidates():
    names = [name for name, _, _ in candidates()]
    assert names[0] == "mean" and len(names) == 10 and len(set(names)) == 10


@pytest.mark.parametrize(
    ("best_f1", "folds_better", "adopt"),
    [(0.40, 4, True), (0.38, 5, False), (0.40, 3, False)],  # +0.043 on 4 folds / +0.023 / only 3 folds
)
def test_the_decision_rule_needs_both_the_gain_and_the_folds(best_f1, folds_better, adopt):
    table = pd.DataFrame(
        {"cv_macro_f1": [0.357, best_f1, 0.30], "folds_better_than_mean": [0, folds_better, 1]},
        index=["mean", "top2", "top8"],
    )
    decision = decide(table)
    assert decision["sanity_mean_reproduces_e3"] and decision["best"] == "top2"
    assert decision["adopt"] is adopt
