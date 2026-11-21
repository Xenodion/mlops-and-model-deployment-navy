import numpy as np
import pandas as pd

from ride_classifier.models.fall_risk import window_ends, window_table

P = {"window_s": 10, "stride_s": 5, "pre_fall_s": 60, "gap_s": 5}


def test_window_ends_stop_gap_before_the_fall_and_start_after_one_full_window():
    ends = window_ends(100_000, 10, 5, 60, 5)
    assert ends.max() <= 95_000 and ends.min() == 35_000 and ends.min() >= 10_000


def test_window_table_pairs_comparison_moments_and_skips_short_rides():
    groups = pd.DataFrame({"ride_id": ["f", "c1", "c2"], "fall_group_id": ["f", "f", "f"]})
    last = pd.Series({"f": 100_000, "c1": 200_000, "c2": 50_000})
    table = window_table(last, pd.Series({"f": 100_000}), groups, P, 5)
    assert set(table.loc[table.ride_id == "c1", "end_ms"]) == set(table.loc[table.ride_id == "f", "end_ms"])
    assert table.loc[table.ride_id == "c2", "end_ms"].max() <= 50_000
    assert table.loc[table.ride_id == "f", "label"].eq(1).all() and table.loc[table.ride_id == "c1", "label"].eq(0).all()
