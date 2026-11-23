import pandas as pd

from ride_classifier.pipeline.falls import DEVELOPMENT, HELD_OUT, pre_fall_window, segment, split_fall_groups


def _rides(groups_per_city: dict[str, int]) -> pd.DataFrame:
    rows = []
    for city, n in groups_per_city.items():
        for g in range(n):
            fall = f"{city}-{g}-fall"
            rows.append((fall, fall, city))
            rows += [(f"{city}-{g}-c{k}", fall, city) for k in range(4)]
    return pd.DataFrame(rows, columns=["ride_id", "fall_group_id", "city"])


def test_split_fall_groups_never_splits_a_group():
    out = split_fall_groups(_rides({"A": 10, "B": 6}), 0.3, 42)
    assert (out.groupby("fall_group_id")["group_split"].nunique() == 1).all()
    assert out.groupby("fall_group_id")["role"].apply(lambda r: (r == "fall").sum()).eq(1).all()


def test_split_fall_groups_is_deterministic_with_the_seed():
    rides = _rides({"A": 10, "B": 6})
    first = split_fall_groups(rides, 0.3, 42)
    again = split_fall_groups(rides.sample(frac=1, random_state=1), 0.3, 42)
    pd.testing.assert_frame_equal(first, again)


def test_split_fall_groups_keeps_every_city_in_both_splits():
    out = split_fall_groups(_rides({"A": 10, "B": 4, "C": 5}), 0.3, 0)
    assert (out.groupby("city")["group_split"].nunique() == 2).all()
    assert set(out["group_split"]) == {DEVELOPMENT, HELD_OUT}


def test_pre_fall_window_ends_before_the_fall_and_lasts_pre_fall_s():
    assert pre_fall_window(100_000, 60, 2) == (38_000, 98_000)


def test_pre_fall_window_clips_the_start_at_zero():
    assert pre_fall_window(30_000, 60, 2) == (0, 28_000)


def test_segment_is_end_exclusive_relative_to_the_ride_start():
    readings = pd.DataFrame({"time_index": range(5_000, 5_500, 100)})
    out = segment(readings, 100, 300)
    assert out["time_index"].tolist() == [5_100, 5_200]
