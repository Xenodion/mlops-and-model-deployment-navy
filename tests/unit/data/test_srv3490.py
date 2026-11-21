import pandas as pd
import pytest

from ride_classifier.data.srv3490 import check_no_overlap, harmonise_rotation, ride_table


def make_readings() -> pd.DataFrame:
    rows = []
    for ride_id, crash_ref, topples in [("c1", None, [0, 0, 1, 1]), ("b1", "c1", [0, 0, 0, 0])]:
        for i, topple in enumerate(topples):
            rows.append((ride_id, crash_ref, 3, i * 100, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, float(topple)))
    cols = ["ride_id", "crash_ride_id", "city", "time_index", "ax", "ay", "az", "rx", "ry", "rz", "topple"]
    return pd.DataFrame(rows, columns=cols)


def test_ride_table_gives_fall_time_for_crash_ride_and_none_for_baseline():
    table = ride_table(make_readings()).set_index("ride_id")
    assert table.loc["c1", "fall_time_index"] == 200
    assert pd.isna(table.loc["b1", "fall_time_index"])
    assert table.loc["c1", "n_readings"] == 4
    assert table.loc["c1", "duration_s"] == pytest.approx(0.3)


def test_ride_table_groups_baseline_with_its_crash_ride():
    table = ride_table(make_readings()).set_index("ride_id")
    assert table.loc["c1", "fall_group_id"] == "c1"
    assert table.loc["b1", "fall_group_id"] == "c1"


def test_harmonise_rotation_scales_rotation_only_and_keeps_input():
    readings = make_readings()
    out = harmonise_rotation(readings, 10.0)
    assert out["rx"].iloc[0] == pytest.approx(1.0)
    assert (out["ax"] == readings["ax"]).all()
    assert readings["rx"].iloc[0] == pytest.approx(0.1)


def test_check_no_overlap_raises_naming_the_count():
    with pytest.raises(ValueError, match="2 ride ids"):
        check_no_overlap(["a", "b", "c"], ["b", "c", "d"])
    check_no_overlap(["a"], ["b"])


def test_load_clean_srv3490_converts_rotation_and_fills_gaps(tmp_path):
    from ride_classifier.data.srv3490 import load_clean_srv3490

    raw = pd.DataFrame(
        {
            "ride_id": ["a", "a", "a", "b"],
            "time_index": [0, 100, 200, 0],
            "ax": [1.0, None, 3.0, 0.0],
            "ay": [0.0] * 4,
            "az": [0.0] * 4,
            "rx": [1.0, 1.0, 1.0, 2.0],
            "ry": [0.0] * 4,
            "rz": [0.0] * 4,
            "topple": [0.0] * 4,
        }
    )
    path = tmp_path / "srv.parquet"
    raw.to_parquet(path)
    out = load_clean_srv3490(path, rotation_factor=10.0, ride_ids=["a"])
    assert list(out["ride_id"].unique()) == ["a"]
    assert out["ax"].tolist() == [1.0, 2.0, 3.0]  # gap interpolated inside the ride
    assert out["rx"].tolist() == [10.0, 10.0, 10.0]
    assert "topple" not in out.columns and out["ax"].dtype == "float32"
