import pandas as pd
import pytest

from ride_classifier.drift.scenarios import scenario_windows


def readings():
    rows = []
    for ride, label in (("a", "Safe"), ("b", "Reckless")):
        for time_index in range(0, 2000, 100):
            rows.append(
                {
                    "ride_id": ride,
                    "label": label,
                    "time_index": time_index,
                    **dict.fromkeys(("ax", "ay", "az", "rx", "ry", "rz"), 1.0),
                }
            )
    return pd.DataFrame(rows)


def test_scenarios_are_reproducible_and_repeated_rides_have_distinct_windows():
    config = {"seed": 42, "n_rides": 12, "riskier_weights": {"Safe": 0, "Reckless": 1}}
    normal, metadata = scenario_windows(readings(), "normal", config, 1, 1)
    again, repeated_metadata = scenario_windows(readings(), "normal", config, 1, 1)
    pd.testing.assert_frame_equal(normal, again)
    assert metadata == repeated_metadata
    assert normal["window_id"].nunique() == 24
    assert normal.groupby("window_id").size().eq(10).all()
    risky, risky_metadata = scenario_windows(readings(), "riskier_mix", config, 1, 1)
    assert set(risky_metadata["rides"]) == {"b"}
    assert set(risky["true_label"]) == {"Reckless"}


def test_rejects_inconsistent_labels():
    data = readings()
    data.loc[0, "label"] = "Reckless"
    with pytest.raises(ValueError, match="one label"):
        scenario_windows(data, "normal", {"seed": 1, "n_rides": 1}, 1, 1)
