import numpy as np
import pandas as pd
import pytest

from ride_classifier.drift.psi import psi
from ride_classifier.features.window_stats import FEATURES, window_stats
from ride_classifier.pipeline.reference import build_reference


def test_psi_uses_actual_reference_shares_and_catches_shift():
    reference = [0.25, 0.75]
    assert psi(reference, [0, 1, 1, 1], [0.5]) == pytest.approx(0)
    assert psi(reference, [0, 0, 0, 1], [0.5]) > 0.25
    with pytest.raises(ValueError):
        psi(reference, [], [0.5])


def test_reference_handles_tied_deciles_and_checks_oof_coverage():
    frame = pd.DataFrame(
        {
            "ride_id": ["a"] * 20,
            "time_index": np.arange(20) * 100,
            **{col: np.arange(20, dtype=float) for col in ("ax", "ay", "az", "rx", "ry", "rz")},
        }
    )
    predictions = pd.DataFrame({"ride_id": ["a"], "predicted": ["Safe"]})
    result = build_reference(frame, predictions, 1, 1)
    assert set(result["features"]) == set(FEATURES)
    assert result["n_windows"] == 2
    assert result["prediction_mix"]["Safe"] == 1
    assert all(sum(feature["shares"]) == pytest.approx(1) for feature in result["features"].values())
    with pytest.raises(ValueError, match="exactly one"):
        build_reference(frame, predictions.assign(ride_id="other"), 1, 1)
    assert set(window_stats(frame)) == set(FEATURES)
