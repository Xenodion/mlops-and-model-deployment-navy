import math

import pytest
from prometheus_client import generate_latest

from ride_classifier.drift.monitor import Exporter, calculate
from ride_classifier.drift.psi import psi, psi_shares
from ride_classifier.features.window_stats import FEATURES


def reference():
    return {"features": {name: {"edges": [1.0], "shares": [0.5, 0.5]} for name in FEATURES}}


def series(low=10, total=20):
    return [
        {"metric": {"feature": name, "le": str(edge)}, "value": [0, str(count)]}
        for name in FEATURES
        for edge, count in ((math.nextafter(1.0, -math.inf), low), (math.inf, total))
    ]


def test_monitor_uses_same_psi_as_offline_and_handles_exact_boundary():
    values = calculate(reference(), series(low=5), 20)
    assert values["rx_std"] == pytest.approx(psi([0.5, 0.5], [0.0] * 5 + [1.0] * 15, [1.0]))
    assert psi_shares([0.5, 0.5], [5, 15]) == pytest.approx(values["rx_std"])


def test_exporter_removes_stale_values_on_missing_data_and_failure():
    exporter = Exporter(reference(), threshold=0.25, min_windows=20, alerts_enabled=True)
    exporter.update(series(low=1))
    output = generate_latest(exporter.registry).decode()
    assert 'drift_retrain_recommended{reason="input_drift"} 1.0' in output
    assert 'drift_psi{feature="rx_std"}' in output
    exporter.update(series(total=10, low=5))
    output = generate_latest(exporter.registry).decode()
    assert 'drift_psi{feature="rx_std"}' not in output
    assert 'drift_retrain_recommended{reason="input_drift"}' not in output
    assert "drift_monitor_up 0.0" in output
    exporter.update(series())
    exporter.fail()
    assert 'drift_psi{feature="rx_std"}' not in generate_latest(exporter.registry).decode()


def test_disabled_alerts_keep_psi_visible_without_retraining_recommendation():
    exporter = Exporter(reference(), threshold=0.25, min_windows=20)
    exporter.update(series(low=1))
    output = generate_latest(exporter.registry).decode()
    assert 'drift_psi{feature="rx_std"}' in output
    assert "drift_monitor_up 1.0" in output
    assert "drift_monitor_alerts_enabled 0.0" in output
    assert 'drift_retrain_recommended{reason="input_drift"}' not in output
    with pytest.raises(ValueError, match="boolean"):
        Exporter(reference(), threshold=0.25, min_windows=20, alerts_enabled="false")


def test_bad_or_incomplete_buckets_are_not_treated_as_valid():
    assert len(calculate(reference(), series()[:-1], 20)) == len(FEATURES) - 1
    with pytest.raises(ValueError, match="Non-monotonic"):
        calculate(reference(), series(low=21, total=20), 20)
    with pytest.raises(ValueError):
        psi_shares([0.5, 0.5], [0, 0])
