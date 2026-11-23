"""A missing upstream rad/s-to-deg/s conversion must be visible to the drift job."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from ride_classifier.config import LABELS
from ride_classifier.drift.job import feature_records, make_report
from ride_classifier.features.window_stats import FEATURES


def test_skipped_rotation_conversion_triggers_drift(tmp_path):
    rng = np.random.default_rng(42)
    rows = []
    for window_id in range(40):
        sensors = rng.normal(size=(100, 6)) * rng.uniform(0.8, 1.2, size=6)
        frame = pd.DataFrame(sensors, columns=["ax", "ay", "az", "rx", "ry", "rz"])
        frame["window_id"] = window_id
        frame["time_index"] = np.arange(100) * 100
        rows.append(frame)
    correct = pd.concat(rows, ignore_index=True)
    expected = feature_records(correct)
    features = {}
    for feature in FEATURES:
        values = expected[feature].to_numpy()
        edges = np.unique(np.quantile(values, np.arange(1, 10) / 10))
        shares = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1) / len(values)
        features[feature] = {"edges": edges.tolist(), "shares": shares.tolist()}
    reference = {"features": features, "prediction_mix": dict.fromkeys(LABELS, 1 / len(LABELS))}
    config = {"psi_alert": 0.1, "psi_retrain": 0.25, "chi2_pvalue": 0.01, "min_labels": 20}
    assert make_report(reference, expected, config, 0.3, "reference")["status"] == "ok"
    unconverted = correct.copy()
    unconverted[["rx", "ry", "rz"]] /= 57.2958
    report = make_report(reference, feature_records(unconverted), config, 0.3, "reference")
    assert report["status"] == "retrain_recommended"
    assert all(report["features"][name]["level"] == "retrain" for name in ("rx_std", "ry_std", "rz_std"))
    assert all(report["features"][name]["level"] == "ok" for name in ("ax_std", "ay_std", "az_std"))

    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    for name, windows, expected_exit in (("converted", correct, 0), ("unit_bug", unconverted, 3)):
        current_path = tmp_path / f"{name}.parquet"
        output_path = tmp_path / name
        windows.to_parquet(current_path, index=False)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ride_classifier.drift.job",
                "--reference",
                str(reference_path),
                "--current",
                str(current_path),
                "--out",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == expected_exit, result.stdout + result.stderr
        saved_report = json.loads((output_path / "drift_report.json").read_text(encoding="utf-8"))
        assert saved_report["status"] == ("ok" if expected_exit == 0 else "retrain_recommended")
        if expected_exit == 3:
            assert all(saved_report["features"][key]["level"] == "retrain" for key in ("rx_std", "ry_std", "rz_std"))
