import json
import subprocess
import sys

import pandas as pd

from ride_classifier.config import LABELS
from ride_classifier.drift.job import feature_records, make_report
from ride_classifier.features.window_stats import FEATURES

CONFIG = {"psi_alert": 0.1, "psi_retrain": 0.25, "chi2_pvalue": 0.01, "min_labels": 20}


def reference():
    return {
        "model_version": "test",
        "features": {name: {"edges": [0.5], "shares": [0.5, 0.5]} for name in FEATURES},
        "prediction_mix": {label: 1 / len(LABELS) for label in LABELS},
    }


def test_normal_and_shifted_input_and_missing_optional_labels():
    normal = pd.DataFrame({feature: [0.0, 1.0] * 20 for feature in FEATURES})
    report = make_report(reference(), normal, CONFIG, 0.3, "reference.json")
    assert report["status"] == "ok" and report["prediction_mix"] is None and report["performance"] is None
    normal["rot_norm_std"] = 0.0
    report = make_report(reference(), normal, CONFIG, 0.3, "reference.json")
    assert report["status"] == "retrain_recommended" and report["reasons"] == ["input_drift"]
    assert report["features"]["rot_norm_std"]["ks_pvalue"] is None


def test_delayed_labels_and_prediction_shift():
    data = pd.DataFrame({feature: [0.0, 1.0] * 20 for feature in FEATURES})
    data["predicted_label"] = "Safe"
    data["true_label"] = "Reckless"
    report = make_report(reference(), data, CONFIG, 0.3, "reference.json")
    assert set(report["reasons"]) == {"prediction_drift", "performance"}
    assert report["performance"]["macro_f1"] == 0


def test_prediction_mix_uses_one_latest_prediction_per_ride():
    data = pd.DataFrame({feature: [0.0, 1.0, 0.0, 1.0] for feature in FEATURES})
    data["ride_id"] = ["a", "b", "a", "b"]
    data["window_index"] = [1, 0, 0, 1]
    data["predicted_label"] = ["Safe", "Safe", "Reckless", "Tandem"]
    data["true_label"] = ["Safe", "Tandem", "Safe", "Tandem"]
    report = make_report(reference(), data, {**CONFIG, "min_labels": 2}, 0.3, "reference.json")
    assert report["n_windows"] == 4
    assert report["prediction_mix"]["n_predictions"] == 2
    assert report["prediction_mix"]["current"]["Safe"] == 0.5
    assert report["prediction_mix"]["current"]["Tandem"] == 0.5
    assert report["performance"]["n_labels"] == 2


def test_raw_sensor_windows_are_converted_to_features():
    data = pd.DataFrame(
        {
            "window_id": ["one", "one", "two", "two"],
            "time_index": [0, 100, 0, 100],
            **{column: [0.0, 2.0, 0.0, 4.0] for column in ("ax", "ay", "az", "rx", "ry", "rz")},
        }
    )
    rows = feature_records(data)
    assert len(rows) == 2
    assert rows["rx_std"].tolist() == [1.0, 2.0]


def test_cli_exit_codes_and_output(tmp_path):
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps(reference()), encoding="utf-8")
    current = tmp_path / "windows.jsonl"
    pd.DataFrame({feature: [0.0] * 20 for feature in FEATURES}).to_json(current, orient="records", lines=True)
    out = tmp_path / "report"
    command = [
        sys.executable,
        "-m",
        "ride_classifier.drift.job",
        "--reference",
        str(ref),
        "--current",
        str(current),
        "--out",
        str(out),
    ]
    assert subprocess.run(command, check=False).returncode == 3
    assert json.loads((out / "drift_report.json").read_text())["status"] == "retrain_recommended"
    assert (out / "drift_report.md").exists()
    current.write_text("{}\n", encoding="utf-8")
    assert subprocess.run(command, check=False).returncode == 1
