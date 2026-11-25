import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ride_classifier.features.store import FeatureView, default_store


def test_feature_store_materializes_offline_and_online_views(tmp_path, labelled_rides):
    readings, _ = labelled_rides
    subset = readings[readings["ride_id"].isin(["Safe-0", "Tandem-0", "Reckless-0"])].copy()
    store = default_store(tmp_path / "fs")

    summary = store.materialize("ride_stats", subset, version="v1")
    assert summary["name"] == "ride_stats"
    assert summary["n_rows"] == 3
    assert summary["n_features"] > 100

    offline = store.get_offline_features("ride_stats", version="v1", entity_ids=["Tandem-0", "Safe-0"])
    assert list(offline.index) == ["Tandem-0", "Safe-0"]
    assert offline["duration_s"].tolist() == [60.0, 60.0]

    online = store.get_online_features("ride_stats", ["Safe-0", "missing-id"], version="v1")
    assert online["missing-id"] is None
    assert online["Safe-0"] is not None
    assert online["Safe-0"]["duration_s"] == 60.0
    np.testing.assert_allclose(online["Safe-0"]["ax_std"], float(offline.loc["Safe-0", "ax_std"]))

    listed = store.list_views()
    assert len(listed) == 1 and listed[0]["name"] == "ride_stats"


def test_feature_store_supports_drift_view_and_version_isolation(tmp_path, labelled_rides):
    readings, _ = labelled_rides
    subset = readings[readings["ride_id"].isin(["Safe-0", "Safe-1"])].copy()
    store = default_store(tmp_path / "fs")

    store.materialize("window_drift_stats", subset, version="v1")
    store.register(
        FeatureView(
            name="window_drift_stats",
            version="v2",
            entity_column="ride_id",
            description="Custom v2 test view",
            compute=lambda df: pd.DataFrame({"n_rows": df.groupby("ride_id").size()}),
        )
    )
    store.materialize("window_drift_stats", subset, version="v2")

    v1 = store.get_online_features("window_drift_stats", ["Safe-0"], version="v1")["Safe-0"]
    v2 = store.get_online_features("window_drift_stats", ["Safe-0"], version="v2")["Safe-0"]
    assert "ax_std" in v1 and "acc_norm_std" in v1
    assert v2 == {"n_rows": 600}

    with pytest.raises(KeyError):
        store.describe("window_drift_stats", version="v99")


def test_feature_store_cli_roundtrip(tmp_path, ride):
    input_path = tmp_path / "rides.parquet"
    ride.to_parquet(input_path, index=False)
    fs_root = tmp_path / "fs"

    mat = subprocess.run(
        [
            sys.executable,
            "-m",
            "ride_classifier.features.store",
            "--root",
            str(fs_root),
            "materialize",
            "--input",
            str(input_path),
            "--view",
            "ride_stats",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(mat.stdout)["n_rows"] == 1

    online = subprocess.run(
        [
            sys.executable,
            "-m",
            "ride_classifier.features.store",
            "--root",
            str(fs_root),
            "online",
            "--view",
            "ride_stats",
            "--entity-id",
            "ride-test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(online.stdout)
    assert payload["ride-test"]["duration_s"] == 60.0
