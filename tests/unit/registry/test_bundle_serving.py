import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ride_classifier.registry.bundle_serving import bundle


def test_bundle_checks_reference_and_records_registry_provenance(tmp_path):
    destination = tmp_path / "model"
    destination.mkdir()
    (destination / "model_meta.json").write_text(json.dumps({"run_id": "ride-run"}), encoding="utf-8")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"model_version": "models:/m-ride"}), encoding="utf-8")
    artifact = tmp_path / "fall.joblib"
    artifact.write_bytes(b"model bytes")
    client = Mock()
    client.get_run.return_value = SimpleNamespace(info=SimpleNamespace(experiment_id="42"))
    client.search_logged_models.return_value = [SimpleNamespace(model_uri="models:/m-ride")]
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="2", run_id="fall-run")
    client.download_artifacts.return_value = str(artifact)

    details = bundle(client, destination, reference, "staging")
    assert (destination / "fall_risk_model.joblib").read_bytes() == b"model bytes"
    assert (destination / "drift_reference.json").read_bytes() == reference.read_bytes()
    assert details["fall_risk"]["run_id"] == "fall-run"
    assert len(details["drift_reference"]["sha256"]) == 64
    assert json.loads((destination / "serving_bundle.json").read_text()) == details

    reference.write_text(json.dumps({"model_version": "models:/m-other"}), encoding="utf-8")
    client.download_artifacts.reset_mock()
    with pytest.raises(ValueError, match="does not match"):
        bundle(client, destination, reference, "staging")
    client.download_artifacts.assert_not_called()
