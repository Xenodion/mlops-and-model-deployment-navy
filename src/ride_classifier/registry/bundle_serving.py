"""Add the registered fall-risk artifact and a matching drift reference to an exported model."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from ride_classifier.config import MLFLOW_TRACKING_URI, ROOT, load_params
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.export_model import EXPORT_DIR, META_FILE
from ride_classifier.registry.register_fall_risk import MODEL_NAME
from ride_classifier.registry.tracking import check_server


def bundle(client: MlflowClient, destination: Path, reference_path: Path, alias: str) -> dict:
    """Bundle only a reference generated from the same training run as the exported classifier."""
    meta = json.loads((destination / META_FILE).read_text(encoding="utf-8"))
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    run_id = meta["run_id"]
    run = client.get_run(run_id)
    logged = client.search_logged_models(
        experiment_ids=[run.info.experiment_id], filter_string=f"source_run_id = '{run_id}'"
    )
    if reference.get("model_version") not in {model.model_uri for model in logged}:
        raise ValueError("Drift reference does not match the exported classifier's training run")
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    if not version.run_id:
        raise ValueError("The fall-risk registry version has no source run")
    artifact_name = Path(load_params()["fall_risk"]["model_path"]).name
    with tempfile.TemporaryDirectory() as temp:
        downloaded = Path(client.download_artifacts(version.run_id, artifact_name, temp))
        shutil.copyfile(downloaded, destination / "fall_risk_model.joblib")
    (destination / "drift_reference.json").write_bytes(reference_bytes)
    details = {
        "fall_risk": {"name": MODEL_NAME, "version": version.version, "alias": alias, "run_id": version.run_id},
        "drift_reference": {
            "model_uri": reference["model_version"],
            "sha256": hashlib.sha256(reference_bytes).hexdigest(),
        },
    }
    (destination / "serving_bundle.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fall-alias", default="staging")
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/drift_reference.json")
    args = parser.parse_args()
    setup()
    check_server(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(json.dumps(bundle(MlflowClient(MLFLOW_TRACKING_URI), EXPORT_DIR, args.reference, args.fall_alias), indent=2))


if __name__ == "__main__":
    main()
