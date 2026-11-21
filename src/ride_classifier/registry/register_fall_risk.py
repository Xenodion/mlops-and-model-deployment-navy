"""Register a traceable, gate-passing fall-risk run as the staging version in MLflow."""

import json

import mlflow
from mlflow import MlflowClient

from ride_classifier.config import MLFLOW_TRACKING_URI, ROOT, load_params
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.register import STAGING
from ride_classifier.registry.tracking import check_server

MODEL_NAME = "fall-risk"


def register(client: MlflowClient, run_id: str, model_uri: str):
    """Register once per run; never promote to production automatically."""
    run = client.get_run(run_id)
    tags = run.data.tags
    if tags.get("git_dirty") != "False" or tags.get("gate") != "passed":
        raise SystemExit("Fall-risk registration requires a clean training run that passed its gate.")
    if tags.get("srv3490_md5") in (None, "unknown") or tags.get("git_commit") in (None, "unknown"):
        raise SystemExit("Fall-risk registration requires a recorded code commit and SRV3490 fingerprint.")
    if run.data.metrics.get("gate_passed") != 1.0:
        raise SystemExit("Fall-risk registration requires the recorded gate_passed metric.")
    if model_uri != f"runs:/{run_id}/model":
        # MLflow 3 returns models:/m-... URIs. Verify that the URI belongs to this run via its logged models.
        models = client.search_logged_models(
            experiment_ids=[run.info.experiment_id], filter_string=f"source_run_id = '{run_id}'"
        )
        if not any(model.model_uri == model_uri for model in models):
            raise SystemExit("The model URI does not belong to the selected training run.")
    existing = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'")
    version = (
        existing[0]
        if existing
        else mlflow.register_model(
            model_uri, MODEL_NAME, tags={"git_commit": tags["git_commit"], "srv3490_md5": tags["srv3490_md5"]}
        )
    )
    client.set_registered_model_alias(MODEL_NAME, STAGING, version.version)
    return version


def main():
    setup()
    check_server(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run = json.loads((ROOT / load_params()["fall_risk"]["run_path"]).read_text(encoding="utf-8"))
    version = register(MlflowClient(MLFLOW_TRACKING_URI), run["run_id"], run["model_uri"])
    print(f"{MODEL_NAME} v{version.version} is now @{STAGING} (run {run['run_id']})")


if __name__ == "__main__":
    main()
