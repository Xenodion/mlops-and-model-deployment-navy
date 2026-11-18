"""Register the latest trained model in the MLflow Model Registry and manage its aliases (docs/PLAN.md §8).

uv run python -m ride_classifier.registry.register             # latest training -> new version, alias @staging
uv run python -m ride_classifier.registry.register --promote   # version @staging -> also @production

Consumers load a model by name and alias (models:/ride-classifier@staging), never by file path. Aliases replace
MLflow's deprecated stages (lab 1). A run is registered only if it can be trusted and rebuilt, see
`registration_problems`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from ride_classifier.config import MLFLOW_TRACKING_URI, MODEL_NAME, ROOT, load_params
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.tracking import check_server

logger = logging.getLogger(__name__)

STAGING, PRODUCTION = "staging", "production"
VERSION_TAGS = ("git_commit", "data_md5", "labels_md5", "model_type", "experiment_id")  # copied from the run


def registration_problems(tags: dict, metrics: dict, production_score: float | None) -> list[str]:
    """Why this run must not be registered (empty list = it can be)."""
    problems = []
    if tags.get("git_dirty") != "False":
        problems.append(
            f"trained on uncommitted code (git_dirty={tags.get('git_dirty')}): commit, retrain, then register"
        )
    if tags.get("gate") != "passed":
        problems.append(f"quality gate not passed (gate={tags.get('gate')}): run the evaluate stage")
    score = metrics.get("golden_macro_f1")
    if production_score is not None and (score is None or score < production_score):
        problems.append(f"golden macro-F1 {score} is below the production model's {production_score:.3f}")
    return problems


def alias_version(client: MlflowClient, alias: str, name: str = MODEL_NAME) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None  # no such model or alias yet


def register(client: MlflowClient, run_id: str, model_uri: str, name: str = MODEL_NAME) -> ModelVersion:
    """Register the run's model (or reuse its version if already registered) and point @staging at it."""
    run = client.get_run(run_id)
    production = alias_version(client, PRODUCTION, name)
    production_score = float(production.tags["golden_macro_f1"]) if production else None
    problems = registration_problems(run.data.tags, run.data.metrics, production_score)
    if problems:
        raise SystemExit(f"Registration refused for run {run_id}:\n- " + "\n- ".join(problems))

    existing = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
    if existing:
        version = existing[0]
    else:
        tags = {key: run.data.tags.get(key, "unknown") for key in VERSION_TAGS}
        tags.update(
            run_id=run_id,
            cv_macro_f1=f"{run.data.metrics['cv_macro_f1']:.4f}",
            golden_macro_f1=f"{run.data.metrics['golden_macro_f1']:.4f}",
        )
        version = mlflow.register_model(model_uri, name, tags=tags)
    client.set_registered_model_alias(name, STAGING, version.version)
    return version


def promote(client: MlflowClient, name: str = MODEL_NAME) -> ModelVersion:
    """Point @production at the version currently @staging (in CD: only after the manual approval)."""
    version = alias_version(client, STAGING, name)
    if version is None:
        raise SystemExit(f"No {name}@{STAGING} to promote: register a model first")
    client.set_registered_model_alias(name, PRODUCTION, version.version)
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--promote", action="store_true", help="move @production to the version @staging")
    args = parser.parse_args()
    setup()
    check_server(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if args.promote:
        version = promote(client)
        logger.info("%s v%s is now @%s", MODEL_NAME, version.version, PRODUCTION)
        return
    run_info = json.loads(Path(ROOT / load_params()["train"]["run_path"]).read_text(encoding="utf-8"))
    version = register(client, run_info["run_id"], run_info["model_uri"])
    logger.info("%s v%s is now @%s (run %s)", MODEL_NAME, version.version, STAGING, run_info["run_id"])


if __name__ == "__main__":
    main()
