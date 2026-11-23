"""Copy registered models from one MLflow server to another, with their runs and aliases (docs/PLAN.md §11).

The API images contain no model (ADR 007): in the cluster, the API loads @production from the cluster's MLflow.
This fills it from your dev MLflow (or the shared one), with the same version numbers:

    make k8s-model                                        # dev MLflow (:5000) -> cluster MLflow (:8082)
    make k8s-model MLFLOW_SOURCE=http://localhost:5001    # from the shared MLflow (make shared-mlflow first)
    uv run python -m ride_classifier.registry.mirror --source URL --target URL [--name ride-classifier]

Each version keeps its run (params, metric history, tags, files) and its exact model files. Re-runnable: what is
already copied (tags mirrored_from_*) is reused, then the aliases are set as on the source. Only runs that pass the
registration rules of register.py (committed code, quality gate passed) are copied.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.entities import Param
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from ride_classifier.config import MODEL_NAME
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.register import registration_problems
from ride_classifier.registry.tracking import check_server

logger = logging.getLogger(__name__)

RUN_TAG, MODEL_TAG = "mirrored_from_run_id", "mirrored_from_model_id"


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


@contextmanager
def _current_server(uri: str):
    """Make `uri` the global tracking server for a moment: MLflow resolves some `models:/<id>` URIs on the global
    server whatever client is used (a TODO in MLflow's registry store)."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)


def copy_run(source: MlflowClient, target: MlflowClient, run_id: str) -> str:
    """The target copy of a source run (created once): same experiment name, params, metric history, tags, files."""
    run = source.get_run(run_id)
    experiment_name = source.get_experiment(run.info.experiment_id).name
    experiment = target.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else target.create_experiment(experiment_name)
    found = target.search_runs([experiment_id], filter_string=f"tags.{RUN_TAG} = '{run_id}'")
    if found:
        return found[0].info.run_id

    tags = {key: value for key, value in run.data.tags.items() if not key.startswith("mlflow.")}
    tags[RUN_TAG] = run_id
    copy = target.create_run(experiment_id, start_time=run.info.start_time, tags=tags, run_name=run.info.run_name)
    copy_id = copy.info.run_id
    for batch in _chunks([Param(key, value) for key, value in run.data.params.items()], 100):
        target.log_batch(copy_id, params=batch)
    history = [point for key in run.data.metrics for point in source.get_metric_history(run_id, key)]
    for batch in _chunks(history, 1000):
        target.log_batch(copy_id, metrics=batch)
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(run_id=run_id, tracking_uri=source.tracking_uri, dst_path=tmp)
        target.log_artifacts(copy_id, local)
    target.set_terminated(copy_id, status=run.info.status, end_time=run.info.end_time)
    return copy_id


def copy_model(source: MlflowClient, target: MlflowClient, model_id: str, run_id: str) -> str:
    """The target copy of a logged model (created once): the same files, with ids pointing to the copies."""
    model = source.get_logged_model(model_id)
    experiment_id = target.get_run(run_id).info.experiment_id
    for existing in target.search_logged_models([experiment_id]):
        if existing.tags.get(MODEL_TAG) == model_id:
            return existing.model_id

    tags = {key: value for key, value in model.tags.items() if not key.startswith("mlflow.")}
    tags[MODEL_TAG] = model_id
    copy = target.create_logged_model(
        experiment_id,
        name=model.name,
        source_run_id=run_id,
        tags=tags,
        params=model.params,
        model_type=model.model_type,
    )
    with tempfile.TemporaryDirectory() as tmp:
        # From the model's storage location, not "models:/<id>": MLflow resolves that one on the *current* tracking
        # server, whatever tracking_uri says
        local = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=model.artifact_location, tracking_uri=source.tracking_uri, dst_path=tmp
            )
        )
        mlmodel = local / "MLmodel"
        meta = yaml.safe_load(mlmodel.read_text(encoding="utf-8"))
        meta.update(run_id=run_id, model_id=copy.model_id)
        mlmodel.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        target.log_model_artifacts(copy.model_id, str(local))
    target.finalize_logged_model(copy.model_id, "READY")
    return copy.model_id


def _versions(client: MlflowClient, name: str) -> list[ModelVersion]:
    return sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))


def mirror(source: MlflowClient, target: MlflowClient, name: str = MODEL_NAME) -> dict[str, str]:
    """Copy every version of `name` and its aliases. Returns {source version: target version}."""
    wanted = _versions(source, name)
    if not wanted:
        raise SystemExit(f"No registered model '{name}' on {source.tracking_uri}")
    untrusted = {
        v.version: problems
        for v in wanted
        if (run := source.get_run(v.run_id))
        and (problems := registration_problems(run.data.tags, run.data.metrics, production_score=None))
    }
    if untrusted:
        raise SystemExit(f"Not copied, these versions would not pass the registration rules: {untrusted}")

    existing = {}  # source run id -> target version already copied
    for version in _versions(target, name):
        origin = target.get_run(version.run_id).data.tags.get(RUN_TAG) if version.run_id else None
        if origin is None:
            raise SystemExit(
                f"'{name}' v{version.version} on {target.tracking_uri} was not copied by this tool: the version "
                "numbers would differ from the source. Use an empty target (e.g. a new cluster)."
            )
        existing[origin] = str(version.version)

    numbers = {}
    for version in wanted:
        if version.run_id in existing:
            numbers[str(version.version)] = existing[version.run_id]
            continue
        if not str(version.source).startswith("models:/"):
            raise SystemExit(f"v{version.version}: source {version.source} is not a logged model (models:/...)")
        run_id = copy_run(source, target, version.run_id)
        model_id = copy_model(source, target, version.source.removeprefix("models:/"), run_id)
        try:
            target.create_registered_model(name)
        except MlflowException:
            pass  # already there
        with _current_server(target.tracking_uri):
            copied = target.create_model_version(
                name,
                source=f"models:/{model_id}",
                run_id=run_id,
                model_id=model_id,
                tags={**version.tags, "run_id": run_id},
            )
        numbers[str(version.version)] = str(copied.version)
        logger.info("%s v%s copied as v%s", name, version.version, copied.version)
        if str(copied.version) != str(version.version):
            logger.warning(
                "v%s got number v%s on the target (deleted versions on the source?)", version.version, copied.version
            )

    for alias, number in source.get_registered_model(name).aliases.items():
        target.set_registered_model_alias(name, alias, numbers[str(number)])
        logger.info("@%s -> v%s", alias, numbers[str(number)])
    return numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="MLflow to copy from, e.g. http://127.0.0.1:5000")
    parser.add_argument("--target", required=True, help="MLflow to copy to, e.g. http://127.0.0.1:8082")
    parser.add_argument("--name", action="append", help=f"registered model (repeatable), default {MODEL_NAME}")
    args = parser.parse_args()
    setup()
    for uri in (args.source, args.target):
        check_server(uri)
    source, target = MlflowClient(args.source), MlflowClient(args.target)
    for name in args.name or [MODEL_NAME]:
        mirror(source, target, name)


if __name__ == "__main__":
    main()
