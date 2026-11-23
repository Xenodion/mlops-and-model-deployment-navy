"""Copy registered models from one MLflow server to another, with their runs and aliases (docs/PLAN.md §11).

The API images contain no model (ADR 007): in the cluster, the API loads model 1 (ride-classifier), model 2
(fall-risk) and the drift reference (a file of model 1's training run) from the cluster's MLflow. This fills it from
your dev MLflow or the shared one, with the same version numbers:

    make k8s-model                                        # dev MLflow (:5000) -> cluster MLflow (:8082)
    make k8s-model MLFLOW_SOURCE=http://localhost:5001    # from the shared MLflow (make shared-mlflow first)
    uv run python -m ride_classifier.registry.mirror --source URL --target URL [--name ride-classifier] ...

Each version keeps its run (params, metric history, tags, files) and its exact model files. Re-runnable: copies are
recognised by the id of the *original* run (tags mirrored_from_run_id / migrated_from_run_id), whichever server the
source is; files added to a run since (e.g. the drift reference) are copied too; the aliases follow the source.
Only runs that pass the registration rules of register.py (committed code, quality gate passed) are copied.
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
from mlflow.entities import Param, Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from ride_classifier.config import FALL_RISK_NAME, MODEL_NAME
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.register import registration_problems
from ride_classifier.registry.tracking import check_server

logger = logging.getLogger(__name__)

RUN_TAG, MODEL_TAG = "mirrored_from_run_id", "mirrored_from_model_id"
MIGRATED_TAGS = ("migrated_from_run_id",)  # copies made by hand before this tool (shared MLflow, 21/11)


class NotRegistered(SystemExit):
    """The source has no such registered model."""


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


def copied_from(run: Run) -> str | None:
    """The id of the original run this run is a copy of, or None if it is an original."""
    tags = run.data.tags
    return next((tags[key] for key in (RUN_TAG, *MIGRATED_TAGS) if key in tags), None)


def origin_of(run: Run) -> str:
    """The id of the original run: the same for every copy of it, on any server."""
    return copied_from(run) or run.info.run_id


def sync_files(source: MlflowClient, target: MlflowClient, run_id: str, copy_id: str) -> list[str]:
    """Copy the run's files (top-level files and folders) that the target copy doesn't have yet."""
    have = {artifact.path for artifact in target.list_artifacts(copy_id)}
    missing = [artifact for artifact in source.list_artifacts(run_id) if artifact.path not in have]
    for artifact in missing:
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=artifact.path, tracking_uri=source.tracking_uri, dst_path=tmp
            )
            if artifact.is_dir:
                target.log_artifacts(copy_id, local, artifact_path=artifact.path)
            else:
                target.log_artifact(copy_id, local)
    return [artifact.path for artifact in missing]


def copy_run(source: MlflowClient, target: MlflowClient, run_id: str) -> str:
    """The target copy of a source run (created once, then only its new files): same experiment name, params,
    metric history, tags and files."""
    run = source.get_run(run_id)
    origin = origin_of(run)
    experiment_name = source.get_experiment(run.info.experiment_id).name
    experiment = target.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else target.create_experiment(experiment_name)
    for key in (RUN_TAG, *MIGRATED_TAGS):
        found = target.search_runs([experiment_id], filter_string=f"tags.{key} = '{origin}'")
        if found:
            added = sync_files(source, target, run_id, found[0].info.run_id)
            if added:
                logger.info("run %s: new files copied %s", found[0].info.run_id, added)
            return found[0].info.run_id

    tags = {key: value for key, value in run.data.tags.items() if not key.startswith("mlflow.")}
    tags[RUN_TAG] = origin
    copy = target.create_run(experiment_id, start_time=run.info.start_time, tags=tags, run_name=run.info.run_name)
    copy_id = copy.info.run_id
    for batch in _chunks([Param(key, value) for key, value in run.data.params.items()], 100):
        target.log_batch(copy_id, params=batch)
    history = [point for key in run.data.metrics for point in source.get_metric_history(run_id, key)]
    for batch in _chunks(history, 1000):
        target.log_batch(copy_id, metrics=batch)
    sync_files(source, target, run_id, copy_id)
    target.set_terminated(copy_id, status=run.info.status, end_time=run.info.end_time)
    return copy_id


def copy_model(source: MlflowClient, target: MlflowClient, model_id: str, run_id: str) -> str:
    """The target copy of a logged model (created once): the same files, with ids pointing to the copies."""
    model = source.get_logged_model(model_id)
    origin = model.tags.get(MODEL_TAG) or model_id
    experiment_id = target.get_run(run_id).info.experiment_id
    for existing in target.search_logged_models([experiment_id]):
        if existing.tags.get(MODEL_TAG) == origin:
            return existing.model_id

    tags = {key: value for key, value in model.tags.items() if not key.startswith("mlflow.")}
    tags[MODEL_TAG] = origin
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
        raise NotRegistered(f"No registered model '{name}' on {source.tracking_uri}")
    runs = {v.run_id: source.get_run(v.run_id) for v in wanted}
    untrusted = {
        v.version: problems
        for v in wanted
        if (problems := registration_problems(runs[v.run_id].data.tags, runs[v.run_id].data.metrics, None))
    }
    if untrusted:
        raise SystemExit(f"Not copied, these versions would not pass the registration rules: {untrusted}")

    existing = {}  # original run id -> target version already copied
    for version in _versions(target, name):
        origin = copied_from(target.get_run(version.run_id)) if version.run_id else None
        if origin is None:
            raise SystemExit(
                f"'{name}' v{version.version} on {target.tracking_uri} was not copied by this tool: the version "
                "numbers would differ from the source. Use an empty target (e.g. a new cluster)."
            )
        existing[origin] = str(version.version)

    numbers = {}
    for version in wanted:
        run_id = copy_run(source, target, version.run_id)  # also brings files added to the run since
        if (origin := origin_of(runs[version.run_id])) in existing:
            numbers[str(version.version)] = existing[origin]
            continue
        if not str(version.source).startswith("models:/"):
            raise SystemExit(f"v{version.version}: source {version.source} is not a logged model (models:/...)")
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
        logger.info("%s @%s -> v%s", name, alias, numbers[str(number)])
    return numbers


def mirror_all(source: MlflowClient, target: MlflowClient, names: list[str]) -> dict[str, dict[str, str]]:
    """Mirror each model. Model 1 is required; the others (model 2) are skipped when the source has none."""
    copied = {}
    for name in names:
        try:
            copied[name] = mirror(source, target, name)
        except NotRegistered:
            if name == MODEL_NAME:
                raise
            logger.warning("No '%s' on %s: skipped (the API then serves model 1 alone)", name, source.tracking_uri)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="MLflow to copy from, e.g. http://127.0.0.1:5000")
    parser.add_argument("--target", required=True, help="MLflow to copy to, e.g. http://127.0.0.1:8082")
    parser.add_argument(
        "--name", action="append", help=f"registered model (repeatable), default {MODEL_NAME} and {FALL_RISK_NAME}"
    )
    args = parser.parse_args()
    setup()
    for uri in (args.source, args.target):
        check_server(uri)
    mirror_all(MlflowClient(args.source), MlflowClient(args.target), args.name or [MODEL_NAME, FALL_RISK_NAME])


if __name__ == "__main__":
    main()
