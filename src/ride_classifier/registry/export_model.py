"""Download the registered model behind an alias into model/, ready to be copied into the API image (labs 2, 5).

uv run python -m ride_classifier.registry.export_model              # @staging
uv run python -m ride_classifier.registry.export_model production   # @production

Next to the model, model_meta.json says exactly which model it is (version, run, commit, data, scores):
the API reports it, so every prediction can be traced back to its training.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from ride_classifier.config import MLFLOW_TRACKING_URI, MODEL_NAME, ROOT
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.register import STAGING, alias_version
from ride_classifier.registry.tracking import check_server

logger = logging.getLogger(__name__)

EXPORT_DIR = ROOT / "model"  # git-ignored; copied into the API image
META_FILE = "model_meta.json"


def export(client: MlflowClient, alias: str, destination: Path = EXPORT_DIR, name: str = MODEL_NAME) -> dict:
    """Replace `destination` with the model behind name@alias, plus model_meta.json. Returns the metadata."""
    version = alias_version(client, alias, name)
    if version is None:
        raise SystemExit(f"No {name}@{alias} in the registry: run the register step first")

    shutil.rmtree(destination, ignore_errors=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}@{alias}", dst_path=str(destination))
    meta = {
        "name": name,
        "version": version.version,
        "alias": alias,
        **version.tags,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (destination / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("alias", nargs="?", default=STAGING, help="registry alias to export (default: staging)")
    args = parser.parse_args()
    setup()
    check_server(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    meta = export(MlflowClient(MLFLOW_TRACKING_URI), args.alias)
    logger.info("%s@%s (v%s) -> %s/", meta["name"], meta["alias"], meta["version"], EXPORT_DIR.name)


if __name__ == "__main__":
    main()
