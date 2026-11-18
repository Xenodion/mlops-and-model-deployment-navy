"""Rebuild a training run from scratch and check it gives the same scores (lab 1 deliverable, docs/PLAN.md §8).

uv run python -m ride_classifier.registry.repro_check            # the run in artifacts/train_run.json
uv run python -m ride_classifier.registry.repro_check RUN_ID

1. code: the training code here is the code of the run's commit (no difference, nothing uncommitted)
2. data: the raw files here have the fingerprints recorded by the run
3. scores: prepare -> train -> evaluate rerun in a temporary folder, with a throwaway MLflow store,
   give every CV and golden score of the run within 1e-6
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from ride_classifier.config import MLFLOW_TRACKING_URI, ROOT, load_params
from ride_classifier.pipeline import evaluate, prepare, train
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.tracking import check_server, dvc_md5

logger = logging.getLogger(__name__)

TOLERANCE = 1e-6
# Everything that shapes the model (the dvc.yaml dependencies of prepare, train and evaluate)
TRAINING_PATHS = [
    "src/ride_classifier/config.py",
    "src/ride_classifier/data",
    "src/ride_classifier/features",
    "src/ride_classifier/models",
    "src/ride_classifier/pipeline",
    "params.yaml",
    "uv.lock",
]


def code_problem(run_commit: str) -> str | None:
    """None if the training code here is exactly the run's; otherwise what differs and how to fix it."""
    committed = subprocess.run(
        ["git", "diff", "--quiet", run_commit, "HEAD", "--", *TRAINING_PATHS], cwd=ROOT, check=False
    ).returncode
    uncommitted = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *TRAINING_PATHS], cwd=ROOT, check=False)
    if committed != 0:
        return f"training code differs from the run's commit: git checkout {run_commit}, then uv sync --locked"
    if uncommitted.returncode != 0:
        return "uncommitted changes in the training code: commit or stash them"
    return None


def data_problems(run_tags: dict, data_params: dict) -> list[str]:
    """Raw files whose fingerprint differs from the one the run recorded."""
    problems = []
    for tag, key in (("data_md5", "raw_rides"), ("labels_md5", "raw_labels")):
        here = dvc_md5(ROOT / f"{data_params[key]}.dvc")
        if here != run_tags.get(tag):
            problems.append(f"{data_params[key]}: md5 {here} here, {run_tags.get(tag)} in the run (make data)")
    return problems


def compare(original: dict[str, float], rebuilt: dict[str, float]) -> dict[str, float]:
    """Absolute difference for every score of the rebuilt run that the original run also has."""
    return {name: abs(rebuilt[name] - original[name]) for name in sorted(rebuilt) if name in original}


def rebuild(params: dict) -> dict[str, float]:
    """Rerun prepare -> train -> evaluate from the raw files, in a temporary folder. Returns all scores."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        for key in ("raw_rides", "raw_labels"):
            (root / params["data"][key]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / params["data"][key], root / params["data"][key])
        tracking_uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.create_experiment("repro-check", artifact_location=(root / "mlruns").as_uri())

        prepare.run(params["data"], root=root)
        scores = train.run(params, root=root, tracking_uri=tracking_uri, experiment="repro-check")
        scores.update(evaluate.run(params, root=root, tracking_uri=tracking_uri))
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", nargs="?", help="MLflow run to rebuild (default: artifacts/train_run.json)")
    args = parser.parse_args()
    setup()
    params = load_params()
    run_id = args.run_id or json.loads((ROOT / params["train"]["run_path"]).read_text(encoding="utf-8"))["run_id"]
    check_server(MLFLOW_TRACKING_URI)
    run = MlflowClient(MLFLOW_TRACKING_URI).get_run(run_id)

    code = code_problem(run.data.tags["git_commit"])
    print(f"{'FAIL' if code else 'PASS'} code  (training code of commit {run.data.tags['git_commit'][:7]})", flush=True)
    data = data_problems(run.data.tags, params["data"])
    print(f"{'FAIL' if data else 'PASS'} data  (dvc md5 of the raw files)", flush=True)
    problems = ([code] if code else []) + data
    if problems:
        raise SystemExit("NOT REPRODUCIBLE HERE:\n- " + "\n- ".join(problems))

    logger.info("rebuilding run %s from the raw data (a few minutes)...", run_id)
    deltas = compare(run.data.metrics, rebuild(params))
    for name, delta in deltas.items():
        print(f"{'PASS' if delta <= TOLERANCE else 'FAIL'} {name:32s} delta={delta:.1e}")
    if any(delta > TOLERANCE for delta in deltas.values()):
        raise SystemExit("NOT REPRODUCED: some scores differ")
    print("\nREPRODUCED")


if __name__ == "__main__":
    main()
