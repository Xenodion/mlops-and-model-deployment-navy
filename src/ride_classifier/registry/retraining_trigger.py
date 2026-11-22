"""Record the reason for a new, reproducible retraining run before its evaluation gate."""

import argparse
import json
import os
from pathlib import Path

from mlflow import MlflowClient

from ride_classifier.config import MLFLOW_TRACKING_URI, ROOT, load_params
from ride_classifier.registry.tracking import check_server

REASONS = {"schedule", "input_drift", "prediction_drift", "performance", "manual"}


def record_trigger(client, run_id: str, previous_run: str, commit: str, reason: str, evidence: str, workflow: str):
    """Reject stale or untraceable runs before attaching trigger metadata."""
    if reason not in REASONS:
        raise ValueError(f"Unknown trigger reason: {reason}")
    if not run_id or run_id == previous_run:
        raise ValueError("Training did not produce a new MLflow run")
    if not commit or (reason not in {"schedule", "manual"} and not evidence.strip()):
        raise ValueError("A commit is required; drift and performance triggers also require evidence")
    run = client.get_run(run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("git_dirty") != "False":
        raise ValueError("Training run must be finished and produced from a clean working tree")
    if run.data.tags.get("git_commit") != commit:
        raise ValueError("Training run commit does not match the workflow checkout")
    for key, value in {"trigger_reason": reason, "trigger_evidence": evidence, "trigger_workflow": workflow}.items():
        client.set_tag(run_id, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", default="")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--reason", required=True, choices=sorted(REASONS))
    parser.add_argument("--evidence", default="")
    parser.add_argument("--workflow", default="")
    args = parser.parse_args()
    check_server(MLFLOW_TRACKING_URI)
    run_path = ROOT / load_params()["train"]["run_path"]
    run_id = json.loads(Path(run_path).read_text(encoding="utf-8"))["run_id"]
    record_trigger(
        MlflowClient(MLFLOW_TRACKING_URI),
        run_id,
        args.previous_run,
        args.commit,
        args.reason,
        args.evidence,
        args.workflow,
    )
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as output:
            output.write(f"Training run: `{run_id}`\n\nTrigger: `{args.reason}`\n")


if __name__ == "__main__":
    main()
