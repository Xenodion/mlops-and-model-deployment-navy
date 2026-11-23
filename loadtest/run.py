"""Run a Locust step test only after validating and tracking the target in MLflow."""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from urllib.request import urlopen

import mlflow

from ride_classifier.config import MLFLOW_TRACKING_URI, ROOT, load_params
from ride_classifier.pipeline.cli import setup
from ride_classifier.registry.tracking import check_server, run_tags


def target_health(host: str, expected_version: str) -> dict:
    with urlopen(host.rstrip("/") + "/health", timeout=10) as response:
        health = json.load(response)
    if not health.get("model_loaded") or str(health.get("model_version")) != expected_version:
        raise ValueError("API model is not loaded or does not match the requested model version")
    return health


def main() -> None:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="API base URL")
    parser.add_argument("--model-version", required=True, help="Expected model version reported by /health")
    parser.add_argument("--speed", type=float, default=None, help="Replay speed factor; 1 means real time")
    parser.add_argument("--target-cpu", type=float, required=True, help="Declared target CPU limit in cores")
    parser.add_argument("--replicas", type=int, default=1, help="Declared number of API replicas")
    parser.add_argument("--target-env", required=True, help="Declared target machine and deployment environment")
    args = parser.parse_args()
    params = load_params()
    speed = args.speed if args.speed is not None else params["loadtest"]["speed"]
    if (
        not math.isfinite(speed)
        or speed <= 0
        or not math.isfinite(args.target_cpu)
        or args.target_cpu <= 0
        or args.replicas < 1
        or args.model_version in {"", "unregistered", "none"}
        or not args.target_env.strip()
    ):
        parser.error(
            "speed, target-cpu and replicas must be positive and model-version must identify a registered model"
        )
    check_server(MLFLOW_TRACKING_URI)
    health = target_health(args.host, args.model_version)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("live-load-test")
    with mlflow.start_run(run_name="live-ride-steps") as run:
        output = ROOT / "reports/load" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{run.info.run_id}"
        output.mkdir(parents=True, exist_ok=False)
        mlflow.set_tags(run_tags({"gbg500_md5": ROOT / "data/raw/GBG500.parquet.dvc"}))
        mlflow.log_params(
            {
                "host": args.host,
                "model_version": args.model_version,
                "fall_risk_loaded": str(health.get("fall_risk_loaded")),
                "declared_target_cpu_cores": args.target_cpu,
                "declared_api_replicas": args.replicas,
                "declared_target_environment": args.target_env,
                "speed": speed,
                "window_s": params["train"]["cnn"]["window_s"],
                "stride_s": params["train"]["cnn"]["stride_s"],
                **{key: value for key, value in params["loadtest"].items() if key != "speed"},
            }
        )
        env = {**os.environ, "LOADTEST_SPEED": str(speed)}
        command = [
            sys.executable,
            "-m",
            "locust",
            "-f",
            "loadtest/locustfile.py,loadtest/shape.py",
            "--headless",
            "--host",
            args.host,
            "--csv",
            str(output / "locust"),
            "--csv-full-history",
        ]
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        finally:
            mlflow.log_artifacts(str(output), artifact_path="locust")
        if completed.returncode:
            raise SystemExit(f"Locust failed with exit code {completed.returncode}")
        print(f"MLflow run {run.info.run_id}; raw CSV files: {output}")


if __name__ == "__main__":
    main()
