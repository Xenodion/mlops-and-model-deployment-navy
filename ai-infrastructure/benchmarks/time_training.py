"""Time one release training stage in a fresh process and isolated output directory."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "train_fall_risk"])
    parser.add_argument("workspace", type=Path, help="Isolated release checkout containing the data inputs")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    if (root / "artifacts").exists():
        parser.error("Use a fresh workspace with no artifacts directory")
    (root / "artifacts").mkdir()
    tracking = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment("ai-infrastructure")
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PARAMS_PATH=str(root / "params.yaml"),
               MKL_NUM_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MLFLOW_EXPERIMENT_NAME="infra-training-stages")
    with mlflow.start_run(run_name=f"training-{args.stage}") as run:
        mlflow.log_params({"stage": args.stage, "release": os.environ["BENCHMARK_RELEASE"],
                           "platform": platform.platform(), "threads": 2, "device": "cpu"})
        started = time.perf_counter()
        proc = subprocess.Popen([sys.executable, "-m", f"ride_classifier.pipeline.{args.stage}"], cwd=root, env=env)
        process = psutil.Process(proc.pid)
        peak = 0
        while proc.poll() is None:
            try:
                processes = [process, *process.children(recursive=True)]
                rss = 0
                for member in processes:
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                peak = max(peak, rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.2)
        if proc.wait():
            raise RuntimeError(f"{args.stage} failed with exit code {proc.returncode}")
        values = {"elapsed_s": time.perf_counter() - started, "sampled_process_tree_peak_rss_mib": peak / 2**20}
        mlflow.log_metrics(values)
        filename = "train_run.json" if args.stage == "train" else "fall_risk_run.json"
        source = json.loads((root / "artifacts" / filename).read_text())
        result = dict(values, stage=args.stage, stage_run_id=source["run_id"], benchmark_run_id=run.info.run_id,
                      release=env["BENCHMARK_RELEASE"], sample_interval_s=0.2, platform=platform.platform(),
                      host_ram_mib=psutil.virtual_memory().total / 2**20, threads=2)
        args.output.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(args.output))


if __name__ == "__main__":
    main()
