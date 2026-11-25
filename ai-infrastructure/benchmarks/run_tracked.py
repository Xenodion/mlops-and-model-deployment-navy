"""Run the fixed E3 CPU benchmark and record inputs, conditions and results in MLflow."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from ride_classifier.config import MLFLOW_TRACKING_URI, ROOT
from ride_classifier.registry.tracking import check_server, run_tags

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--golden", type=Path, default=ROOT / "data/processed/golden.parquet")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch-sizes", default="1,8,32,128,512")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        parser.error("warmup must be positive and repeats must be at least 2")
    meta = json.loads((args.model / "model_meta.json").read_text(encoding="utf-8"))
    if str(meta.get("version")) != "2" or meta.get("experiment_id") != "E3" or not meta.get("run_id"):
        raise ValueError("Benchmark requires the exported E3 registry version 2 and its source run ID")
    check_server(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ai-infrastructure")
    with mlflow.start_run(run_name="E3-laptop-cpu-baseline") as run, tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        mlflow.set_tags({**run_tags({}), "reference_model_run_id": meta["run_id"], "reference_model_version": "2"})
        mlflow.log_params(
            {
                "device": "cpu",
                "runtime": "torch",
                "warmup": args.warmup,
                "repeats": args.repeats,
                "batch_sizes": args.batch_sizes,
                "concurrency": 1,
            }
        )
        mlflow.log_dict(meta, "inputs/model_meta.json")
        subprocess.run(
            [
                sys.executable,
                str(HERE / "make_windows.py"),
                "--model",
                str(args.model),
                "--golden",
                str(args.golden),
                "--out-dir",
                str(work),
            ],
            check=True,
        )
        fingerprints = {
            path.name: sha256(path)
            for path in [args.golden, work / "windows.npy", work / "net_state.pt", work / "reference_argmax.npy"]
        }
        mlflow.log_dict(fingerprints, "inputs/sha256.json")
        result = work / "baseline-results.csv"
        subprocess.run(
            [
                sys.executable,
                str(HERE / "bench.py"),
                "inference",
                "--runtime",
                "torch",
                "--device",
                "cpu",
                "--target",
                "laptop",
                "--experiment",
                "E3-laptop-cpu",
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
                "--batch-sizes",
                args.batch_sizes,
                "--windows",
                str(work / "windows.npy"),
                "--state",
                str(work / "net_state.pt"),
                "--reference",
                str(work / "reference_argmax.npy"),
                "--out",
                str(result),
            ],
            check=True,
        )
        rows = pd.read_csv(result)
        for row in rows.itertuples(index=False):
            for key in ("latency_mean_ms", "latency_p95_ms", "throughput_windows_s", "peak_ram_mb"):
                mlflow.log_metric(f"batch_{row.batch_size}_{key}", float(getattr(row, key)))
        mlflow.log_artifact(str(result), "results")
        print(f"Run {run.info.run_id}: raw results logged to MLflow under results/baseline-results.csv")


if __name__ == "__main__":
    main()
