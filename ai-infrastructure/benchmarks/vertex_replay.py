"""Run the published joint API on a Vertex CPU worker using a private replay bundle."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import mlflow
import pandas as pd
import requests
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Enable instrumented classifier placement")
    parser.add_argument("--clients", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        with tarfile.open(args.bundle) as archive:
            archive.extractall(work, filter="data")
        store = work / "tracking"
        store.mkdir()
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(store.as_uri())
        mlflow.set_experiment("release-replay")
        args.output.mkdir(parents=True, exist_ok=True)
        try:
            with mlflow.start_run(run_name="joint-api-vertex-cpu"):
                manifest = json.loads((work / "release.json").read_text())
                mlflow.log_params(manifest)
                mlflow.log_params({"platform": platform.platform(), "torch": torch.__version__, "threads": 2})
                env = dict(os.environ, MODEL_URI=str(work / "ride-classifier"), FALL_RISK_URI=str(work / "fall-risk"),
                           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PREDICTION_LOG_PATH=str(work / "requests.jsonl"))
                command = ["-m", "uvicorn", "ride_classifier.api.main:app",
                           "--host", "127.0.0.1", "--port", "18172", "--no-access-log"]
                if args.device:
                    env.update(BENCH_DEVICE=args.device, BENCH_DEVICE_REPORT=str(args.output / "device.json"))
                    command = [str(Path(__file__).with_name("gpu_api.py"))]
                    mlflow.log_param("classifier_device", args.device)
                api = subprocess.Popen([sys.executable, *command], env=env,
                                       stdout=subprocess.DEVNULL)
                try:
                    for _ in range(90):
                        if api.poll() is not None:
                            raise RuntimeError("API exited before readiness")
                        try:
                            health = requests.get("http://127.0.0.1:18172/health", timeout=2).json()
                            if health.get("model_loaded") and health.get("fall_risk_loaded"):
                                break
                        except requests.RequestException:
                            pass
                        time.sleep(1)
                    else:
                        raise RuntimeError("Both models must be ready")
                    replay = work / "ai-infrastructure/benchmarks/replay_release.py"
                    result = subprocess.check_output([sys.executable, str(replay)], text=True)
                    metrics = json.loads(result)
                    if len(args.clients) not in (1, args.rounds):
                        raise ValueError("Provide one concurrency or one per round")
                    mlflow.log_param("clients_by_round", str(args.clients))
                    for rnd in range(1, args.rounds + 1):
                        clients = args.clients[(rnd - 1) % len(args.clients)]
                        raw = work / f"round-{rnd}.csv"
                        observed = subprocess.check_output([sys.executable, str(replay), "--timings", str(raw),
                                                            "--clients", str(clients)], text=True)
                        if json.loads(observed) != metrics:
                            raise ValueError("Replay quality changed between rounds")
                        shutil.copyfile(raw, args.output / raw.name)
                        rows = pd.read_csv(raw)
                        seconds = float(raw.with_suffix(".seconds").read_text())
                        summary = {"clients": clients, "elapsed_s": seconds, "requests_s": len(rows) / seconds}
                        (args.output / f"round-{rnd}.json").write_text(json.dumps(summary))
                        mlflow.log_metric("requests_s", summary["requests_s"], step=rnd)
                        for route, part in rows.groupby("route"):
                            key = "live" if route.endswith("/window") else "post_ride"
                            mlflow.log_metric(f"{key}_p95_ms", part.latency_ms.quantile(.95), step=rnd)
                    if args.rounds:
                        peak = next(line for line in Path(f"/proc/{api.pid}/status").read_text().splitlines()
                                    if line.startswith("VmHWM:"))
                        metrics["api_peak_rss_mib"] = int(peak.split()[1]) / 1024
                        total = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                                     if line.startswith("MemTotal:"))
                        metrics["worker_ram_mib"] = int(total.split()[1]) / 1024
                    mlflow.log_metrics(metrics)
                    (args.output / "replay.json").write_text(json.dumps({**manifest, **metrics}, indent=2))
                    print("REPLAY_RESULT", result.strip(), flush=True)
                finally:
                    api.terminate()
                    api.wait(timeout=30)
        finally:
            tracking = shutil.make_archive(str(work / "tracking"), "gztar", store)
            shutil.copyfile(tracking, args.output / "mlruns.tar.gz")


if __name__ == "__main__":
    main()
