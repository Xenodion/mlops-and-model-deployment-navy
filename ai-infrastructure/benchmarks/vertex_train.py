"""Time the release E3 stage in fresh processes on one Vertex CPU/GPU host."""

import json
import os
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path


def configure_source(root):
    tracking = root / "src/ride_classifier/registry/tracking.py"
    source = tracking.read_text()
    for old, new in [
        ("except OSError, subprocess.CalledProcessError:", "except (OSError, subprocess.CalledProcessError):"),
        ("except OSError, KeyError, IndexError, TypeError:", "except (OSError, KeyError, IndexError, TypeError):"),
    ]:
        if source.count(old) != 1:
            raise ValueError(f"Unexpected tracking source: {old}")
        source = source.replace(old, new)
    tracking.write_text(source)
    path = root / "src/ride_classifier/models/cnn.py"
    source = path.read_text()
    for old, new in [
        (
            "self.net_ = RideCNN(x.shape[1], self.filters, self.kernel, self.dropout)",
            'device = __import__("os").environ["BENCH_DEVICE"]\n        '
            "self.net_ = RideCNN(x.shape[1], self.filters, self.kernel, self.dropout).to(device)",
        ),
        ("optimizer = torch.optim.Adam", "loss_fn = loss_fn.to(device)\n        optimizer = torch.optim.Adam"),
        ("loss_fn(self.net_(xb), yb)", "loss_fn(self.net_(xb.to(device)), yb.to(device))"),
        ("self.net_.eval()\n        return self", "self.net_.cpu().eval()\n        return self"),
    ]:
        if source.count(old) != 1:
            raise ValueError(f"Unexpected training source: {old}")
        source = source.replace(old, new)
    path.write_text(source)


def stage(root, output):
    import mlflow
    import torch

    from ride_classifier.config import load_params
    from ride_classifier.pipeline import evaluate, train

    device = os.environ["BENCH_DEVICE"]
    if not torch.cuda.is_available():
        raise RuntimeError("Paired experiment requires CUDA")
    torch.set_num_threads(2)
    params = load_params(root / "params.yaml")
    store = root / "tracking"
    store.mkdir(exist_ok=True)
    uri = store.as_uri()
    mlflow.set_tracking_uri(uri)
    started = time.perf_counter()
    metrics = train.run(params, root, uri, "vertex-e3-training")
    elapsed = time.perf_counter() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    peak_vram = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
    if device == "cuda" and peak_vram <= 0:
        raise RuntimeError("No CUDA memory allocated during training")
    golden = evaluate.run(params, root, uri)
    run = json.loads((root / params["train"]["run_path"]).read_text())
    summary = dict(
        device=device,
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        elapsed_s=elapsed,
        peak_process_rss_mib=peak_rss,
        peak_allocated_vram_mib=peak_vram,
        cv_macro_f1=metrics["cv_macro_f1"],
        golden=golden,
        stage_run_id=run["run_id"],
    )
    with mlflow.start_run(run_id=run["run_id"]):
        mlflow.log_metrics(
            {"training_elapsed_s": elapsed, "process_peak_rss_mib": peak_rss, "cuda_peak_allocated_mib": peak_vram}
        )
        mlflow.log_param("benchmark_device", device)
    output.write_text(json.dumps(summary, indent=2, default=float))


def main(bundle, output):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with tarfile.open(bundle) as archive:
            archive.extractall(root, filter="data")
        configure_source(root)
        (root / "artifacts").mkdir(exist_ok=True)
        env = dict(
            os.environ,
            PYTHONPATH=str(root / "src"),
            PARAMS_PATH=str(root / "params.yaml"),
            MLFLOW_ALLOW_FILE_STORE="true",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
        try:
            for rnd in range(1, 4):
                for device in ("cuda", "cpu") if rnd % 2 else ("cpu", "cuda"):
                    env["BENCH_DEVICE"] = device
                    subprocess.run(
                        [sys.executable, __file__, "stage", str(root), str(output / f"{rnd}-{device}.json")],
                        check=True,
                        env=env,
                        cwd=root,
                    )
        finally:
            if (root / "tracking").exists():
                archive = shutil.make_archive(str(root / "mlruns"), "gztar", root / "tracking")
                shutil.copyfile(archive, output / "mlruns.tar.gz")


if __name__ == "__main__":
    if sys.argv[1] == "stage":
        stage(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
