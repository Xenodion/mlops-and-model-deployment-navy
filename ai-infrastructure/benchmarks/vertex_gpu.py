"""Paired CPU/CUDA forward passes of the frozen production E3 classifier."""

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import f1_score

from bench import bench_batch, environment, load_net


def main(inputs, output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this paired experiment")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    windows = np.load(inputs / "windows.npy")
    reference = np.load(inputs / "reference_argmax.npy")
    with (inputs / "ride-boundaries.csv").open() as stream:
        rides = list(csv.DictReader(stream))
    labels = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]
    store = Path(tempfile.mkdtemp())
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(store.as_uri())
    mlflow.set_experiment("vertex-gpu")
    try:
        with mlflow.start_run(run_name="E3-paired"):
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()}
            mlflow.log_params({"rounds": 5, "threads": 2, "model": "ride-classifier/2", "warmup": 20, "repeats": 200})
            (output / "environment.json").write_text(json.dumps(dict(environment("cuda", 1), inputs=hashes), indent=2))
            for rnd in range(5):
                order = np.random.default_rng(rnd).permutation(len(windows))
                for device in (("cpu", "cuda") if rnd % 2 == 0 else ("cuda", "cpu")):
                    net = load_net(inputs / "net_state.pt").to(device)
                    with torch.inference_mode():
                        scores = np.concatenate([net(torch.from_numpy(x).to(device)).softmax(1).cpu().numpy()
                                                 for x in np.array_split(windows, 32)])
                    predicted = [labels[scores[int(r["start_inclusive"]):int(r["stop_exclusive"])].mean(0).argmax()]
                                 for r in rides]
                    quality = f1_score([r["label"] for r in rides], predicted, average="macro")
                    agreement = float((scores.argmax(1) == reference).mean())
                    with (output / f"round-{rnd+1}-{device}.csv").open("w", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["batch", "repeat", "ms", "sampled_rss_mib", "peak_allocated_vram_mib",
                                         "window_agreement", "ride_macro_f1"])
                        for batch in (1, 8, 32, 128, 512):
                            x = torch.from_numpy(windows[order[:batch]]).to(device)
                            if device == "cuda":
                                torch.cuda.reset_peak_memory_stats()
                            times, rss, _ = bench_batch(net, x, device, 20, 200)
                            vram = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else ""
                            writer.writerows((batch, i, t, rss, vram, agreement, quality) for i, t in enumerate(times))
                            mlflow.log_metric(f"{device}_batch{batch}_ms", float(np.mean(times)), step=rnd)
                    print(f"Completed round {rnd+1} {device}", flush=True)
    finally:
        archive = shutil.make_archive(str(store) + "-export", "gztar", store)
        shutil.copyfile(archive, output / "mlruns.tar.gz")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
