"""Run the preregistered CPU-only ONNX comparison and track it in MLflow."""

import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmarks"
sys.path.insert(0, str(BENCH))
from bench import bench_batch, load_net  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri or not uri.startswith(("http://", "https://")):
        raise RuntimeError("Set MLFLOW_TRACKING_URI to the shared tracking server before running.")
    mlflow.set_tracking_uri(uri)
    mlflow.search_experiments(max_results=1)  # Fail closed before any experiment.
    mlflow.set_experiment("ai-infrastructure")
    inputs = [BENCH / n for n in ("net_state.pt", "windows.npy", "ride-boundaries.csv")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    windows = np.load(inputs[1])
    with inputs[2].open(newline="") as stream:
        rides = list(csv.DictReader(stream))
    labels = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]
    torch.set_num_threads(2)
    net = load_net(inputs[0])
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    with mlflow.start_run(run_name="E3-cpu-onnx-fp32") as run, tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params(
            {
                "release": "c193360",
                "model": "ride-classifier/2",
                "threads": 2,
                "rounds": 3,
                "warmups": 20,
                "repeats": 200,
                "batches": "1,8,32,128",
                "device": "cpu",
            }
        )
        export = str(Path(tmp) / "model.onnx")
        torch.onnx.export(
            net,
            torch.from_numpy(windows[:1]),
            export,
            input_names=["windows"],
            output_names=["logits"],
            dynamic_axes={"windows": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(export, sess_options=options, providers=["CPUExecutionProvider"])

        def onnx_predict(x):
            return torch.from_numpy(session.run(None, {"windows": x.numpy()})[0])

        runtimes = {"pytorch": net, "onnx": onnx_predict}
        quality = {}
        with torch.inference_mode():
            for name, predict in runtimes.items():
                logits = torch.cat([predict(torch.from_numpy(chunk)) for chunk in np.array_split(windows, 32)])
                scores = logits.softmax(1).numpy()
                predicted = [
                    labels[scores[int(r["start_inclusive"]) : int(r["stop_exclusive"])].mean(0).argmax()] for r in rides
                ]
                quality[name] = {
                    "scores": scores,
                    "f1": f1_score([r["label"] for r in rides], predicted, average="macro"),
                }
        agreement = float(np.mean(quality["pytorch"]["scores"].argmax(1) == quality["onnx"]["scores"].argmax(1)))
        max_diff = float(np.max(np.abs(quality["pytorch"]["scores"] - quality["onnx"]["scores"])))
        rows = []
        for rnd in range(3):
            order = ("pytorch", "onnx") if rnd % 2 == 0 else ("onnx", "pytorch")
            indices = np.random.default_rng(rnd).permutation(len(windows))
            for name in order:
                for batch in (1, 8, 32, 128):
                    x = torch.from_numpy(np.ascontiguousarray(windows[indices[:batch]]))
                    times, rss, _ = bench_batch(runtimes[name], x, "cpu", 20, 200)
                    rows.extend(
                        {
                            "runtime": name,
                            "round": rnd + 1,
                            "batch": batch,
                            "repeat": i + 1,
                            "latency_ms": t,
                            "process_rss_mib": rss,
                        }
                        for i, t in enumerate(times)
                    )
                    mlflow.log_metric(f"{name}_batch{batch}_mean_ms", float(np.mean(times)), step=rnd)
        with (output / "latencies.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        meta = {
            "mlflow_run_id": run.info.run_id,
            "input_sha256": hashes,
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "threads": 2,
            "window_argmax_agreement": agreement,
            "max_probability_difference": max_diff,
            "pytorch_ride_macro_f1": quality["pytorch"]["f1"],
            "onnx_ride_macro_f1": quality["onnx"]["f1"],
            "memory_note": "Process RSS includes both loaded runtimes, not isolated runtime memory.",
        }
        (output / "environment.json").write_text(json.dumps(meta, indent=2))
        mlflow.log_metrics(
            {
                "window_argmax_agreement": agreement,
                "max_probability_difference": max_diff,
                "pytorch_ride_macro_f1": quality["pytorch"]["f1"],
                "onnx_ride_macro_f1": quality["onnx"]["f1"],
            }
        )
        mlflow.log_artifacts(str(output), artifact_path="onnx-comparison")
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
