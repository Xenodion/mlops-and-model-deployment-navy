"""Run the preregistered CPU-only ONNX comparison (FP32 and optional dynamic INT8) and track it in MLflow."""

import argparse
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
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmarks"
sys.path.insert(0, str(BENCH))
from bench import bench_batch, load_net  # noqa: E402


def _make_ort_predictor(model_path: Path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])

    def predict(x):
        return torch.from_numpy(session.run(None, {"windows": x.numpy()})[0])

    return predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-int8", action="store_true", help="Also export and benchmark dynamic INT8 ONNX")
    args = parser.parse_args(argv)
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
    run_name = "E3-cpu-onnx-fp32-int8" if args.include_int8 else "E3-cpu-onnx-fp32"
    with mlflow.start_run(run_name=run_name) as run, tempfile.TemporaryDirectory() as tmp:
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
                "include_int8": args.include_int8,
            }
        )
        export = Path(tmp) / "model.onnx"
        torch.onnx.export(
            net,
            torch.from_numpy(windows[:1]),
            str(export),
            input_names=["windows"],
            output_names=["logits"],
            dynamic_axes={"windows": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        runtimes = {"pytorch": net, "onnx": _make_ort_predictor(export)}
        if args.include_int8:
            int8_path = Path(tmp) / "model_int8.onnx"
            quantize_dynamic(str(export), str(int8_path), weight_type=QuantType.QUInt8)
            runtimes["onnx_int8"] = _make_ort_predictor(int8_path)
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
        runtime_names = list(runtimes)
        for rnd in range(3):
            order = runtime_names if rnd % 2 == 0 else list(reversed(runtime_names))
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
        if args.include_int8:
            meta["int8_window_argmax_agreement"] = float(
                np.mean(quality["pytorch"]["scores"].argmax(1) == quality["onnx_int8"]["scores"].argmax(1))
            )
            meta["int8_max_probability_difference"] = float(
                np.max(np.abs(quality["pytorch"]["scores"] - quality["onnx_int8"]["scores"]))
            )
            meta["onnx_int8_ride_macro_f1"] = quality["onnx_int8"]["f1"]
        (output / "environment.json").write_text(json.dumps(meta, indent=2))
        metrics = {
            "window_argmax_agreement": agreement,
            "max_probability_difference": max_diff,
            "pytorch_ride_macro_f1": quality["pytorch"]["f1"],
            "onnx_ride_macro_f1": quality["onnx"]["f1"],
        }
        if args.include_int8:
            metrics["int8_window_argmax_agreement"] = meta["int8_window_argmax_agreement"]
            metrics["int8_max_probability_difference"] = meta["int8_max_probability_difference"]
            metrics["onnx_int8_ride_macro_f1"] = meta["onnx_int8_ride_macro_f1"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(output), artifact_path="onnx-comparison")
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
