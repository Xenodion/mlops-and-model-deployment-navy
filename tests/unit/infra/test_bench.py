import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch

BENCH = Path(__file__).resolve().parents[3] / "ai-infrastructure" / "benchmarks" / "bench.py"
CONTRACT_COLUMNS = [
    "experiment", "status", "assumptions", "target", "cpu", "gpu", "n_gpus", "ram_gb", "vram_gb",
    "python", "torch", "onnxruntime", "cuda", "runtime", "batch_size", "concurrency", "warmup", "repeats",
    "latency_mean_ms", "latency_p95_ms", "throughput_windows_s", "peak_ram_mb", "peak_vram_mb",
    "quality_agreement", "quality_macro_f1", "timestamp", "git_sha",
]  # fmt: skip


def load_bench():
    spec = importlib.util.spec_from_file_location("bench", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_columns_match_contract() -> None:
    assert load_bench().COLUMNS == CONTRACT_COLUMNS


def test_p95_on_known_list() -> None:
    assert load_bench().p95(list(range(1, 101))) == np.percentile(np.arange(1, 101), 95)


def test_inference_cpu_writes_one_row_per_batch_size(tmp_path: Path) -> None:
    bench = load_bench()
    kwargs = {"n_channels": 6, "filters": [4, 8], "kernel": 3, "dropout": 0.1, "n_classes": 5}
    net = bench.RideCNN(**kwargs).eval()
    windows = np.random.default_rng(0).normal(size=(4, 6, 300)).astype(np.float32)
    np.save(tmp_path / "w.npy", windows)
    torch.save({"kwargs": kwargs, "state_dict": net.state_dict()}, tmp_path / "s.pt")
    with torch.no_grad():
        np.save(tmp_path / "r.npy", net(torch.from_numpy(windows)).argmax(dim=1).numpy())
    out = tmp_path / "out.csv"
    argv = [
        "inference", "--batch-sizes", "1,2", "--warmup", "2", "--repeats", "3", "--target", "test",
        "--windows", str(tmp_path / "w.npy"), "--state", str(tmp_path / "s.pt"),
        "--reference", str(tmp_path / "r.npy"), "--out", str(out),
    ]  # fmt: skip
    bench.main(argv)
    bench.main(argv)  # appending: the header must not repeat

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert list(rows[0]) == CONTRACT_COLUMNS
    for row in rows:
        assert float(row["latency_mean_ms"]) > 0 and float(row["latency_p95_ms"]) > 0
        assert 0.0 <= float(row["quality_agreement"]) <= 1.0
        assert row["status"] == "measured" and row["peak_vram_mb"] == ""
