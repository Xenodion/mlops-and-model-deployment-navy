"""Standalone benchmark tool for the ride classifier (contract: specs/001-final-deliverable/contracts/).

Imports only the stdlib, numpy, torch and psutil (onnxruntime lazily), and stays Python 3.10 compatible so the same
file runs on the laptop, Colab/Kaggle and in a Vertex image. Inputs come from make_windows.py.

    python bench.py inference --runtime torch --device cpu --batch-sizes 1,8,32 --out results/laptop-cpu.csv

--gpus 2 wraps the net in torch.nn.DataParallel over 2 GPUs (inference only, data-parallel split of each batch).
Batch sizes larger than the number of windows are filled by sampling windows WITH replacement (fixed seed
SAMPLING_SEED); agreement is computed on the windows actually used.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn

SAMPLING_SEED = 0
HERE = Path(__file__).resolve().parent

COLUMNS = [
    "experiment", "status", "assumptions", "target",
    "cpu", "gpu", "n_gpus", "ram_gb", "vram_gb",
    "python", "torch", "onnxruntime", "cuda",
    "runtime", "batch_size", "concurrency", "warmup", "repeats",
    "latency_mean_ms", "latency_p95_ms", "throughput_windows_s",
    "peak_ram_mb", "peak_vram_mb",
    "quality_agreement", "quality_macro_f1",
    "timestamp", "git_sha",
]  # fmt: skip


class RideCNN(nn.Module):
    """Same architecture as ride_classifier.models.cnn.RideCNN (copied so this file stays standalone)."""

    def __init__(self, n_channels: int, filters: list[int], kernel: int, dropout: float, n_classes: int = 5):
        super().__init__()
        blocks, channels = [], n_channels
        for out_channels in filters:
            blocks += [
                nn.Conv1d(channels, out_channels, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.features(x))


def load_net(state_path: Path) -> nn.Module:
    """Rebuild the network from net_state.pt ({"kwargs", "state_dict"}), in eval mode."""
    blob = torch.load(state_path, map_location="cpu", weights_only=True)
    net = RideCNN(**blob["kwargs"])
    net.load_state_dict(blob["state_dict"])
    return net.eval()


def p95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), 95))


def _git_sha() -> str:
    """Short commit hash, or "" outside a git checkout (no `git`, not a repo)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE)
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def environment(device: str, n_gpus: int) -> dict[str, object]:
    use_cuda = device == "cuda" and torch.cuda.is_available()
    try:
        import onnxruntime

        ort_version = onnxruntime.__version__
    except ImportError:
        ort_version = ""
    return {
        "cpu": _cpu_name(),
        "gpu": torch.cuda.get_device_name(0) if use_cuda else "",
        "n_gpus": n_gpus if use_cuda else 0,
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if use_cuda else "",
        "python": platform.python_version(),
        "torch": torch.__version__.split("+")[0],
        "onnxruntime": ort_version,
        "cuda": torch.version.cuda or "",
    }


def pick_indices(n_windows: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Windows used for one batch size: the first ones, or a with-replacement sample when there are too few."""
    if batch_size <= n_windows:
        return np.arange(batch_size)
    return rng.choice(n_windows, size=batch_size, replace=True)


def bench_batch(predict, x: torch.Tensor, device: str, warmup: int, repeats: int):
    """Times predict(x): warm-up, then `repeats` timed calls. Returns latencies (ms), peak RSS (MB), last argmax."""
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    proc = psutil.Process(os.getpid())
    peak_rss = proc.memory_info().rss
    out = None
    with torch.no_grad():
        for _ in range(warmup):
            out = predict(x)
        sync()
        latencies = []
        for _ in range(repeats):
            start = time.perf_counter()
            out = predict(x)
            sync()
            latencies.append((time.perf_counter() - start) * 1000)
            peak_rss = max(peak_rss, proc.memory_info().rss)
    return latencies, peak_rss / 1024**2, out.argmax(dim=1).cpu().numpy()


def run_inference(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.runtime in ("onnx", "onnx-int8"):
        raise NotImplementedError(f"runtime {args.runtime!r} is added by issue #5; only 'torch' is available now")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    if args.gpus == 2 and (args.device != "cuda" or torch.cuda.device_count() < 2):
        raise SystemExit("--gpus 2 needs --device cuda and at least 2 visible GPUs")

    windows = np.load(args.windows)
    reference = np.load(args.reference) if args.reference.exists() else None
    net = load_net(args.state).to(args.device)
    if args.gpus == 2:
        net = nn.DataParallel(net, device_ids=[0, 1])
    env = environment(args.device, args.gpus)
    rng = np.random.default_rng(SAMPLING_SEED)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 (UTC alias is 3.11+)
    sha = _git_sha()

    rows = []
    for batch_size in [int(b) for b in args.batch_sizes.split(",")]:
        idx = pick_indices(len(windows), batch_size, rng)
        x = torch.from_numpy(np.ascontiguousarray(windows[idx])).to(args.device)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        latencies, peak_ram, predicted = bench_batch(net, x, args.device, args.warmup, args.repeats)
        mean_ms = float(np.mean(latencies))
        agreement = float(np.mean(predicted == reference[idx])) if reference is not None else ""
        peak_vram = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if args.device == "cuda" else ""
        rows.append(
            {
                "experiment": args.experiment, "status": "measured", "assumptions": "", "target": args.target,
                **env,
                "runtime": args.runtime if args.gpus == 1 else f"{args.runtime}-dp{args.gpus}",
                "batch_size": batch_size, "concurrency": 1, "warmup": args.warmup, "repeats": args.repeats,
                "latency_mean_ms": round(mean_ms, 4), "latency_p95_ms": round(p95(latencies), 4),
                "throughput_windows_s": round(batch_size / (mean_ms / 1000), 2),
                "peak_ram_mb": round(peak_ram, 1), "peak_vram_mb": peak_vram,
                "quality_agreement": agreement, "quality_macro_f1": "",
                "timestamp": stamp, "git_sha": sha,
            }
        )  # fmt: skip
    return rows


def write_rows(rows: list[dict[str, object]], out: Path) -> None:
    """Append to `out`; the header is written only when the file is new or empty."""
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists() or out.stat().st_size == 0
    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def print_csv(rows: list[dict[str, object]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
    print("@@CSV")
    writer.writeheader()
    writer.writerows(rows)
    print("@@END")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    inf = sub.add_parser("inference", help="latency / throughput / memory per batch size")
    inf.add_argument("--runtime", choices=["torch", "onnx", "onnx-int8"], default="torch")
    inf.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    inf.add_argument("--gpus", type=int, choices=[1, 2], default=1, help="2 = nn.DataParallel over 2 GPUs")
    inf.add_argument("--batch-sizes", default="1,8,32,128,512")
    inf.add_argument("--warmup", type=int, default=20)
    inf.add_argument("--repeats", type=int, default=100)
    inf.add_argument("--target", default="", help="free text, e.g. laptop, vertex-1xa100")
    inf.add_argument("--experiment", default="")
    inf.add_argument("--windows", type=Path, default=HERE / "windows.npy")
    inf.add_argument("--state", type=Path, default=HERE / "net_state.pt")
    inf.add_argument("--reference", type=Path, default=HERE / "reference_argmax.npy")
    inf.add_argument("--out", type=Path, default=None)
    inf.add_argument("--print-csv", action="store_true", help="also print the rows between @@CSV and @@END")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = run_inference(args)
    if args.out is not None:
        write_rows(rows, args.out)
    if args.print_csv:
        print_csv(rows)


if __name__ == "__main__":
    main()
