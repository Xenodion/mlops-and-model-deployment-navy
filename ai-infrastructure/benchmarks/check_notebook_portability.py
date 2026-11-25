"""Run the analysis notebook from a minimal public-evidence-only copy."""

import shutil
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "ai-infrastructure" / "benchmarks"
ARCH = ROOT / "ai-infrastructure" / "architecture"
OPT = ROOT / "ai-infrastructure" / "optimization" / "results"

with tempfile.TemporaryDirectory(prefix="cloud-notebook-") as temp:
    dest = Path(temp) / "clone" / "ai-infrastructure"
    b = dest / "benchmarks"
    (b / "results").mkdir(parents=True)
    for name in ("benchmark.ipynb", "analyze_evidence.py", "sizing.csv"):
        shutil.copy2(BENCH / name, b / name)
    for name in (
        "vertex-2031761767163494400",
        "vertex-2097063961760366592",
        "vertex-2985610094491729920",
    ):
        shutil.copytree(
            BENCH / "results" / name,
            b / "results" / name,
            ignore=shutil.ignore_patterns("mlruns.tar.gz", "vertex-logs.json"),
        )
    for name in ("training-fall-risk-local-1.json", "training-fall-risk-local-1-metrics.json"):
        shutil.copy2(BENCH / "results" / name, b / "results" / name)
    a = dest / "architecture"
    a.mkdir()
    for name in ("current-system.png", "measured-replay.png"):
        shutil.copy2(ARCH / name, a / name)
    shutil.copytree(OPT, dest / "optimization" / "results")
    notebook = nbformat.read(b / "benchmark.ipynb", as_version=4)
    NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(b)}}).execute()
    expected = ("live_latency.png", "post_ride_latency.png", "training_duration.png", "onnx_latency.png")
    assert all((b / "figures" / name).is_file() for name in expected)
    print("Notebook ran from a minimal evidence-only copy; representative figures were regenerated.")
