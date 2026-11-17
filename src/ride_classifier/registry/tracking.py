"""What every MLflow run records to answer "which code, which data?" months later (lab 1, docs/PLAN.md §8)."""

from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import yaml

from ride_classifier.config import ROOT


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return None  # no git (e.g. inside a container)


def dvc_md5(pointer: Path) -> str:
    """Fingerprint of a DVC-tracked file, read from its .dvc pointer file."""
    try:
        return yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"]
    except OSError, KeyError, IndexError, TypeError:
        return "unknown"


def run_tags(data_pointers: dict[str, Path]) -> dict[str, str]:
    """git_commit + git_dirty (which code) and one md5 per raw data file (which data).

    git_dirty=True means the code had uncommitted changes: such a run cannot be rebuilt and is never registered.
    """
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": "unknown" if status is None else str(bool(status)),
        **{name: dvc_md5(path) for name, path in data_pointers.items()},
    }


def flatten(params: dict, prefix: str = "") -> dict[str, object]:
    """{"train": {"baseline": {"C": 0.05}}} -> {"train.baseline.C": 0.05}, for mlflow.log_params."""
    flat: dict[str, object] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def check_server(tracking_uri: str, timeout_s: float = 3.0) -> None:
    """Fail fast with a clear message instead of MLflow retrying for minutes when the server is down."""
    if not tracking_uri.startswith("http"):
        return  # local store (tests): nothing to reach
    try:
        urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=timeout_s)
    except OSError as exc:
        raise SystemExit(f"MLflow is not reachable at {tracking_uri} ({exc}). Start it with `make dev-up`.") from exc
