"""Build the fixed benchmark inputs from the exported production model (E3) and the golden split.

Run locally with the project environment (needs ride_classifier + mlflow). Writes next to this file:
  windows.npy          float32 (n, 6, 300), already normalised with the model's own mean/std
  net_state.pt         {"kwargs": RideCNN constructor kwargs, "state_dict": weights}, so bench.py needs neither
                       the ride_classifier package nor MLflow (it must run in a Python 3.10 Vertex image)
  reference_argmax.npy int64 (n,), argmax per window from CPU torch (baseline for quality_agreement)

The golden split is used read-only: only its sensor readings are windowed, nothing is chosen from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from ride_classifier.data.clean import clean_rides

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--golden", type=Path, default=ROOT / "data" / "processed" / "golden.parquet")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()

    # Same loading path as the API (mlflow.pyfunc), then the wrapped CNNModel.
    model = mlflow.pyfunc.load_model(str(args.model)).unwrap_python_model().model
    readings = clean_rides(pd.read_parquet(args.golden))
    per_ride = model.prepare(readings)
    x = np.concatenate(per_ride.to_list())
    x = ((x - model.mean_) / model.std_).astype(np.float32)

    net = model.net_.eval()
    net_kwargs = {
        "n_channels": int(x.shape[1]),
        "filters": list(model.filters),
        "kernel": int(model.kernel),
        "dropout": float(model.dropout),
        "n_classes": int(net.classify[-1].out_features),
    }
    with torch.no_grad():
        reference = net(torch.from_numpy(x)).argmax(dim=1).numpy().astype(np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "windows.npy", x)
    np.save(args.out_dir / "reference_argmax.npy", reference)
    torch.save({"kwargs": net_kwargs, "state_dict": net.state_dict()}, args.out_dir / "net_state.pt")
    print(f"{len(per_ride)} rides -> windows {x.shape} {x.dtype}; kwargs {net_kwargs}")


if __name__ == "__main__":
    main()
