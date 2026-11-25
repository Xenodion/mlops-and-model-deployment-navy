"""Verify frozen benchmark inputs against the local export and split manifests."""

import hashlib
import json

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score

from make_windows import HERE, ROOT
from ride_classifier.config import LABELS
from ride_classifier.data.clean import clean_rides
from ride_classifier.pipeline.train_fall_risk import paired_auc


def main():
    manifest_path = HERE / "input-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Frozen file changed: {path.name}")

    model = mlflow.pyfunc.load_model(str(ROOT / "model")).unwrap_python_model().model
    golden = ROOT / "data/processed/golden.parquet"
    rides = model.prepare(clean_rides(pd.read_parquet(golden)))
    frozen = np.load(HERE / "windows.npy", mmap_mode="r")
    reference = np.load(HERE / "reference_argmax.npy")
    checkpoint = torch.load(HERE / "net_state.pt", map_location="cpu")["state_dict"]
    for name, weight in model.net_.state_dict().items():
        torch.testing.assert_close(weight.cpu(), checkpoint[name], rtol=0, atol=0)

    labels = pd.read_csv(ROOT / "artifacts/golden_predictions.csv").set_index("ride_id")
    assert labels.index.is_unique and set(labels.index) == set(rides.index)
    start = 0
    boundaries = []
    predictions = []
    for ride_id, windows in rides.items():
        stop = start + len(windows)
        np.testing.assert_array_equal((windows - model.mean_) / model.std_, frozen[start:stop])
        probs = model._window_probas(windows)
        np.testing.assert_array_equal(probs.argmax(axis=1), reference[start:stop])
        predictions.append(LABELS[probs.mean(axis=0).argmax()])
        boundaries.append((ride_id, start, stop, labels.at[ride_id, "label"]))
        start = stop
    assert start == len(frozen) == len(reference)
    table = pd.DataFrame(boundaries, columns=["ride_id", "start_inclusive", "stop_exclusive", "label"])

    groups_path = ROOT / "data/processed/fall_groups.csv"
    groups = pd.read_csv(groups_path)
    assert groups.groupby("fall_group_id").group_split.nunique().eq(1).all()
    held = groups.query("group_split == 'held_out'")
    scores = pd.read_csv(ROOT / "artifacts/fall_risk_heldout_scores.csv")
    joined = scores.merge(held, on=["ride_id", "fall_group_id"], validate="many_to_one")
    assert len(joined) == len(scores)
    assert set(joined.gap_s) == {5, 10}
    metrics = json.loads((ROOT / "artifacts/fall_risk_metrics.json").read_text())
    by_gap = {}
    for gap, rows in joined.groupby("gap_s"):
        observed = paired_auc(rows, rows.risk.to_numpy())
        prefix = "held_out_" if gap == 5 else "held_out_gap10s_"
        for key, value in observed.items():
            np.testing.assert_allclose(value, metrics[prefix + key])
        by_gap[str(gap)] = dict(rows=len(rows), **observed)

    path = HERE / "ride-boundaries.csv"
    table.to_csv(path, index=False)
    manifest["local_verification"] = {
        "export_run": yaml.safe_load((ROOT / "model/MLmodel").read_text())["run_id"],
        "golden_macro_f1": f1_score(table.label, predictions, average="macro"),
        "changed_vs_saved_ride_predictions": int(np.sum(labels.loc[rides.index, "predicted"].to_numpy() != predictions)),
        "fall_metrics_from_saved_scores": by_gap,
        "mapping_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, groups_path]},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["local_verification"], indent=2))


if __name__ == "__main__":
    main()
