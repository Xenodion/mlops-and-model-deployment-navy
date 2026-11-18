import pandas as pd

from ride_classifier.config import LABELS
from ride_classifier.pipeline.prepare import run


def test_prepare_writes_train_golden_and_the_split_table(tmp_path, ride_factory):
    labels = [(f"{label}-{i}", label) for label in LABELS for i in range(10)]  # 50 fake rides
    rides = pd.concat([ride_factory(n_samples=50, ride_id=ride_id, seed=i) for i, (ride_id, _) in enumerate(labels)])
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    rides.to_parquet(raw / "rides.parquet", index=False)
    pd.DataFrame(labels, columns=["RIDE_ID", "LABEL"]).to_csv(raw / "labels.csv", index=False)
    params = {
        "raw_rides": "data/raw/rides.parquet",
        "raw_labels": "data/raw/labels.csv",
        "train_path": "data/processed/train.parquet",
        "golden_path": "data/processed/golden.parquet",
        "splits_path": "data/processed/splits.csv",
        "golden_share": 0.2,
        "cv_folds": 5,
        "seed": 42,
    }

    splits = run(params, root=tmp_path)

    train = pd.read_parquet(tmp_path / params["train_path"])
    golden = pd.read_parquet(tmp_path / params["golden_path"])
    assert set(train["ride_id"]).isdisjoint(golden["ride_id"])
    assert len(train) + len(golden) == len(rides)
    assert {"label", "time_index", "ax"} <= set(train.columns)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / params["splits_path"]), splits, check_dtype=False)
