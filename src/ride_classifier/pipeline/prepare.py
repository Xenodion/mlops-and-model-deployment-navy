"""DVC stage `prepare`: raw files -> clean readings split into train / golden, plus the ride-level split table.

make pipeline                                       (all stages, only what changed)
uv run python -m ride_classifier.pipeline.prepare   (this stage alone, outside DVC)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ride_classifier.config import ID_COLUMN, LABEL_COLUMN, LABELS, ROOT, load_params
from ride_classifier.data.clean import clean_rides
from ride_classifier.data.load import check_rides_match_labels, load_labels, load_rides
from ride_classifier.data.split import GOLDEN, SPLIT_COLUMN, TRAIN, make_splits

logger = logging.getLogger(__name__)


def run(params: dict, root: Path = ROOT) -> pd.DataFrame:
    """Run the stage with the `data` section of params.yaml (paths relative to `root`). Returns the split table."""
    rides = load_rides(root / params["raw_rides"])
    labels = load_labels(root / params["raw_labels"])
    check_rides_match_labels(rides, labels)

    clean = clean_rides(rides)
    splits = make_splits(labels, params["golden_share"], params["cv_folds"], params["seed"])
    splits = splits.join(clean.groupby(ID_COLUMN).size().rename("n_readings"), on=ID_COLUMN)

    labelled = clean.merge(
        splits[[ID_COLUMN, LABEL_COLUMN, SPLIT_COLUMN]], on=ID_COLUMN, how="left", validate="many_to_one"
    )
    for split, path_key in ((TRAIN, "train_path"), (GOLDEN, "golden_path")):
        path = root / params[path_key]
        path.parent.mkdir(parents=True, exist_ok=True)
        labelled[labelled[SPLIT_COLUMN] == split].drop(columns=SPLIT_COLUMN).to_parquet(path, index=False)
    splits.to_csv(root / params["splits_path"], index=False)

    counts = pd.crosstab(splits[LABEL_COLUMN], splits[SPLIT_COLUMN]).reindex(LABELS)
    logger.info("rides per label and split:\n%s", counts.to_string())
    return splits


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(load_params()["data"])


if __name__ == "__main__":
    main()
