"""Split rides into train and a frozen golden set, and give each train ride a cross-validation fold."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ride_classifier.config import ID_COLUMN, LABEL_COLUMN

SPLIT_COLUMN = "split"
FOLD_COLUMN = "fold"
TRAIN, GOLDEN = "train", "golden"
NO_FOLD = -1  # golden rides are never used in cross-validation


def make_splits(labels: pd.DataFrame, golden_share: float, cv_folds: int, seed: int) -> pd.DataFrame:
    """One row per ride: ride_id, label, split ("train" or "golden"), fold (0..cv_folds-1, or -1 for golden).

    Both steps are stratified on the label, so the rare categories (6 Reckless rides) appear in every part.
    Rides are sorted by id first: same labels + same seed give the same split whatever the file order,
    which keeps the golden set frozen between runs.
    """
    rides = labels[[ID_COLUMN, LABEL_COLUMN]].sort_values(ID_COLUMN).reset_index(drop=True)

    _, golden_ids = train_test_split(
        rides[ID_COLUMN], test_size=golden_share, stratify=rides[LABEL_COLUMN], random_state=seed
    )
    rides[SPLIT_COLUMN] = TRAIN
    rides.loc[rides[ID_COLUMN].isin(golden_ids), SPLIT_COLUMN] = GOLDEN

    rides[FOLD_COLUMN] = NO_FOLD
    train = rides[rides[SPLIT_COLUMN] == TRAIN]
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for fold, (_, fold_rows) in enumerate(folds.split(train[ID_COLUMN], train[LABEL_COLUMN])):
        rides.loc[train.index[fold_rows], FOLD_COLUMN] = fold
    return rides
