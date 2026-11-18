import pandas as pd
import pytest

from ride_classifier.data.split import make_splits

# Same category sizes as GBG500
COUNTS = {"Safe": 416, "Safe overall": 38, "Bad weather": 22, "Tandem": 18, "Reckless": 6}


@pytest.fixture
def labels():
    rows = [(f"{label}-{i}", label) for label, n in COUNTS.items() for i in range(n)]
    return pd.DataFrame(rows, columns=["ride_id", "label"])


@pytest.fixture
def splits(labels):
    return make_splits(labels, golden_share=0.2, cv_folds=5, seed=42)


def test_one_fifth_of_the_rides_go_to_golden(splits):
    assert (splits["split"] == "golden").sum() == 100


def test_every_category_is_in_train_and_in_golden(splits):
    counts = pd.crosstab(splits["label"], splits["split"])
    assert (counts > 0).all().all()
    assert counts.loc["Reckless", "golden"] == 1


def test_split_is_frozen_whatever_the_file_order(labels, splits):
    reshuffled = make_splits(labels.sample(frac=1, random_state=0), golden_share=0.2, cv_folds=5, seed=42)
    pd.testing.assert_frame_equal(reshuffled, splits)


def test_folds_cover_train_rides_only_and_each_has_one_reckless_ride(splits):
    train = splits[splits["split"] == "train"]
    assert sorted(train["fold"].unique()) == [0, 1, 2, 3, 4]
    assert (splits.loc[splits["split"] == "golden", "fold"] == -1).all()
    assert train[train["label"] == "Reckless"].groupby("fold").size().tolist() == [1, 1, 1, 1, 1]
