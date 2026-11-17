import pandas as pd
import pytest

from ride_classifier.data.load import DataValidationError, check_rides_match_labels, load_labels, validate_rides


def write_labels(tmp_path, rows):
    path = tmp_path / "labels.csv"
    pd.DataFrame(rows, columns=["RIDE_ID", "LABEL"]).to_csv(path, index=False)
    return path


def test_labels_get_the_project_column_names(tmp_path):
    labels = load_labels(write_labels(tmp_path, [("r1", "Safe"), ("r2", "Reckless")]))
    assert list(labels.columns) == ["ride_id", "label"]


def test_unknown_label_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="unknown labels"):
        load_labels(write_labels(tmp_path, [("r1", "Dangerous")]))


def test_ride_with_two_labels_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="several labels"):
        load_labels(write_labels(tmp_path, [("r1", "Safe"), ("r1", "Tandem")]))


def test_missing_sensor_column_is_rejected(ride):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_rides(ride.drop(columns="ry"))


def test_rides_and_labels_must_match(ride_factory):
    rides = pd.concat([ride_factory(ride_id="r1"), ride_factory(ride_id="r2")])
    labels = pd.DataFrame({"ride_id": ["r1", "r3"], "label": ["Safe", "Safe"]})
    with pytest.raises(DataValidationError, match="rides without a label: 1, labels without a ride: 1"):
        check_rides_match_labels(rides, labels)
