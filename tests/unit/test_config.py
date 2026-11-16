from ride_classifier.config import ID_COLUMN, LABELS, SENSOR_COLUMNS, TIME_COLUMN, load_params


def test_labels_are_the_five_categories_in_a_fixed_order():
    assert LABELS == ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]


def test_params_file_loads():
    assert "data" in load_params()


def test_fake_ride_follows_the_ride_schema(ride):
    assert list(ride.columns) == [ID_COLUMN, TIME_COLUMN, *SENSOR_COLUMNS]
    assert ride[TIME_COLUMN].diff().dropna().eq(100).all()
