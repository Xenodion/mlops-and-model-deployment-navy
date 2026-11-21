import numpy as np
import pandas as pd
import pytest
import torch

from ride_classifier.config import LABELS
from ride_classifier.models.autoencoder import DangerModel, RideAutoencoder
from ride_classifier.models.timeline import danger_at

PARAMS = dict(
    window_s=5,
    stride_s=2.5,
    filters=[4, 8, 8],
    kernel=7,
    epochs=2,
    batch_size=8,
    learning_rate=0.001,
    weight_decay=0.0001,
    head_C=0.05,
    class_weight="sqrt",
    seed=0,
)


def make_inputs(n_rides: int = 6, n_windows: int = 5, length: int = 50) -> pd.Series:
    rng = np.random.default_rng(0)
    rides = {f"r{i}": rng.normal(size=(n_windows, 6, length)).astype(np.float32) for i in range(n_rides)}
    return pd.Series(rides, dtype=object)


@pytest.mark.parametrize("length", [50, 100, 300, 600])
def test_autoencoder_output_has_input_shape_for_window_length(length):
    net = RideAutoencoder(6, [4, 8, 8], 7).eval()
    x = torch.randn(3, 6, length)
    assert net(x).shape == x.shape
    assert net.embed(x).shape == (3, 8)


def test_fit_without_labels_works_and_predict_proba_raises():
    model = DangerModel(**PARAMS).fit(make_inputs())
    assert model.danger_windows(make_inputs()["r0"]).shape == (5,)
    with pytest.raises(RuntimeError):
        model.predict_proba(make_inputs())


def test_fit_with_labels_gives_a_head_with_one_row_per_ride():
    inputs = make_inputs()
    labels = pd.Series(["Safe", "Safe", "Tandem", "Reckless", "Safe", "Tandem"], index=inputs.index)
    probas = DangerModel(**PARAMS).fit(inputs, labels).predict_proba(inputs)
    assert list(probas.columns) == LABELS
    np.testing.assert_allclose(probas.sum(axis=1), 1.0)


def test_danger_scores_do_not_depend_on_labels():
    inputs = make_inputs()
    labels = pd.Series(["Safe", "Safe", "Tandem", "Reckless", "Safe", "Tandem"], index=inputs.index)
    without = DangerModel(**PARAMS).fit(inputs).danger_windows(inputs["r1"])
    with_labels = DangerModel(**PARAMS).fit(inputs, labels).danger_windows(inputs["r1"])
    np.testing.assert_array_equal(without, with_labels)


def test_timeline_last_cumulative_danger_is_mean_of_window_dangers():
    inputs = make_inputs()
    timeline = DangerModel(**PARAMS).fit(inputs).predict_timeline(inputs)["r2"]
    assert timeline.cumulative_danger[-1] == pytest.approx(timeline.danger.mean())
    assert np.isnan(timeline.window_probas).all()  # no head


def test_danger_at_gives_no_answer_before_first_window_ends():
    inputs = make_inputs()
    timeline = DangerModel(**PARAMS).fit(inputs).predict_timeline(inputs)["r0"]
    assert danger_at(timeline, timeline.end_s[0] - 0.1) is None
    assert danger_at(timeline, timeline.end_s[0]) == pytest.approx(timeline.danger[0])
    assert danger_at(timeline, float("inf")) == pytest.approx(timeline.cumulative_danger[-1])
