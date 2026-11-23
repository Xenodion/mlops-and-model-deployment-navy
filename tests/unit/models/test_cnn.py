import numpy as np
import pytest
import torch

from ride_classifier.config import LABELS
from ride_classifier.models.cnn import CNNModel, RideCNN, class_weights

SMALL = dict(window_s=30, stride_s=15, filters=[8, 16], kernel=5, dropout=0.0, epochs=8, batch_size=32,
             learning_rate=3e-3, weight_decay=0.0, augment=True, seed=0)  # fmt: skip


@pytest.fixture(scope="module")
def data(labelled_rides):
    readings, labels = labelled_rides
    model = CNNModel(**SMALL)
    return model.prepare(readings), labels.set_index("ride_id")["label"]


def test_the_network_turns_windows_into_one_score_per_category():
    scores = RideCNN(n_channels=6, filters=[8, 16], kernel=5, dropout=0.0)(torch.zeros(4, 6, 300))
    assert scores.shape == (4, len(LABELS))


def test_class_weights_balanced_softened_or_none():
    y = np.repeat(np.arange(len(LABELS)), [800, 50, 50, 50, 50])  # one big category, four small ones
    balanced, sqrt, none = (class_weights(y, mode) for mode in ("balanced", "sqrt", "none"))
    assert balanced[1] / balanced[0] == pytest.approx(16)  # 800 / 50
    assert sqrt[1] / sqrt[0] == pytest.approx(4)  # its square root
    assert (none == 1).all()
    with pytest.raises(ValueError):
        class_weights(y, "heavy")


def test_same_seed_same_model(data):
    inputs, labels = data
    first, second = (CNNModel(**{**SMALL, "epochs": 1}).fit(inputs, labels) for _ in range(2))
    np.testing.assert_array_equal(first.predict_proba(inputs).to_numpy(), second.predict_proba(inputs).to_numpy())


def test_it_learns_the_fake_categories(data):
    inputs, labels = data  # the fake categories differ by how much they shake: easy to learn
    model = CNNModel(**SMALL).fit(inputs, labels)
    accuracy = (model.predict_proba(inputs).idxmax(axis=1) == labels.loc[inputs.index]).mean()
    assert accuracy >= 0.5  # random guessing: 0.2 (it reaches 0.6 after 8 epochs)
    assert model.history_[-1] < model.history_[0]  # the training loss went down


def test_the_scaling_is_learned_from_the_training_rides_only(data):
    inputs, labels = data
    train_ids = inputs.index[::2]
    model = CNNModel(**{**SMALL, "epochs": 1}).fit(inputs.loc[train_ids], labels)
    expected = np.concatenate(inputs.loc[train_ids].to_list()).mean(axis=(0, 2))
    np.testing.assert_allclose(model.mean_.ravel(), expected, rtol=1e-5)


def test_predict_timeline_final_cumulative_equals_predict_proba(data):
    inputs, labels = data
    model = CNNModel(**{**SMALL, "epochs": 2}).fit(inputs, labels)
    timelines = model.predict_timeline(inputs)
    probas = model.predict_proba(inputs)
    assert set(timelines) == set(inputs.index)
    for ride_id, timeline in timelines.items():
        np.testing.assert_allclose(timeline.cumulative[-1], probas.loc[ride_id].to_numpy(), atol=1e-6)
        assert timeline.end_s[0] == SMALL["window_s"]
