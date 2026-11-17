import pytest

from ride_classifier.config import LABELS
from ride_classifier.models.metrics import classification_metrics

GBG500_COUNTS = {"Safe": 416, "Safe overall": 38, "Bad weather": 22, "Tandem": 18, "Reckless": 6}


def test_perfect_predictions_score_one_everywhere():
    assert all(value == 1.0 for value in classification_metrics(LABELS * 2, LABELS * 2).values())


def test_always_answering_safe_gives_the_reference_scores_of_the_plan():
    y_true = [label for label, n in GBG500_COUNTS.items() for _ in range(n)]
    metrics = classification_metrics(y_true, ["Safe"] * len(y_true))
    assert metrics["macro_f1"] == pytest.approx(0.18, abs=0.005)  # docs/PLAN.md §6.4
    assert metrics["accuracy"] == pytest.approx(0.832)
    assert metrics["recall_safe"] == 1.0
    assert metrics["recall_reckless"] == 0.0
