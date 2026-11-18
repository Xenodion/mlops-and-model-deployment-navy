import pytest

from ride_classifier.config import LABELS
from ride_classifier.models.metrics import classification_metrics, reference_scores

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


def test_references_match_always_safe_and_random_guessing():
    y_true = [label for label, n in GBG500_COUNTS.items() for _ in range(n)]
    references = reference_scores(y_true, train_labels=y_true, seed=42)
    assert references["majority"] == pytest.approx(0.18, abs=0.005)
    assert references["random"] == pytest.approx(0.20, abs=0.02)  # random guessing ≈ 1 / 5 categories
