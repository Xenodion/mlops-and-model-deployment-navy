"""Classification metrics shared by the train and evaluate stages (docs/PLAN.md §6.4)."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from ride_classifier.config import LABELS


def metric_key(label: str) -> str:
    """'Safe overall' -> 'safe_overall', for metric names."""
    return label.lower().replace(" ", "_")


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """macro_f1 (the primary metric), balanced accuracy, accuracy, and the recall of each category.

    Accuracy is for reference only: always answering "Safe" already scores 0.83.
    """
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        **{f"recall_{metric_key(label)}": float(r) for label, r in zip(LABELS, recalls, strict=True)},
    }
