"""Classification metrics shared by the train and evaluate stages (docs/PLAN.md §6.4)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
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


def reference_scores(
    y_true: Sequence[str], train_labels: Sequence[str], seed: int, draws: int = 200
) -> dict[str, float]:
    """Macro-F1 of the two trivial references, on the same rides as the model:

    - majority: always answering the most frequent train category ("Safe")
    - random: guessing each category with its train frequency, averaged over `draws` guesses so that
      one lucky draw doesn't decide the gate
    """
    y_true = list(y_true)
    shares = pd.Series(list(train_labels)).value_counts(normalize=True)  # most frequent first
    majority = classification_metrics(y_true, [shares.index[0]] * len(y_true))["macro_f1"]
    rng = np.random.default_rng(seed)
    random_f1 = [
        f1_score(
            y_true,
            rng.choice(shares.index.to_numpy(), size=len(y_true), p=shares.to_numpy()),
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        for _ in range(draws)
    ]
    return {"majority": majority, "random": float(np.mean(random_f1))}
