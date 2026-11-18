"""Charts logged to MLflow by the train and evaluate stages."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

from ride_classifier.config import LABELS


def confusion_figure(y_true: Sequence[str], y_pred: Sequence[str], title: str) -> Figure:
    """Rows = true category, columns = predicted category: the diagonal is what the model got right."""
    figure = Figure(figsize=(7, 6), layout="constrained")
    axes = figure.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=LABELS, ax=axes, colorbar=False, xticks_rotation=30)
    axes.set_title(title)
    return figure
