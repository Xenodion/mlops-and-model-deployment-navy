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


def roc_figure(curves: dict[str, tuple[Sequence[int], Sequence[float]]], title: str) -> Figure:
    """Classic ROC curves, one per entry: name -> (true labels, scores). The legend shows each AUC."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, (y_true, score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC {roc_auc_score(y_true, score):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="chance (AUC 0.50)")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title, xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
