"""Cumulative prediction of a ride as it unfolds: running mean of window probabilities, scored at elapsed times."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ride_classifier.config import LABELS


@dataclass(frozen=True)
class Timeline:
    """One ride's predictions over time. Row k of the arrays belongs to window k."""

    ride_id: str
    end_s: np.ndarray  # (n_windows,) elapsed seconds when each window is complete
    window_probas: np.ndarray  # (n_windows, n_labels)
    cumulative: np.ndarray  # (n_windows, n_labels) running mean of window_probas
    danger: np.ndarray | None = None  # (n_windows,) z-scored reconstruction error (danger detector only)
    cumulative_danger: np.ndarray | None = None  # (n_windows,) running mean of danger

    @property
    def label(self) -> str:
        """Final category: argmax of the cumulative probabilities after the last window."""
        return LABELS[int(np.argmax(self.cumulative[-1]))]

    @property
    def confidence(self) -> float:
        """Final probability of that category."""
        return float(np.max(self.cumulative[-1]))


def cumulative_mean(window_probas: np.ndarray) -> np.ndarray:
    """Running mean along axis 0: row k is the mean of rows 0..k."""
    if len(window_probas) == 0:
        raise ValueError("window_probas is empty")
    return np.cumsum(window_probas, axis=0) / np.arange(1, len(window_probas) + 1)[:, None]


def window_end_times(n_windows: int, window_s: float, stride_s: float) -> np.ndarray:
    """Elapsed seconds at which each window is complete: k * stride_s + window_s."""
    return np.arange(n_windows) * stride_s + window_s


def cumulative_at(timeline: Timeline, t_s: float) -> np.ndarray | None:
    """Cumulative probabilities using only the windows complete by t_s (end_s <= t_s); math.inf gives the last row.

    None if no window is complete yet: the model has not answered (returning the first window would use data
    recorded after t_s).
    """
    completed = int(np.searchsorted(timeline.end_s, t_s, side="right"))
    return timeline.cumulative[completed - 1] if completed else None


def danger_at(timeline: Timeline, t_s: float) -> float | None:
    """Cumulative danger using only the windows complete by t_s; None before the first window ends or if unscored."""
    if timeline.cumulative_danger is None:
        return None
    completed = int(np.searchsorted(timeline.end_s, t_s, side="right"))
    return float(timeline.cumulative_danger[completed - 1]) if completed else None


def f1_vs_elapsed(timelines: Mapping[str, Timeline], labels: pd.Series, times_s: Sequence[float]) -> pd.DataFrame:
    """Macro-F1 of the cumulative prediction at each elapsed time.

    Rides shorter than t keep their last cumulative value and are counted. If some ride has no complete window
    at t (t shorter than the window), macro_f1 is NaN: the model has not answered yet. Columns: t_s (inf = whole
    ride), t_label, macro_f1, n_rides, n_rides_complete (rides whose last window ends at or after t; all at inf).
    """
    ride_ids = list(timelines)
    y_true = [labels.loc[ride_id] for ride_id in ride_ids]
    rows = []
    for t in times_s:
        answers = [cumulative_at(timelines[r], t) for r in ride_ids]
        answered = all(a is not None for a in answers)
        y_pred = [LABELS[int(np.argmax(a))] for a in answers] if answered else []
        rows.append(
            {
                "t_s": t,
                "t_label": "end" if math.isinf(t) else f"{t:g}",
                "macro_f1": (
                    f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0) if answered else math.nan
                ),
                "n_rides": len(ride_ids),
                "n_rides_complete": sum(math.isinf(t) or bool(timelines[r].end_s[-1] >= t) for r in ride_ids),
            }
        )
    return pd.DataFrame(rows)
