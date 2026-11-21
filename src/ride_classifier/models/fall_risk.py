"""Model 2, fall risk: a 10 s window of sensor readings -> probability that a fall follows (ADR 001, ADR 004).

Training windows of a fall ride end in the `pre_fall_s` seconds that stop `gap_s` before the first topple reading;
negatives are the windows ending at the same elapsed moments of the comparison rides of the same fall group.
Each window is summarised with the same statistics as model 1's baseline, then scored by a logistic regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ride_classifier.config import ID_COLUMN, SAMPLING_PERIOD_MS, SENSOR_COLUMNS, TIME_COLUMN
from ride_classifier.features.stats import ride_features

GROUP, END_MS, LABEL = "fall_group_id", "end_ms", "label"


def window_ends(fall_ms: int, window_s: float, stride_s: float, pre_fall_s: float, gap_s: float) -> np.ndarray:
    """End times (ms) of the windows of the segment [fall - gap - pre_fall, fall - gap], whole windows only."""
    last = fall_ms - gap_s * 1000
    first = max(window_s * 1000, last - pre_fall_s * 1000)
    return np.arange(first, last + 1, stride_s * 1000).astype(int)


def window_table(
    ride_last_ms: pd.Series, fall_ms: pd.Series, groups: pd.DataFrame, params: dict, gap_s: float
) -> pd.DataFrame:
    """One row per (ride, window end): ride_id, fall_group_id, end_ms, label (1 = window of the fall ride).

    `ride_last_ms`: last reading time per ride; `fall_ms`: first topple time per fall ride (index = fall ride id =
    group id); `groups`: ride_id, fall_group_id. A comparison ride too short for a moment gives no row for it.
    """
    rows = []
    for ride, group in groups[[ID_COLUMN, GROUP]].itertuples(index=False):
        if group not in fall_ms.index or ride not in ride_last_ms.index:
            continue
        ends = window_ends(int(fall_ms[group]), params["window_s"], params["stride_s"], params["pre_fall_s"], gap_s)
        rows += [(ride, group, int(e), int(ride == group)) for e in ends if e <= ride_last_ms[ride]]
    return pd.DataFrame(rows, columns=[ID_COLUMN, GROUP, END_MS, LABEL])


def window_features(readings: pd.DataFrame, table: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Statistics of each window of `table` (same row order). `readings`: clean rides, 10 Hz, sorted by time."""
    size = round(window_s * 1000 / SAMPLING_PERIOD_MS)
    by_ride = {ride: r[[TIME_COLUMN, *SENSOR_COLUMNS]].set_index(TIME_COLUMN) for ride, r in readings.groupby(ID_COLUMN)}
    parts = []
    for i, (ride, end) in enumerate(zip(table[ID_COLUMN], table[END_MS], strict=True)):
        win = by_ride[ride].loc[:end].tail(size).reset_index(drop=True)
        parts.append(win.assign(**{ID_COLUMN: i}))
    features = ride_features(pd.concat(parts, ignore_index=True))
    return features.drop(columns=["duration_s"]).reset_index(drop=True)


class FallRiskModel:
    """Standardised statistics -> logistic regression; `predict_risk` gives P(fall follows) per window."""

    def __init__(self, c: float, seed: int):
        self.pipeline = Pipeline(
            [("scale", StandardScaler()), ("logit", LogisticRegression(C=c, max_iter=2000, random_state=seed))]
        )
        self.columns_: list[str] = []

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> FallRiskModel:
        self.columns_ = list(features.columns)
        self.pipeline.fit(features.to_numpy(), labels.to_numpy())
        return self

    def predict_risk(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(features[self.columns_].to_numpy())[:, 1]
