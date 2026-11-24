"""Both models on one time axis (docs/PLAN.md API-4): model 1's category so far above model 2's fall risk.

Two charts with the same time scale (0 to the ride's duration) and the same left margin, so a moment sits at the same
place in both. Kept apart from app.py so they can be tested without a browser. Colours: the project's visual charter
(Okabe-Ito, colour-vision safe), copied here because the web app doesn't import our package.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

CATEGORY_COLORS = {
    "Safe": "#009E73",
    "Safe overall": "#56B4E9",
    "Bad weather": "#E69F00",
    "Tandem": "#CC79A7",
    "Reckless": "#D55E00",
}
RISK_COLOR, ALERT_COLOR, THRESHOLD_COLOR = "#0072B2", "#D55E00", "#7F7F7F"
Y_AXIS_WIDTH = 50  # the same on both charts, so their time axes line up


def _time(duration_s: float, title: str | None = None) -> alt.X:
    return alt.X("time_s:Q", title=title, scale=alt.Scale(domain=[0, duration_s], nice=False))


def _probability(title: str) -> alt.Y:
    return alt.Y("value:Q", title=title, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(minExtent=Y_AXIS_WIDTH))


def category_chart(so_far: pd.DataFrame, duration_s: float) -> alt.Chart:
    """Model 1: the probability of each category so far, after each 30 s window (the top line is its answer so far)."""
    categories = [category for category in CATEGORY_COLORS if category in so_far]
    long = so_far.melt(id_vars=["time_s"], value_vars=categories, var_name="ride type", value_name="value")
    return (
        alt.Chart(long, title="Model 1 · ride type so far (30 s windows, one every 15 s)")
        .mark_line(point=True)
        .encode(
            x=_time(duration_s),
            y=_probability("Probability so far"),
            color=alt.Color(
                "ride type:N",
                scale=alt.Scale(domain=list(CATEGORY_COLORS), range=list(CATEGORY_COLORS.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("time_s:Q", title="time (s)", format=".0f"),
                "ride type:N",
                alt.Tooltip("value:Q", title="probability so far", format=".2f"),
            ],
        )
        .properties(height=220)
    )


def fall_risk_chart(risk: pd.DataFrame, threshold: float, duration_s: float) -> alt.LayerChart:
    """Model 2: the fall risk of the last 10 s, every 5 s, the alert threshold (dashed) and the alert moments (dots)."""
    data = risk.rename(columns={"risk": "value"})
    line = (
        alt.Chart(data)
        .mark_line(color=RISK_COLOR)
        .encode(
            x=_time(duration_s, "Time in the ride (s)"),
            y=_probability("Fall risk"),
            tooltip=[alt.Tooltip("time_s:Q", title="time (s)", format=".0f"), alt.Tooltip("value:Q", format=".2f")],
        )
    )
    alerts = (
        alt.Chart(data)
        .transform_filter(alt.datum.alert)
        .mark_point(color=ALERT_COLOR, filled=True, size=45)
        .encode(x=_time(duration_s), y=_probability("Fall risk"))
    )
    rule = (
        alt.Chart(pd.DataFrame({"value": [threshold]}))
        .mark_rule(color=THRESHOLD_COLOR, strokeDash=[6, 4])
        .encode(y=_probability("Fall risk"))
    )
    return alt.layer(line, alerts, rule, title="Model 2 · fall risk of the last 10 s (one every 5 s)").properties(
        height=170
    )
