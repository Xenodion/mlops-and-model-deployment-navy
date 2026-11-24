"""Both models on one time axis (docs/PLAN.md API-4): model 1's category so far above model 2's fall risk.

Two charts with the same time scale (0 to the ride's duration) and the same left margin, so a moment sits at the same
place in both. Kept apart from app.py so they can be tested without a browser. Colours: the project's visual charter
(Okabe-Ito, colour-vision safe), copied here because the web app doesn't import our package.
"""

from __future__ import annotations

import math

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


def _time(duration_s: float, title: str | None = None, field: str = "time_s", axis: bool = True) -> alt.X:
    scale = alt.Scale(domain=[0, duration_s], nice=False)
    return alt.X(f"{field}:Q", title=title, scale=scale, axis=alt.Axis() if axis else None)


def _probability(title: str, top: float = 1.0) -> alt.Y:
    return alt.Y("value:Q", title=title, scale=alt.Scale(domain=[0, top]), axis=alt.Axis(minExtent=Y_AXIS_WIDTH))


def _top_of_scale(values: pd.Series) -> float:
    """The probability axis stops just above the highest line (next 0.1 step), to spread the changes over the height.
    It still starts at 0, so the zoom doesn't exaggerate the differences."""
    return min(1.0, math.ceil(round(float(values.max()) * 10, 6)) / 10 + 0.1)  # round: 0.6 * 10 must give 6


def category_chart(so_far: pd.DataFrame, duration_s: float) -> alt.Chart:
    """Model 1: the probability of each category so far, after each 30 s window (the top line is its answer so far)."""
    categories = [category for category in CATEGORY_COLORS if category in so_far]
    long = so_far.melt(id_vars=["time_s"], value_vars=categories, var_name="ride type", value_name="value")
    return (
        alt.Chart(long, title="Model 1 · ride type so far (30 s windows, one every 15 s)")
        .mark_line(point=True)
        .encode(
            x=_time(duration_s),
            y=_probability("Probability so far", top=_top_of_scale(long["value"])),
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
        .properties(height=400)
    )


def answer_strips(answer: pd.DataFrame, true_label: str | None, duration_s: float) -> alt.LayerChart:
    """Under model 1's lines: its answer so far (one block per change) and, for example rides, the true category.

    The truth is one block for the whole ride: GBG500 labels a ride, not its moments. The rows sit on a numeric axis
    (row centres 0.5, 1.5) so that it takes the same room as the probability axes and the time axes line up.
    """
    model_row = 1 if true_label else 0
    rows = [answer.assign(row=model_row, who="model 1 so far")]
    if true_label:
        truth = {"start_s": 0.0, "end_s": duration_s, "category": true_label, "row": 0, "who": "true category"}
        rows.append(pd.DataFrame([truth]))
    data = pd.concat(rows, ignore_index=True)
    data["low"], data["high"], data["centre"] = data["row"] + 0.08, data["row"] + 0.92, data["row"] + 0.5
    data["middle"] = (data["start_s"] + data["end_s"]) / 2
    n_rows = model_row + 1
    names = "datum.value > 1 ? 'model 1' : 'truth'" if true_label else "'model 1'"
    y_axis = alt.Axis(
        values=[row + 0.5 for row in range(n_rows)], labelExpr=names, minExtent=Y_AXIS_WIDTH, grid=False, ticks=False
    )
    blocks = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=_time(duration_s, field="start_s", axis=False),
            x2="end_s:Q",
            y=alt.Y("low:Q", title="Answer", scale=alt.Scale(domain=[0, n_rows]), axis=y_axis),
            y2="high:Q",
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=list(CATEGORY_COLORS), range=list(CATEGORY_COLORS.values())),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("who:N", title=" "),
                "category:N",
                alt.Tooltip("start_s:Q", title="from (s)", format=".0f"),
                alt.Tooltip("end_s:Q", title="to (s)", format=".0f"),
            ],
        )
    )
    labels = (  # the category's name inside the blocks wide enough to hold it
        alt.Chart(data)
        .transform_filter(f"datum.end_s - datum.start_s > {0.1 * duration_s}")
        .mark_text(color="#1A1A1A", fontWeight="bold")
        .encode(x=_time(duration_s, field="middle", axis=False), y=alt.Y("centre:Q"), text="category:N")
    )
    return alt.layer(blocks, labels).properties(height=34 * n_rows)


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
        height=220
    )
