"""The shared time axis: both charts span the same time (0 to the ride's duration) and are valid Vega-Lite."""

import pandas as pd
import pytest
from api_client import category_timeline, fall_risk_frame
from charts import _top_of_scale, category_chart, fall_risk_chart


def x_domains(spec: dict) -> list:
    """Every time-axis domain in a chart spec (single or layered)."""
    layers = spec.get("layer", [spec])
    return [layer["encoding"]["x"]["scale"]["domain"] for layer in layers if "x" in layer.get("encoding", {})]


def test_both_charts_share_the_ride_s_time_axis(prediction):
    top = category_chart(category_timeline(prediction), duration_s=75.0).to_dict()  # to_dict validates the spec
    bottom = fall_risk_chart(fall_risk_frame(prediction), threshold=0.6, duration_s=75.0).to_dict()
    assert x_domains(top) == [[0, 75.0]]
    assert set(map(tuple, x_domains(bottom))) == {(0, 75.0)}


def test_the_fall_chart_shows_the_threshold_and_the_alert_moments(prediction):
    layers = fall_risk_chart(fall_risk_frame(prediction), threshold=0.6, duration_s=75.0).to_dict()["layer"]
    assert [layer["mark"]["type"] for layer in layers] == ["line", "point", "rule"]
    assert layers[2]["mark"]["strokeDash"] == [6, 4]  # dashed threshold


def test_model_1_s_axis_zooms_on_its_lines_but_still_starts_at_zero(prediction):
    low, top = category_chart(category_timeline(prediction), 75.0).to_dict()["encoding"]["y"]["scale"]["domain"]
    assert (low, top) == (0, pytest.approx(0.7))  # highest line 0.6 (Safe, first window): next step + 0.1
    assert [_top_of_scale(pd.Series([v])) for v in (0.42, 0.6, 0.95)] == pytest.approx([0.6, 0.7, 1.0])
