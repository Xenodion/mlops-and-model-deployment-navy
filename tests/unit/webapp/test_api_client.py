import io

import httpx2 as httpx
import numpy as np
import pytest
from api_client import (
    ApiError,
    answer_blocks,
    category_timeline,
    clock,
    fall_risk_frame,
    fall_risk_summary,
    health,
    predict,
    read_ride,
    ride_payload,
)

API = "http://api.test"


def fake_api(status: int, body: dict) -> httpx.MockTransport:
    """An API that always answers `status` with `body`, without any network."""
    return httpx.MockTransport(lambda request: httpx.Response(status, json=body, headers={"X-Request-ID": "r1"}))


def unreachable(request):
    raise httpx.ConnectError("connection refused", request=request)


def test_a_csv_or_parquet_ride_is_read_in_time_order(ride):
    shuffled = ride.drop(columns="ride_id").sample(frac=1, random_state=0)
    parquet = io.BytesIO()
    shuffled.to_parquet(parquet, index=False)
    assert read_ride("r.csv", shuffled.to_csv(index=False).encode())["time_index"].is_monotonic_increasing
    assert read_ride("r.parquet", parquet.getvalue())["time_index"].is_monotonic_increasing


def test_a_file_without_a_channel_is_refused_with_its_name(ride):
    with pytest.raises(ValueError, match="missing columns: ry"):
        read_ride("r.csv", ride.drop(columns="ry").to_csv(index=False).encode())


def test_the_payload_has_one_array_per_channel_and_null_for_gaps(ride):
    ride.loc[3, "ax"] = np.nan
    payload = ride_payload(ride, ride_id="r1")
    assert payload["ride_id"] == "r1" and len(payload["ax"]) == len(payload["time_index"]) == len(ride)
    assert payload["ax"][3] is None


def test_a_prediction_and_its_headers_come_back():
    answer = {"label": "Safe", "probabilities": {"Safe": 1.0}, "reckless_score": 0.0, "is_reckless": False}
    prediction, headers = predict(API, {}, transport=fake_api(200, answer))
    assert prediction["label"] == "Safe" and headers["x-request-id"] == "r1"


def test_a_rejected_ride_names_the_wrong_field():
    errors = {"detail": [{"loc": ["body", "time_index"], "msg": "List should have at least 300 items"}]}
    with pytest.raises(ApiError, match="time_index: List should have at least 300 items"):
        predict(API, {}, transport=fake_api(422, errors))


def test_an_unreachable_api_gives_a_readable_error():
    with pytest.raises(ApiError, match="API unreachable"):
        health(API, transport=httpx.MockTransport(unreachable))
    with pytest.raises(ApiError, match="API unreachable"):
        predict(API, {}, transport=httpx.MockTransport(unreachable))


def test_model_1_along_the_ride_is_its_running_answer_so_far(prediction):
    so_far = category_timeline(prediction)
    assert so_far["time_s"].tolist() == [30.0, 45.0, 60.0]
    assert so_far["category"].tolist() == ["Safe", "Safe", "Reckless"]  # Safe at first, Reckless once enough evidence
    assert so_far["Reckless"].tolist() == pytest.approx([0.1, 0.35, 0.4667], abs=1e-3)  # running mean, not the window


def test_without_a_timeline_there_is_nothing_to_draw(prediction):
    assert category_timeline({**prediction, "timeline": {"windows": []}}) is None


def test_model_2_gives_the_riskiest_moment_and_the_alert_moments(prediction):
    risk = fall_risk_frame(prediction)
    assert risk.columns.tolist() == ["time_s", "risk", "alert"] and len(risk) == 4
    assert fall_risk_summary(risk) == {"max_risk": 0.95, "at_s": 20.0, "alerts": 2, "moments": 4}


def test_without_model_2_there_is_no_fall_risk(prediction):
    assert fall_risk_frame({**prediction, "fall_risk": None}) is None


def test_times_read_as_minutes_and_seconds():
    assert (clock(0), clock(9.6), clock(155), clock(600)) == ("0:00", "0:10", "2:35", "10:00")


def test_model_1_s_answer_so_far_becomes_one_block_per_change(prediction):
    blocks = answer_blocks(category_timeline(prediction), duration_s=75.0)
    assert blocks.to_dict("records") == [
        {"start_s": 30.0, "end_s": 60.0, "category": "Safe"},  # no answer before the first 30 s window
        {"start_s": 60.0, "end_s": 75.0, "category": "Reckless"},  # the last answer lasts until the end of the ride
    ]
