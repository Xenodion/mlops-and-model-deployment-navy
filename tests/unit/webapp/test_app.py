"""The whole page, run by Streamlit's test runner (no browser), against an API that isn't there."""

from streamlit.testing.v1 import AppTest

from ride_classifier.config import ROOT


def run_page(monkeypatch, examples_dir) -> AppTest:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    monkeypatch.setenv("EXAMPLES_DIR", str(examples_dir))  # the real examples are course data, absent in CI
    return AppTest.from_file(str(ROOT / "webapp" / "app.py"), default_timeout=60).run()


def fake_examples(tmp_path, ride):
    """One fake example ride, in the format of `make webapp-examples` (with its true category)."""
    ride.drop(columns="ride_id").assign(label="Tandem").to_parquet(tmp_path / "tandem.parquet", index=False)
    return tmp_path


def test_the_page_shows_an_example_ride_even_when_the_api_is_down(monkeypatch, tmp_path, ride):
    page = run_page(monkeypatch, fake_examples(tmp_path, ride))
    assert not page.exception
    assert "API unreachable" in page.sidebar.error[0].value
    assert [m.label for m in page.metric] == ["Readings", "Duration", "Missing readings"]


def test_without_example_rides_the_page_says_how_to_get_them(monkeypatch, tmp_path):
    page = run_page(monkeypatch, tmp_path)
    assert not page.exception
    assert "Upload a file" in page.info[0].value and "make webapp-examples" in page.info[0].value


def test_classifying_without_an_api_shows_an_error_not_a_crash(monkeypatch, tmp_path, ride):
    page = run_page(monkeypatch, fake_examples(tmp_path, ride))
    page.button[0].click().run()
    assert not page.exception
    assert any("API unreachable" in e.value for e in page.error)


def fake_api(monkeypatch, prediction):
    """Both models answering, without any network: health and predict as the real API returns them."""
    import api_client

    status = {"status": "ok", "model_version": "2", "fall_risk_version": "1", "reckless_threshold": 0.5}
    monkeypatch.setattr(api_client, "health", lambda url: {**status, "fall_risk_threshold": 0.6})
    monkeypatch.setattr(api_client, "predict", lambda url, payload: (prediction, {"x-process-time-ms": "120"}))


def test_the_answer_shows_both_models_then_both_along_the_ride(monkeypatch, tmp_path, ride, prediction):
    fake_api(monkeypatch, prediction)
    page = run_page(monkeypatch, fake_examples(tmp_path, ride))
    assert any("**Model 2** · fall risk: v1" in m.value for m in page.sidebar.markdown)

    page.button[0].click().run()
    assert not page.exception
    assert [s.value for s in page.subheader] == ["Model 1 · Ride type", "Model 2 · Fall risk", "Along the ride"]
    metrics = {m.label: m.value for m in page.metric}
    assert metrics["Category of the whole ride"] == "Reckless"
    assert metrics["Riskiest moment"] == "0.95 at 0:20"
    assert metrics["Alert moments (risk ≥ 0.6)"] == "2 of 4"
    # 2 sensor charts + model 1's bar chart, then along the ride: model 1 so far and model 2's risk
    assert len(page.get("vega_lite_chart")) == 3 + 2


def test_without_model_2_the_page_says_so_and_still_shows_model_1(monkeypatch, tmp_path, ride, prediction):
    fake_api(monkeypatch, {**prediction, "fall_risk": None})
    page = run_page(monkeypatch, fake_examples(tmp_path, ride))
    page.button[0].click().run()
    assert not page.exception
    assert any("model 2 isn't loaded" in i.value for i in page.info)
    assert len(page.get("vega_lite_chart")) == 3 + 1  # along the ride: model 1 only
