"""The whole page, run by Streamlit's test runner (no browser), against an API that isn't there."""

from streamlit.testing.v1 import AppTest

from ride_classifier.config import ROOT


def run_page(monkeypatch) -> AppTest:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    return AppTest.from_file(str(ROOT / "webapp" / "app.py"), default_timeout=60).run()


def test_the_page_shows_an_example_ride_even_when_the_api_is_down(monkeypatch):
    page = run_page(monkeypatch)
    assert not page.exception
    assert "API unreachable" in page.sidebar.error[0].value
    assert [m.label for m in page.metric] == ["Readings", "Duration", "Missing readings"]


def test_classifying_without_an_api_shows_an_error_not_a_crash(monkeypatch):
    page = run_page(monkeypatch)
    page.button[0].click().run()
    assert not page.exception
    assert any("API unreachable" in e.value for e in page.error)
