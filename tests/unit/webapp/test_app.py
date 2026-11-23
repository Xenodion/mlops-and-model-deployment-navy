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
    assert "make webapp-examples" in page.info[0].value


def test_classifying_without_an_api_shows_an_error_not_a_crash(monkeypatch, tmp_path, ride):
    page = run_page(monkeypatch, fake_examples(tmp_path, ride))
    page.button[0].click().run()
    assert not page.exception
    assert any("API unreachable" in e.value for e in page.error)
