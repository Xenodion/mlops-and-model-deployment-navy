"""The web app isn't part of our package: make its modules importable, as `streamlit run webapp/app.py` does."""

import sys

import pytest

from ride_classifier.config import ROOT

WEBAPP_DIR = ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))


def _probabilities(safe: float, reckless: float) -> dict[str, float]:
    rest = (1 - safe - reckless) / 3
    return {"Safe": safe, "Safe overall": rest, "Bad weather": rest, "Tandem": rest, "Reckless": reckless}


@pytest.fixture
def prediction() -> dict:
    """An answer of /v1/predict with both models, in the API's format: 3 model-1 windows (30 s, every 15 s) that
    turn from Safe to Reckless, and 4 model-2 windows (10 s, every 5 s), 2 of them above the alert threshold."""
    return {
        "ride_id": "r1",
        "label": "Reckless",
        "probabilities": _probabilities(0.3, 0.47),
        "reckless_score": 0.47,
        "is_reckless": True,
        "model_version": "2",
        "n_readings": 600,
        "timeline": {
            "window_s": 30,
            "stride_s": 15,
            "windows": [
                {"end_s": 30.0, "danger": None, "probabilities": _probabilities(0.6, 0.1)},
                {"end_s": 45.0, "danger": None, "probabilities": _probabilities(0.2, 0.6)},
                {"end_s": 60.0, "danger": None, "probabilities": _probabilities(0.1, 0.7)},
            ],
        },
        "fall_risk": {
            "window_s": 10,
            "stride_s": 5,
            "threshold": 0.6,
            "retrospective": True,
            "note": "computed on a recorded ride: for analysis, not a forecast",
            "windows": [
                {"end_s": 10.0, "risk": 0.1, "alert": False},
                {"end_s": 15.0, "risk": 0.3, "alert": False},
                {"end_s": 20.0, "risk": 0.95, "alert": True},
                {"end_s": 25.0, "risk": 0.7, "alert": True},
            ],
        },
    }
