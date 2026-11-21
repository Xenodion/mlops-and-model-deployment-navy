"""The web app isn't part of our package: make its modules importable, as `streamlit run webapp/app.py` does."""

import sys

from ride_classifier.config import ROOT

WEBAPP_DIR = ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))
