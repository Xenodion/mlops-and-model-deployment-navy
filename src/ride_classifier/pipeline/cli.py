"""Start-up shared by the pipeline stages when they run from the command line (`dvc repro`)."""

from __future__ import annotations

import logging
import sys


def setup() -> None:
    """UTF-8 output and INFO logs.

    MLflow prints emojis; on Windows, output written to a pipe (e.g. under `dvc repro`) defaults to cp1252,
    which cannot encode them and crashes the stage at the very end of an MLflow run.
    """
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
