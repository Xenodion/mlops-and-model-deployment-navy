import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ride_classifier.config import LABELS
from ride_classifier.viz.palette import CATEGORY_COLORS
from ride_classifier.viz.theme import save_fig


def test_save_fig_writes_png_and_svg_when_id_registered(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    png, svg = save_fig(fig, "F36", "palette_test", out_dir=tmp_path)
    plt.close(fig)
    assert png.exists() and svg.exists()
    assert png.name == "F36_palette_test.png" and svg.suffix == ".svg"


def test_save_fig_raises_when_id_unknown(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    with pytest.raises(ValueError):
        save_fig(fig, "F99", "x", out_dir=tmp_path)
    plt.close(fig)


def test_save_fig_raises_when_slug_invalid(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    with pytest.raises(ValueError):
        save_fig(fig, "F36", "Bad Slug", out_dir=tmp_path)
    plt.close(fig)


def test_category_colors_has_one_valid_distinct_color_per_label() -> None:
    assert list(CATEGORY_COLORS) == LABELS
    values = list(CATEGORY_COLORS.values())
    assert all(re.fullmatch(r"#[0-9A-Fa-f]{6}", v) for v in values)
    assert len(set(values)) == len(LABELS)
