"""Matplotlib theme and figure saving for the visual charter."""

import logging
import re
from functools import cache
from pathlib import Path

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from ride_classifier.config import LABELS, ROOT
from ride_classifier.viz.palette import CATEGORY_COLORS, HEATMAP

logger = logging.getLogger(__name__)

ASSETS_DIR = ROOT / "assets" / "figures"
REGISTRY_PATH = ASSETS_DIR / "README.md"
DPI = 200
_ID_RE = re.compile(r"F\d{2}")
_SLUG_RE = re.compile(r"[a-z0-9_]+")
_ROW_RE = re.compile(r"^\|\s*(F\d{2})\s*\|", re.MULTILINE)


def apply_theme() -> None:
    """Set matplotlib rcParams: white background, slide-friendly sizes, category colours."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=[CATEGORY_COLORS[k] for k in LABELS]),
        }
    )


@cache
def _registered_ids(registry: Path = REGISTRY_PATH) -> frozenset[str]:
    return frozenset(_ROW_RE.findall(registry.read_text(encoding="utf-8")))


def save_fig(fig: Figure, fig_id: str, slug: str, out_dir: Path = ASSETS_DIR) -> tuple[Path, Path]:
    """Save ``fig`` as ``{fig_id}_{slug}.png`` (200 dpi) and ``.svg`` in ``out_dir``."""
    if not _ID_RE.fullmatch(fig_id):
        raise ValueError(f"fig_id must match F + two digits, got {fig_id!r}")
    if fig_id not in _registered_ids():
        raise ValueError(f"{fig_id} is not listed in {REGISTRY_PATH}")
    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(f"slug must match [a-z0-9_]+, got {slug!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    png, svg = out_dir / f"{fig_id}_{slug}.png", out_dir / f"{fig_id}_{slug}.svg"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    logger.info("Saved figure %s and %s", png, svg)
    return png, svg


def heatmap_cmap() -> Colormap:
    """Return the charter heatmap colormap (seaborn mako)."""
    import seaborn as sns

    return sns.color_palette(HEATMAP, as_cmap=True)
