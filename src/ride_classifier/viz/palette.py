"""Colour palette of the visual charter (Okabe-Ito, colour-vision safe)."""

from ride_classifier.config import LABELS

_OKABE_ITO: tuple[str, ...] = ("#009E73", "#56B4E9", "#E69F00", "#CC79A7", "#D55E00")

# Safe, Safe overall, Bad weather, Tandem, Reckless
CATEGORY_COLORS: dict[str, str] = dict(zip(LABELS, _OKABE_ITO, strict=True))
MODEL_COLOR = "#0072B2"
REFERENCE_COLORS: dict[str, str] = {
    "majority": "#7F7F7F",
    "random": "#BFBFBF",
    "reference model": "#4D4D4D",
}
HEATMAP = "mako"
RISK = "YlOrRd"
