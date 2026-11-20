# Contract: figures, theme and assets

## Theme module

- `ride_classifier.viz.palette` (no dependencies): `CATEGORY_COLORS: dict[str, str]` (Okabe-Ito, one per
  category, Reckless `#D55E00`), `MODEL_COLOR = "#0072B2"`, `REFERENCE_COLORS` (greys), `HEATMAP = "mako"`,
  `RISK = "YlOrRd"`. Imported by notebooks and the web app; the same hex values are typed into the Grafana dashboard.
- `ride_classifier.viz.theme`: `apply_theme() -> None` (white background, font sizes for slides, no top/right
  spines) and `save_fig(fig, fig_id: str, slug: str) -> tuple[Path, Path]`.

## `save_fig`

- Writes `assets/figures/{fig_id}_{slug}.png` (200 dpi, white background) and `.svg`, overwriting.
- `fig_id` must match `F\d{2}`; unknown ids raise `ValueError` (the registry below is the list).
- Returns both paths; logs them.

## Registry

`assets/figures/README.md`: one table row per catalogue ID (spec Figure Catalogue F01-F46) with `slug`, story,
core/bonus, source, status (`planned | draft | final`). Screenshots (F17-F21, F32, F41, F42) are saved with the
same naming by hand. Demo fallbacks (FR-004) go to `assets/demos/` (recordings).

## Notebooks

| Notebook | Figures |
|---|---|
| `notebooks/00_visual_charter.ipynb` | F36 |
| `notebooks/01_data_overview.ipynb` | F01-F03 |
| `notebooks/02_units_and_cities.ipynb` | F04, F05 |
| `notebooks/03_live_classifier.ipynb` | F06-F08, F47 |
| `notebooks/04_evaluation.ipynb` | F09-F11 |
| `notebooks/05_srv3490_pseudo_labels.ipynb` | F12, F13 |
| `notebooks/06_drift_scenarios.ipynb` | F22, F44 (+ F45 from the job output) |
| `notebooks/07_fall_forecast.ipynb` | F38 (bonus) |
| `notebooks/08_encoder.ipynb` | F39 (bonus) |
| `ai-infrastructure/benchmarks/benchmark.ipynb` | F23-F27 |
| `ai-infrastructure/optimization/comparison.ipynb` | F28 |
| `ai-infrastructure/distributed-training/` notebook | F29, F30, F46 |
| `ai-infrastructure/cost-resilience/capacity.ipynb` | F31, F33, F34 |

Notebooks read results files (CSV, JSON); they do not rerun benchmarks, so they stay fast and reproducible.
