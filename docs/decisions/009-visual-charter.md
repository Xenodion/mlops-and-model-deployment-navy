# ADR 009 - One visual charter for every figure, screenshot and slide

**Status:** accepted, 20/11/2026 (palette test F36 reviewed 21/11).

## Context

Two presentations are built on one base; figures come from many notebooks and people. Without a charter the decks
mix colours and styles, and a colour can mean two things.

## Decision

- White background, light mode everywhere (notebooks, web app, Grafana, screenshots).
- One Okabe-Ito colour per category, the same everywhere; Reckless gets vermillion `#D55E00`; our model blue
  `#0072B2`; references in greys.
- Heatmaps in Mako (chosen over viridis in F36: no bright yellow competing with the category hues); risk and
  drift intensity in YlOrRd (yellow to red).
- One theme module (`ride_classifier.viz`) and one `save_fig` that writes PNG and SVG to `assets/figures/`, named
  by catalogue ID (F01-F49, registry in `assets/figures/README.md`).

## Consequences

Safe overall and Bad weather have the same grey level: category names are always written next to the colours.

## Evidence

`notebooks/00_visual_charter.ipynb`, `assets/figures/F36_palette_test.png`, commit 37310be, issue #10.
