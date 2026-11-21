# Project journal

What was done, why, and where the evidence is, newest first. Decisions are detailed in
[decisions/](decisions/README.md); this log is the story behind them, written to be reread when preparing the
presentations. One entry per working session; mistakes found and corrected are logged too.

## 21/11/2026

**Reframing after reading the course `data.txt`** (ADR 001). The archive shipped with SRV3490 contains a data
description: GBG500 labels are "used only for evaluation", Reckless is "the positive class we aim to detect", and
Safe overall, Bad weather and Tandem are counterfactuals that "should not be flagged". Together with the teacher's
briefing (option 2: unsupervised, validated with labels, linked to falls), this showed that our 20/11 decision
(supervised classifier) was option 1. We also found that the golden set holds a single Reckless ride. New model: a
danger score learned without labels on all ~4,000 rides, cumulated live, plus a 5-class head for the brief; E3
kept as reference. Spec, plan, research, contracts and issues updated (commits f5bde04, d816e61).

**Falls as a label-free check** (ADR 004). Vincent's idea: with only 6 Reckless rides, use the 698 SRV3490 falls to
ask whether danger precedes a fall. Fall groups split 70/30 by city (489 development, 209 held-out groups, a group
never split); pre-fall segment = 60 s ending 2 s before the fall; comparison rides scored at the same elapsed time
(commit e4a02b8). Using falls as training examples ("weak supervision") stays an experiment, only if this check
shows a signal (issue #28).

**Built, by five parallel agents then reviewed** (commits 37310be to 958fc87, then c163a67 to 8248e55):

- visual charter: palette, theme, `save_fig`, palette test F36 (Mako kept over viridis);
- timeline: cumulative values window by window, `predict_timeline` on E3;
- SRV3490 loader: 3,490 rides, 698 crash rides with exactly 4 comparison rides each, 8 cities, 100 ms sampling,
  no id overlap with GBG500; notebook 02 with F04 and F05;
- `bench.py`: one benchmark tool, CSV schema fixed by contract, Python 3.10-compatible for Vertex AI; first laptop
  CPU baseline (measured): 0.9 ms per window at batch 1, ~9,300 windows/s at batch 512;
- `DangerModel`: label-free convolutional autoencoder + logistic 5-class head; a test checks the encoder never reads
  labels;
- API: danger score and timeline in `/v1/predict`, stateless `/v1/predict/window` (ADR 005), `/v1/feedback`,
  metrics of monitoring layers 3 and 4;
- window study rewritten for the danger detector; pre-registration revised before any full run (ADR 002).

**Corrected during review:**

- the SRV3490 rotation factor had been set to the measured ratio (60.04); replaced by 180/pi (ADR 003);
- the timeline returned the first window's value before that window was complete, which uses future data
  (a 60 s window would have "answered" at 30 s); it now returns no answer;
- the pre-fall score was the cumulative danger since the start of the ride; replaced by the mean over the windows
  ending in the pre-fall segment;
- 1.6 % of SRV3490 sensor values are missing: cleaning moved into one loader and a DVC stage `prepare_srv`
  (commit 648fe25); the served model still cleans every ride it receives, so live data goes through the same step.

**Data facts:** SRV3490 tracked with DVC from the course archive (the archive itself is git-ignored). 334 of 2,792
comparison rides are shorter than their fall's pre-fall segment and give no pair. No fall happens before 120 s of
riding.

**Open:** full window study (candidate 30 s running first to measure time and memory on the laptop); CI test of the
unit-bug drift (task T059); fall analysis on held-out groups (issue #29).

## 20/11/2026

**Teacher's briefing.** Two options (supervised on 500 rides, or unsupervised on all data validated with labels);
not a plain k-means; explore features predictive of falls. Written up in `docs/PROJECT_FRAMING.md` (commit beb6e1f).

**First framing: a cumulative live classifier** (superseded on 21/11). A ride read window by window, its category
"so far" at every moment. Fall forecast and live replay as bonus (commit c1f043c).

**One design for both courses** (commit 75869b5): live load test (one Locust user = one ride streaming at real
speed), inference hardware CPU / 1 GPU / 2 GPUs, training cost vs data volume (real data replicated x1-x8), drift
scenarios built by resampling real rides (new city, real sensor unit difference, riskier mix).

**Planning with Spec Kit** (commits aa72d7f to 01df461): constitution (8 principles, local), spec with 4 user
stories and a 46-figure catalogue, plan in phases (core MLOps first, AI infrastructure in parallel, bonus last),
research decisions, contracts, 54 tasks, 19 GitHub issues with colour-coded priority labels and checkpoint
milestones (23/11, 24/11, 25/11 20:00). Four-layer monitoring, drift job and hybrid retraining with human approval
made core (ADR 008). GCP confirmed as the only allowed deviation from the brief (ADR 006). Visual charter fixed
(ADR 009).

**Window length questioned** (ADR 002): 30 s is not the data's granularity (10 Hz) but E3's untested choice; made an
open choice to be decided by a pre-registered study.

## 16/11 - 19/11/2026 (before this journal)

System built from the course labs: DVC pipeline (prepare, train, evaluate with a quality gate), frozen golden split,
statistics baseline then 1D CNN (E1-E3, E3 golden macro-F1 0.51), MLflow tracking and registry with aliases,
FastAPI service, Streamlit web app, Docker images on Docker Hub, CI, deployment on kind with a canary. Details in
`docs/PLAN.md` and the git history (commits 5bd852a to 0d35241).
