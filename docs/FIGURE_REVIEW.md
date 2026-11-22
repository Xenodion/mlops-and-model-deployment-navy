# Figure review for the two presentations

Review date: 22 November 2026. This is a visual and narrative review of every currently exported figure
(`assets/figures/F*.png`) and every image output in the committed notebooks. Thirteen exported figures exist;
the other catalogue entries remain planned. Notebook images correspond to these exported figures. This review
does not assert that the underlying measurements have been independently reproduced.

For each proposed slide, state the one question it answers, choose a representation for that question, show the
sample and units, and make comparisons and limitations visible on the figure or in the spoken explanation.
No figure becomes final solely because it exists. F36 is an internal style test, not a presentation result.

| Figure | Intended message | Review and next action |
|---|---|---|
| F04 | Rotation units differ across datasets and conversion brings their medians closer. | **Revised draft.** Before/after ratios of per-ride standard-deviation medians show the unit factor and ride-bootstrap intervals; the bootstrap is not group-aware. Similar medians do not establish identical distributions, which the plot does not show in full. |
| F05 | SRV3490 comes from eight cities with uneven coverage. | **Revised draft.** A count table states that fall rides are included in all rides and that the four-comparison matching is a dataset design, not a natural fall rate. |
| F06 | Model 1 consumes overlapping 30-second windows at 15-second steps. | **Keep as a diagram.** The interval representation fits the explanation. Make the overlap and the step explicit for a slide audience. |
| F07 | Averaging windows stabilizes the category score for one ride. | **Keep with context.** A time series fits; label both curves directly and identify this as one selected example, not a population result. |
| F08 | Golden-set macro-F1 changes with elapsed ride time. | **Revised draft.** All 100 rides stay in the cohort, but rides without a later complete window retain their last prediction: 47 have a window at or after 300 s and none at or after 600 s. The reference lines use whole rides. |
| F09 | The classifier's mistakes differ by known category. | **Contrast fixed.** Counts are now readable. Keep explicit counts and expose the imbalance and the single Reckless example; consider a separate row-normalized view only if counts remain visible. |
| F10 | The golden macro-F1 clears the quality gate and reference scores. | **Revised draft.** A common-axis dot plot labels the three scores and the floor directly. Explain macro-F1 and the held-out population when presenting. |
| F11 | Show what was predicted Reckless, including the limited evidence for that category. | **Revised draft.** A count table replaces the 100% bar from one ride, keeps false positives visible and states why one ride cannot establish detection reliability. |
| F22 | An upstream rotation-unit error produces a clear input shift. | **Revised draft.** An annotated threshold table replaces grouped bars and unequal log axes. Normal and riskier samples stay below provisional thresholds; the city also shifts acceleration Z. Thresholds are not calibrated; seek presentation review. |
| F36 | Test colours and grayscale choices. | **Keep internal only.** It uses illustrative values and is not evidence or an oral result. |
| F47 | A prior development study compared window lengths for a label-free score. | **Revised draft.** The figure identifies the earlier score and development groups, says it is not model 2, and labels the 95% fall-group bootstrap band. |
| F48 | Model 1 aggregates completed windows while model 2 scores the latest 10 seconds. | **Keep as a diagram, refine labels.** Make window overlap and the current time visually unambiguous; no accuracy claim is implied. |
| F49 | Held-out window-level ROC curves compare the 5 s and 10 s gaps. | **Keep the ROC for this question.** Clearly distinguish window AUC 0.71/0.69 from paired AUC 0.89/0.76. A separate paired-result comparison is needed for the gap-sensitivity story; do not label the ROC as paired AUC. |

Next: review the remaining draft diagrams with the presentation narrative. F22's threshold labels and any
sensitivity claims must be revisited after calibration. Audit new figures when created.
