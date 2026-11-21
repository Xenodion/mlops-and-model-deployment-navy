# Decision records

One file per decision that shapes the project: context, decision, alternatives rejected, consequences, evidence.
They are written to be read when preparing the presentations: each one answers "why did we do it this way?".
The day-by-day log of what was done is in [../JOURNAL.md](../JOURNAL.md).

| ADR | Decision | Status | Date |
|---|---|---|---|
| [001](001-danger-detector-learned-without-labels.md) | The model is a live danger detector learned without labels, plus a 5-class head | accepted | 21/11 |
| [002](002-window-length-by-preregistered-study.md) | The window length is decided by a pre-registered study, not inherited | accepted, study running | 20/11 |
| [003](003-srv3490-rotation-unit-conversion.md) | SRV3490 rotation converted with 180/pi, not with the measured ratio | accepted | 21/11 |
| [004](004-pre-fall-segment.md) | Pre-fall danger: 60 s segment, 2 s margin, comparison rides at the same elapsed time | accepted | 21/11 |
| [005](005-stateless-live-api.md) | The live API is stateless: the client carries the running state | accepted | 20/11 |
| [006](006-gcp-instead-of-aws.md) | GCP instead of AWS; GPU runs as self-terminating Vertex AI SPOT jobs | accepted | 20/11 |
| [007](007-public-images-without-model-or-data.md) | Published Docker Hub images carry neither the model nor data | accepted | 20/11 |
| [008](008-hybrid-retraining-with-human-approval.md) | Hybrid retraining policy with a human approval before production | accepted | 20/11 |
| [009](009-visual-charter.md) | One visual charter for every figure, screenshot and slide | accepted | 20/11 |

Template: copy any record; keep it under one page; link the evidence (commit, notebook, CSV, issue).
