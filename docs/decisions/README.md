# Decision records

One file per decision that shapes the project: context, decision, alternatives rejected, consequences, evidence.
They are written to be read when preparing the presentations: each one answers "why did we do it this way?".
The day-by-day log of what was done is in [../JOURNAL.md](../JOURNAL.md).

| ADR | Decision | Status | Date |
|---|---|---|---|
| [001](001-two-models-ride-type-and-fall-risk.md) | Two distinct models: ride type (cumulative, E3) and fall risk (10 s sliding window) | accepted | 21/11 |
| [002](002-window-lengths.md) | Windows: 30 s for model 1, 10 s for model 2 (measured comparison, study closed) | accepted | 21/11 |
| [003](003-srv3490-rotation-unit-conversion.md) | SRV3490 rotation converted with 180/pi, not with the measured ratio | accepted | 21/11 |
| [004](004-fall-risk-anti-leakage-gap.md) | Fall risk: 60 s before the fall, 5 s anti-leakage gap (10 s check), same moments on comparison rides | accepted | 21/11 |
| [005](005-stateless-live-api.md) | The live API is stateless: the client carries the running state | accepted | 20/11 |
| [006](006-gcp-instead-of-aws.md) | GCP instead of AWS; GPU runs as self-terminating Vertex AI SPOT jobs | accepted | 20/11 |
| [007](007-public-images-without-model-or-data.md) | Published Docker Hub images carry neither the model nor data | accepted | 20/11 |
| [008](008-hybrid-retraining-with-human-approval.md) | Hybrid retraining policy with a human approval before production | accepted | 20/11 |
| [009](009-visual-charter.md) | One visual charter for every figure, screenshot and slide | accepted | 20/11 |
| [010](010-one-service-two-models.md) | One inference service serving both models | accepted | 21/11 |

Template: copy any record; keep it under one page; link the evidence (commit, notebook, CSV, issue).
