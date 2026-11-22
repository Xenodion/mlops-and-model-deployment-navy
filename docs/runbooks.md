# Monitoring runbooks

Thresholds are provisional until calibrated on the normal training-ride scenario. Prometheus alerts are local
rules; no notification channel is configured. A lack of data is not evidence of normal behaviour.

## Latency

**Symptom:** window prediction p95 exceeds 200 ms. Check request volume, CPU and memory, API logs and model
loading. Compare with a repeatable load test before changing replica count or CPU limits.

## Errors

**Symptom:** more than 5% of requests return a server error. Check `/health`, API logs, model availability and
recent deployment changes. Fix or roll back the faulty deployment; distinguish 5xx from invalid client requests.

## Input drift

**Symptom:** one or more feature PSI values exceed 0.25. Check `drift_monitor_up`, window count, API reference
version and the affected feature distributions. The alert does not identify a cause. If rotation values differ
by a factor near 57, inspect the sender's rad/s to deg/s conversion and fix the upstream unit bug; do not retrain
to compensate for it. If the population genuinely changed, investigate representative labelled data and run the
gated training process once available. Do not promote a model based on PSI alone.

## Prediction drift

Compare prediction mix on independent rides with the out-of-fold ride reference. Repeated windows of the same
ride are not independent ride samples. The live prediction-mix alert is not implemented until compatible sampling
is available; use the offline report on a comparable sample in the meantime.

## Performance

**Symptom:** observed feedback accuracy falls below the provisional floor with at least 20 labels. Check feedback
coverage, label quality, class mix and selection bias. Accuracy and the offline macro-F1 gate measure different
things. Investigate a labelled sample before considering gated retraining.
