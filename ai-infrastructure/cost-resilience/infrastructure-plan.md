# Proposed serving, training and recovery

## Serving starting point

Start with one CPU service loading both version-pinned models. An illustrative VM is an
`n1-standard-4` (4 vCPU, 15 GiB RAM) in `europe-west4`. It is a starting configuration, not a
measured capacity recommendation: the paired test used an A100 host with two configured CPU
threads and a single client. The CPU-only mixed test used a different host and a shared closed-loop
load generator. Neither establishes how many live rides a machine can sustain.

Add replicas only after measuring p95, errors and CPU/RAM under representative live and
post-ride concurrency with a separate load generator. Provision headroom, use readiness probes,
and keep per-ride state in the request or an external store so requests can be routed safely.
Scaling, managed deployment and failover are proposals, not demonstrated results.

## Cost illustration, not a bill

An independent [regional price comparison](https://instances.usage.ai/gcp/compute/n1-standard-4)
lists an on-demand `n1-standard-4` in the Netherlands at **$0.2092/hour**, consulted
25 November 2026. Google’s [official pricing page](https://cloud.google.com/products/compute/pricing/general-purpose)
must be checked with the Netherlands region selected before using this estimate: its indexed
tables mix regions, so they do not independently confirm this figure. For one continuously
running VM over 730 hours: 730 x $0.2092 = **$152.72/month**
for VM compute only. This excludes disk, networking, load balancer, monitoring, registry, tax,
discounts and any orchestration charges. One VM is not highly available. Two always-on identical
VMs would double this compute-only estimate to **$305.43/month**, without establishing that two
replicas are required or sufficient. Request volume does not change these fixed VM costs; it
does affect capacity, network and possibly the number of replicas, which are not yet measured.

For classifier training, the measured stage is around 34 s on one A100 worker, but the job also
has startup, setup, data, logging and shutdown time. The stage duration cannot be multiplied by
an hourly rate to claim the actual invoice. As a separate **Compute Engine VM-only illustration**,
an independent [regional list](https://instances.usage.ai/gcp/compute/region/europe-west4) showed
`a2-highgpu-1g` at about $3.748/hour on demand, consulted 25 November 2026: a hypothetical 10-minute
allocation is roughly $0.62 before other charges. This is not the price or bill for a Vertex Spot
training job. Use the current [Vertex AI pricing page](https://cloud.google.com/vertex-ai/pricing)
and billing export for region, machine and actual wall time. Spot rates may change and capacity
may be interrupted. Current credits and actual job cost have not been verified here.

## Monitoring and recovery

Monitor per-route request counts, p95 latency and errors, CPU, process RAM, pod restarts and
model version. Monitor input drift, prediction mix and delayed outcome quality separately. Alert
thresholds and sustained windows require baseline calibration; the project's 200 ms live budget
is a target, not a measured service guarantee.

On a failed release: stop promotion, route traffic back to the previous healthy image and pinned
model versions, verify both outputs and live-state behavior, and inspect errors and telemetry.
If no healthy replica remains, restore the last known-good deployment and artifacts. Keep a
five-minute recovery target as a proposed acceptance criterion until an actual timed drill proves
it. Do not claim autoscaling or rollback tests that were not performed.
