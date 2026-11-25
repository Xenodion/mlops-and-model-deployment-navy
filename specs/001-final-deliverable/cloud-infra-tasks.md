# Cloud infrastructure - execution tasks

Purpose: make `ai-infrastructure/` understandable, reproducible and ready for a teammate to build the oral. The brief is `lectures/ai-infrastructure/project-requirements.md` (local, ignored). This list is the current Cloud execution queue; the older C01–C12 table remains historical where it differs. Alex owns model, API and front changes; use the frozen release until he provides a new one. Do not modify either slide outline without Vincent's review. No push without asking.

## Progress update, 25 November

- CI01–02: the three principal Vertex jobs are confirmed successful and no active custom job was
  found in the tested project/region. Joint API counts, device reports and six training JSONs are
  validated by `benchmarks/analyze_evidence.py`; the six imported training runs and principal API,
  network and fall-risk MLflow runs are linked in `benchmarks/evidence-index.md`.
- CI04–05: `sizing.csv`, `joint-api-results.csv`, `training-results.csv` and `mixed-load-results.csv`
  are available; review the measurement boundaries and any missing hardware details.
- CI06–07: the pre-registered CPU ONNX comparison completed, saved 4,800 raw timings and was
  tracked in MLflow; the run was explicitly closed after a Windows console encoding issue.
  See `optimization/results-summary.md`. RSS does not isolate runtimes.
- CI08–11: distributed analysis, proposed cost/recovery and current/proposed diagrams are drafted;
  verify actual/current scope and complete cost assumptions and artifact review.
- CI12–16: Vincent requested **white** backgrounds and one substantial notebook with multiple
  explanatory views. `benchmark.ipynb` now includes optimization instead of a separate notebook;
  it was executed locally and in a minimal evidence-only copy. Continue reviewing alternatives and
  traceability. Do not follow the older dark-background or separate-notebook wording below.
- Cloud evidence was committed and pushed on `dev_vincent` as `9275528`. CI17–18 remain open:
  `ai-infrastructure/requirements-review.md` identifies caveats and `oral-evidence.md` proposes
  the presentation narrative. Verify pricing and coordinate the shared MLOps load/rollback tests.

## Priorities and evidence

- **P0** enables the rest or prevents loss of evidence. **P1** covers the teacher's expected contribution and Vincent's requested notebooks and figures. **P2** improves interpretation or a shared MLOps demonstration. **P3** is optional.
- Done means evidence exists and has been validated. The Cloud evidence in commit `9275528` is team-accessible on `dev_vincent`; any subsequent local changes are not shared until pushed. MLflow runs require permission.
- Pinned release `c193360`, image digest `sha256:a9164bacca4e95f4389334d00e4251a1ddaa6c3b1c77176167e9dd7b84d5f7a8`, ride-classifier/2 and fall-risk/1. Both outputs share the API. Live and whole-ride requests have distinct meanings.
- Paired classifier forward job `1129352991829131264`, joint API job `2031761767163494400`, classifier training job `2097063961760366592` succeeded and terminated. Raw results, logs, provenance, MLflow IDs and limitations are in `ai-infrastructure/benchmarks/results/vertex-gpu-summary.md`. CPU and GPU on the same A100 host are not CPU-only hosting prices. One GPU was used; no multi-GPU claim.
- Fall-risk local full training completed once with tracking and quality metrics. Existing CPU mixed-load run `2985610094491729920` is a closed-loop shared-host stress test, not measured live-ride capacity or an autoscaling test.
- The teacher requires sizing, a reproducible baseline, one optimization comparison where feasible, a reasoned distributed-training decision, and a proposed scalable, cost-aware and recoverable architecture. Repository layout is suggested. Separate the actual brief from team-selected stretch work.

## Ordered tasks

### P0 - Preserve and validate the evidence (#3, #4, #6)

- [ ] CI01 Check that each successful Vertex job has terminal status, complete result files and readable local logs, and that the matching MLflow runs exist. Confirm no benchmark job remains active. Inputs: `vertex-gpu-summary.md`, `results/vertex-*/`. Output: evidence index with job IDs, run IDs, source files, counts, image/model versions and known limitations. Never commit datasets, model files or worker MLflow archives.
- [ ] CI02 Validate all API CSVs have three rounds per device, 100 post-ride and 200 live HTTP 200s per round. Recompute mean, p95 and throughput from raw files; check device reports contain CPU/CUDA forward counts. Recompute three classifier training results per backend and confirm golden quality and VRAM. Correct any discrepancy in the summary, preserving raw files. Record the fall-risk run as a single local sample and distinguish its stage from E3's timer.
- [ ] CI03 Review Git status and identify the explicit Cloud-only files needed by teammates. Keep raw private inputs and model artifacts ignored. Verify that any existing local presentation changes remain available for Vincent's review. Prepare a reviewable diff; obtain the required approval before pushing.

### P1 - Core course contribution (#3–#8)

- [ ] CI04 Build `benchmarks/sizing.csv` and a short sizing narrative from `architecture/inventory.md`, the measured training results and environment files. Cover both datasets and models, tensor shape, artifact bytes, observed RAM, allocator VRAM, hardware capacities, training duration and measurement boundaries. Include uncertainty and label the laptop fall-risk sample separately.
- [ ] CI05 Reconcile the baseline into machine-readable tables. Use full joint API live/post-ride results as the main inference comparison, including one client, workload mix, 100 rides, three rounds, warm-up, mean/p95, throughput, RAM, VRAM and quality. Keep CNN batches as diagnostic evidence and report their exclusion of transfers and preprocessing. Include the mixed-load results with their shared-host and closed-loop limitations.
- [ ] CI06 Before running optimization, write the hypothesis and protocol under `optimization/`: export E3 to ONNX FP32, compare against PyTorch CPU on the same prepared inputs, worker, threads, batch sizes and repeated timed boundaries. Define quality checks for per-window agreement and ride-level macro-F1, plus latency, throughput and memory. This is a network optimization, so do not present its result as an end-to-end API improvement.
- [ ] CI07 Implement and run the CI06 comparison with warm-up, repeated rounds, raw samples and tracked MLflow runs. Preserve failures or regressions. If export or runtime is genuinely unavailable, document the exact blocker and still deliver the prerequisite, hypothesis and reproducible procedure. INT8 and additional variants are P3.
- [ ] CI08 Finish `distributed-training/technical-analysis.md` using measured E3 duration, model size, RAM and allocated VRAM, and the single fall-risk CPU run. Explain applicability and communication/memory trade-offs of DDP, FSDP and ZeRO, state limitations and a condition for revisiting. No 2-GPU timing or benefit without an actual comparable run.
- [ ] CI09 Turn the current Mermaid architecture in `architecture/` into a verified, readable diagram and create a proposed diagram with both models, ingress, replicas, monitoring and recovery. Label actual components and proposed components clearly, and include image/model provenance. Export PNG and an editable/vector source.
- [ ] CI10 Complete `cost-resilience/infrastructure-plan.md`: propose a CPU serving starting point and a GPU training option, state region, machine, hours, volume and request mix, retrieve dated authoritative price sources, show calculations and assumptions, and estimate costs. Do not extrapolate reliable replica counts from the existing one-client or mixed shared-host replay. Note credit visibility and Vertex teardown status.
- [ ] CI11 Describe scaling, monitoring and recovery with explicit measured versus proposed status. Reuse an observed kind/canary rollback if the MLOps team supplies evidence. If a drill is unavailable, provide a proposed recovery procedure and mark timing unmeasured. Coordinate with #20 and #19 rather than modifying Alex's API/front work.

### P1 - Reproducible notebooks and figures (#45)

- [ ] CI12 Inspect `src/ride_classifier/viz/theme.py`, palette and figure registry. Establish one shared Cloud plotting style compatible with Vincent's simple dark-slide preference, legible contrast, consistent CPU/GPU/model colors and units. Reuse existing theme infrastructure where practical; document any dark-background override without changing other teams' figures unexpectedly. Avoid decorative content.
- [ ] CI13 Create `benchmarks/benchmark.ipynb` that runs from a repository clone/Colab with documented dependencies, reads saved non-private CSV/JSON only, validates counts and regenerates summary tables. It must not silently launch cloud resources, require private data for viewing, or fabricate points. Provide straightforward rerun instructions and meaningful errors for missing files.
- [ ] CI14 In that notebook generate slide-ready figures from the raw data: joint API live and post-ride CPU/GPU latency with variation, training duration and host/GPU memory with units and boundaries, and measured mixed-load behavior with its caveat. Keep the classifier-only batch figure as optional diagnostic evidence. Visually inspect legibility and export PNG plus SVG using the shared style. Figure choices are proposals for the story, not mandatory slide content.
- [ ] CI15 Create `optimization/comparison.ipynb` from CI07 raw results, using the same style. Show before/after latency, throughput, memory and quality; include an explicit blocker and no invented chart if CI07 cannot run. Export reusable assets and preserve a rerunnable analysis.
- [ ] CI16 Create an evidence index linking each proposed figure/table to a notebook cell, raw file, job or MLflow run, units, conditions and known limitations. Run notebooks from a clean working directory, validate generated files, and verify at least one plotted value independently against the CSV. Update `ai-infrastructure/README.md` with entry points and instructions.

### P1 - Handoff before slide creation (#8, #24)

- [ ] CI17 Review all course requirements against artifacts. Check that current/proposed architecture, both models, training/inference, RAM/VRAM, optimization, distributed decision, scaling, cost and recovery are covered or have a concrete documented blocker. Capture reproducible screenshots only where they add proof and redact sensitive data.
- [ ] CI18 Provide Vincent and teammates the evidence index and open choices for the oral. Slides tell the technical purpose, actions, measured results, decisions and next steps. Cover, index and progress indicator are required by Vincent. Suggest figures without dictating a fixed layout or unsupported fictional rationale. Slide outlines and published changes require his review.

### P2/P3 - Conditional work

- [ ] CI19 P2: if kind and both registered models are available, run a real-time live-window replay at controlled concurrency with memory limits, errors, route p95 and repeatable loads. Record generator placement and cadence. Coordinate with MLOps owners. Do not infer ride capacity from the existing closed-loop mixed replay.
- [ ] CI20 P2: repeat fall-risk CPU training or separately profile feature extraction and fit only if it changes a decision or addresses an evidence gap. The current single full-stage observation remains valid with its limits.
- [ ] CI21 P3: test multi-GPU only if a concrete training or throughput target makes it worthwhile. Write a fair one-versus-two-device protocol first and bound cloud costs and teardown. INT8, replicated datasets and extra hardware are optional, subject to remaining time.

## Dependencies and handoff

CI01–03 first. CI04–05 and CI06 can follow CI02 independently. CI07 requires CI06. CI08 requires CI02 and CI04. CI09–11 use CI04–05 and CI08; CI11 may incorporate CI19 evidence. CI12 can start after CI02. CI13–14 require CI05 and CI12; CI15 requires CI07 and CI12. CI16 requires CI09–15. CI17–18 require CI08–16 or an explicit blocker. CI19–21 are conditional and do not hold the required written brief hostage.

Cloud tasks are owned by the Cloud workstream. Alex owns changes to models, API and front. Before executing a shared kind/rollback drill, identify the MLOps owner and coordinate the actual environment. CI03 keeps Git publication subject to Vincent's review and push approval.
