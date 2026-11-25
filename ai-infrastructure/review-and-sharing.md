# Review and sharing scope (historical)

The Cloud evidence described below was shared in commit `9275528` on `dev_vincent`. This file
records the scope and privacy checks used for that handoff. Other local changes, including slide
plans, were not included in that commit and require separate review.

## Proposed Cloud evidence

- `ai-infrastructure/README.md`, `requirements-review.md`, `architecture/`,
  `distributed-training/`, `cost-resilience/` and `optimization/` (protocol, code, results).
- `benchmarks/benchmark.ipynb`, `make_notebook.py`, `analyze_evidence.py`,
  `check_notebook_portability.py`, `evidence-index.md`, `sizing.csv`, `joint-api-results.csv`,
  `training-results.csv`, `mixed-load-results.csv` and `figures/`.
- The raw JSON/CSV and logs referenced by the evidence index in job directories
  `vertex-1129352991829131264`, `vertex-2031761767163494400`,
  `vertex-2097063961760366592`, `vertex-2985610094491729920`, the two fall-risk local JSONs and
  `vertex-gpu-summary.md`; review the logs' contents before staging. Include the related benchmark
  code and protocols necessary for reproducibility, after reviewing their diff.
- Cloud task updates in `specs/001-final-deliverable/cloud-infra-tasks.md` and the Cloud milestone
  in `docs/JOURNAL.md`; inspect the existing uncommitted changes before staging either file.

Do **not** add private input/model files (`windows.npy`, `net_state.pt`, `reference_argmax.npy`,
`ride-boundaries.csv`), fall-training workspace, worker `mlruns.tar.gz` archives, course materials,
temporary job specifications or credentials. The private files above are covered by the current
benchmark `.gitignore`; verify the staged file list and contents before committing.

The notebook's executed outputs are included for convenient review. The analysis can be rerun
from an evidence-only copy, but re-running model experiments requires private inputs, model
artifacts, the documented runtime and access to MLflow. The dated cost figures are illustrative;
verify the official regional rates before using them for a final presentation.
