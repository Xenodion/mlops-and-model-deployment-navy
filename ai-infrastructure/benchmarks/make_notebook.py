"""Create the small, reproducible benchmark notebook from its source cells."""

from pathlib import Path

import nbformat as nbf

here = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("""# Cloud infrastructure: measured baseline

This notebook reads the saved, non-private measurements and rebuilds the figures. It does not
train models or launch cloud resources. Run it from a clone with Python, pandas, matplotlib and
nbformat installed. In Colab, clone the repository first and open this notebook from that clone.
The evidence is tied to release `c193360`; see [evidence-index.md](evidence-index.md).
"""),
    nbf.v4.new_code_cell("""import importlib
import sys
from pathlib import Path

from IPython.display import Image, display

root = Path.cwd().resolve()
while not (root / 'ai-infrastructure' / 'benchmarks' / 'analyze_evidence.py').exists():
    if root.parent == root:
        raise FileNotFoundError('Run from inside a clone containing ai-infrastructure/benchmarks.')
    root = root.parent
sys.path.insert(0, str(root / 'ai-infrastructure' / 'benchmarks'))
evidence = importlib.import_module('analyze_evidence')
api, training = evidence.analyze()
evidence.plot(api, training)
print(f'Validated {len(api)} route-rounds and {len(training)} classifier training runs.')
"""),
    nbf.v4.new_markdown_cell("""## System and resource inventory

The first diagram summarizes components present in the project; it does not assert a production
deployment. The second shows the specific Vertex experiment: its client and API shared one worker.
The sizing table combines measured facts from different environments and measurement boundaries.
"""),
    nbf.v4.new_code_cell("""import pandas as pd
display(Image(filename=str(root / 'ai-infrastructure' / 'architecture' / 'current-system.png')))
display(Image(filename=str(root / 'ai-infrastructure' / 'architecture' / 'measured-replay.png')))
display(pd.read_csv(evidence.HERE / 'sizing.csv'))
"""),
    nbf.v4.new_markdown_cell("""## What is being measured?

The service receives scooter sensor readings and returns **two different results**: the ride
category so far and the current fall-risk score. A post-ride request returns the category and
retrospective risk analysis. In the CPU/CUDA comparison the same A100 worker is used twice:
only the classifier runs on CUDA in the second case; fall-risk and preprocessing stay on CPU.
This isolates device placement, not the price of two different machine types.

The underlying sample has 100 held-out rides. Each of three measured rounds per device sends
100 post-ride requests and 200 live requests, one client at a time, after warm-up. Dots below
represent rounds, not independent machines or new datasets. Latencies include the HTTP service.
"""),
    nbf.v4.new_markdown_cell("""## Joint API: both outputs, one client

Each point is the p95 of one round. Each round contains 100 whole-ride and 200 live requests;
the generator shares the A100 worker with the service. CUDA applies to the classifier, while
preprocessing and fall-risk inference remain on CPU. This does not measure stream capacity.
"""),
    nbf.v4.new_code_cell("""display(api[['device', 'round', 'route', 'requests', 'mean_ms', 'p95_ms',
             'mixed_requests_s', 'api_peak_rss_mib', 'peak_allocated_vram_mib', 'ride_macro_f1']])
display(Image(filename=str(evidence.OUTPUT / 'joint_api_p95.png')))
"""),
    nbf.v4.new_markdown_cell("""### Option A: live latency, on its own

This view answers how long the API takes to respond during a ride. The bar is the median of
three round-level p95 values; the dots show the rounds. A p95 means 95% of requests in that
round were no slower than the displayed value. Compare variation before claiming a speedup.
"""),
    nbf.v4.new_code_cell("display(Image(filename=str(evidence.OUTPUT / 'live_latency.png')))"),
    nbf.v4.new_markdown_cell("""### Alternative: distribution of the individual live requests

The previous chart summarizes each round. This view shows all 600 requests per device; it is
useful for seeing the spread, but the repeated rides are not 600 independent examples.
"""),
    nbf.v4.new_code_cell("display(Image(filename=str(evidence.OUTPUT / 'live_latency_distribution.png')))"),
    nbf.v4.new_markdown_cell("""### Option B: post-ride latency

A whole ride contains many more readings than one live window, so these values must not be
plotted on the same scale as live latency without explanation. Fall-risk output is retrospective.
"""),
    nbf.v4.new_code_cell("display(Image(filename=str(evidence.OUTPUT / 'post_ride_latency.png')))"),
    nbf.v4.new_markdown_cell("""### Option C: end-to-end host memory

This is the peak resident memory of the API process serving both models. CUDA's host memory
includes its runtime. The CUDA tensor allocator peak was 3.03 MiB; that is not total VRAM
reserved by the process and does not replace a total GPU-memory measurement.
"""),
    nbf.v4.new_code_cell("display(Image(filename=str(evidence.OUTPUT / 'api_host_memory.png')))"),
    nbf.v4.new_markdown_cell("""### GPU memory: what was actually captured

The chart shows only peak tensors allocated by PyTorch during the joint API test and the
classifier training runs. It excludes reserved memory, CUDA runtime and other GPU usage, so it
is a lower bound, not a measurement of total process VRAM or a complete machine sizing.
"""),
    nbf.v4.new_code_cell("display(Image(filename=str(evidence.OUTPUT / 'gpu_tensor_allocation.png')))"),
    nbf.v4.new_markdown_cell("""## Ride classifier training

The timer covers the classifier training stage and model logging, not data preparation or the
subsequent golden evaluation. GPU allocator memory excludes reserved memory and other CUDA usage.
The separate fall-risk CPU run includes different work and should not be compared directly.
"""),
    nbf.v4.new_code_cell("""display(training)
display(Image(filename=str(evidence.OUTPUT / 'classifier_training.png')))
"""),
    nbf.v4.new_markdown_cell("""### Alternative: training duration without the memory panel

The stage includes five folds and a final fit, plus model logging. The three GPU training runs
had different golden macro-F1 scores: faster training is not evidence of identical model quality.
"""),
    nbf.v4.new_code_cell("display(Image(filename=str(evidence.OUTPUT / 'training_duration.png')))"),
    nbf.v4.new_markdown_cell("""### Training quality check

The faster GPU training rounds did not produce identical models. This plot prevents interpreting
training time in isolation from the held-out quality of the resulting classifier.
"""),
    nbf.v4.new_code_cell("display(Image(filename=str(evidence.OUTPUT / 'training_quality.png')))"),
    nbf.v4.new_markdown_cell("""### Fall-risk model: a separate CPU observation

This full laptop run covers feature construction, training, evaluation and logging, so it is
not directly comparable to the classifier-stage timings above. It is a single observation.
"""),
    nbf.v4.new_code_cell("""import json
fall = json.loads((evidence.RESULTS / 'training-fall-risk-local-1.json').read_text())
fall_quality = json.loads((evidence.RESULTS / 'training-fall-risk-local-1-metrics.json').read_text())
print(f\"Full fall-risk stage: {fall['elapsed_s'] / 60:.1f} min; \"
      f\"sampled process-tree peak: {fall['sampled_process_tree_peak_rss_mib']:.0f} MiB.\")
print(f\"Held-out paired AUC: {fall_quality['held_out_paired_auc']:.3f} \"
      f\"across {int(fall_quality['held_out_n_groups'])} groups.\")
"""),
    nbf.v4.new_markdown_cell("""## An exploratory mixed-load test

This separate CPU test combines whole-ride and live requests. It uses a closed-loop client on
the same host as the service and only one run at each concurrency level. It helps show behavior
under this test, but cannot establish the number of live rides supported in production.
"""),
    nbf.v4.new_code_cell("""import pandas as pd
display(pd.read_csv(evidence.HERE / 'mixed-load-results.csv'))
display(Image(filename=str(evidence.OUTPUT / 'mixed_load.png')))
"""),
    nbf.v4.new_markdown_cell("""## Optimization: PyTorch and ONNX Runtime on CPU

This separate experiment tests the classifier **network only**, with already prepared windows;
it is not a measurement of the full two-model API. The same frozen weights, inputs and two CPU
threads were used for both runtimes. Each round has 20 warm-ups and 200 timed calls per batch;
three rounds alternate runtime order. Results include the conversion of ONNX outputs to tensors.
"""),
    nbf.v4.new_code_cell("""opt_dir = root / 'ai-infrastructure' / 'optimization' / 'results'
if (opt_dir / 'summary.csv').exists():
    optimization = pd.read_csv(opt_dir / 'summary.csv')
    display(optimization)
    display(Image(filename=str(evidence.OUTPUT / 'onnx_latency.png')))
    meta = json.loads((opt_dir / 'environment.json').read_text())
    print(f\"Window prediction agreement: {meta['window_argmax_agreement']:.1%}; \"
          f\"ride macro-F1: PyTorch {meta['pytorch_ride_macro_f1']:.4f}, ONNX {meta['onnx_ride_macro_f1']:.4f}.\")
    print('RSS is process-wide with both runtimes loaded; it does not isolate either runtime.')
else:
    print('Optimization results are not available yet; see optimization/protocol.md.')
"""),
    nbf.v4.new_markdown_cell("""### Alternative: network throughput as batch size grows

These are windows per second in an isolated network call, not end-to-end API requests per second.
The logarithmic horizontal scale makes the tested batch sizes visible.
"""),
    nbf.v4.new_code_cell("""if (evidence.OUTPUT / 'onnx_throughput.png').exists():
    display(Image(filename=str(evidence.OUTPUT / 'onnx_throughput.png')))
"""),
    nbf.v4.new_markdown_cell("""## Next analyses

The baseline does not establish performance at concurrent real-time load. Review the resource
sizing, cost assumptions, recovery and proposed architecture before selecting evidence for the
presentation. See the [execution tasks](../../specs/001-final-deliverable/cloud-infra-tasks.md).
"""),
]
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python"}
nbf.write(nb, here / "benchmark.ipynb")
