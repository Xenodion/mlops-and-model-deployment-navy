"""Validate the saved joint API and training results and regenerate figures.

Run from the repository root: python ai-infrastructure/benchmarks/analyze_evidence.py
No private inputs, network access or cloud resources are required.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
API = RESULTS / "vertex-2031761767163494400"
TRAINING = RESULTS / "vertex-2097063961760366592" / "results-v2"
OUTPUT = HERE / "figures"
COLORS = {"cpu": "#60a5fa", "cuda": "#fbbf24"}


def style():
    """Local white presentation style; leaves the project's default theme untouched."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#172334",
            "axes.labelcolor": "#172334",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "axes.edgecolor": "#9ca3af",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        target = OUTPUT / f"{name}.{extension}"
        fig.savefig(target, dpi=200, bbox_inches="tight")
        if extension == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def analyze():
    api_rows, train_rows = [], []
    for device in ("cpu", "cuda"):
        report = json.loads((API / device / "device.json").read_text())
        assert report["device"] == device and report["classifier_forwards"] == 1600
        replay = json.loads((API / device / "replay.json").read_text())
        for run in range(1, 4):
            frame = pd.read_csv(API / device / f"round-{run}.csv")
            assert len(frame) == 300 and (frame.http_status == 200).all()
            timing = json.loads((API / device / f"round-{run}.json").read_text())
            for route, expected in (("/v1/predict", 100), ("/v1/predict/window", 200)):
                values = frame.loc[frame.route == route, "latency_ms"]
                assert len(values) == expected
                api_rows.append(
                    {
                        "device": device,
                        "round": run,
                        "route": route,
                        "requests": len(values),
                        "mean_ms": values.mean(),
                        "p95_ms": values.quantile(0.95),
                        "mixed_requests_s": timing["requests_s"],
                        "api_peak_rss_mib": replay["api_peak_rss_mib"],
                        "peak_allocated_vram_mib": report.get("peak_allocated_mib", 0),
                        "ride_macro_f1": replay["post_ride_macro_f1"],
                    }
                )
            item = json.loads((TRAINING / f"{run}-{device}.json").read_text())
            assert item["device"] == device and item["elapsed_s"] > 0
            train_rows.append(
                {
                    "device": device,
                    "round": run,
                    "elapsed_s": item["elapsed_s"],
                    "peak_process_rss_mib": item["peak_process_rss_mib"],
                    "peak_allocated_vram_mib": item["peak_allocated_vram_mib"],
                    "golden_macro_f1": item["golden"]["golden_macro_f1"],
                    "stage_run_id": item["stage_run_id"],
                }
            )
    api = pd.DataFrame(api_rows)
    training = pd.DataFrame(train_rows)
    api.to_csv(HERE / "joint-api-results.csv", index=False)
    training.to_csv(HERE / "training-results.csv", index=False)
    return api, training


def plot(api, training):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, (route, label) in zip(
        axes, (("/v1/predict/window", "Live window"), ("/v1/predict", "Whole ride")), strict=True
    ):
        for x, device in enumerate(("cpu", "cuda")):
            subset = api[(api.route == route) & (api.device == device)]
            ax.scatter([x] * 3, subset.p95_ms, color=COLORS[device], s=48)
            ax.plot([x - 0.15, x + 0.15], [subset.p95_ms.median()] * 2, color="#172334", lw=2)
        ax.set_xticks([0, 1], ["CPU", "CUDA"])
        ax.set_ylabel("p95 latency (ms)")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Joint API: p95 for each of three rounds")
    save(fig, "joint_api_p95")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, column, label in (
        (axes[0], "elapsed_s", "Training stage (s)"),
        (axes[1], "peak_process_rss_mib", "Peak process RSS (MiB)"),
    ):
        for x, device in enumerate(("cpu", "cuda")):
            subset = training[training.device == device]
            ax.scatter([x] * 3, subset[column], color=COLORS[device], s=48)
            ax.plot([x - 0.15, x + 0.15], [subset[column].median()] * 2, color="#172334", lw=2)
        ax.set_xticks([0, 1], ["CPU", "CUDA"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Ride classifier training: three runs per device")
    save(fig, "classifier_training")

    for route, name, title in (
        ("/v1/predict/window", "live_latency", "Live: time to return both outputs"),
        ("/v1/predict", "post_ride_latency", "Post-ride: time to analyze a whole ride"),
    ):
        subset = api[api.route == route]
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        for x, device in enumerate(("cpu", "cuda")):
            values = subset[subset.device == device]
            median = values.p95_ms.median()
            ax.bar(x, median, color=COLORS[device], width=0.55)
            ax.scatter([x] * 3, values.p95_ms, color="#172334", zorder=3, s=26)
            ax.text(
                x, values.p95_ms.max() + subset.p95_ms.max() * 0.035, f"{median:,.1f} ms", ha="center", weight="bold"
            )
        ax.set_xticks([0, 1], ["Classifier on CPU", "Classifier on CUDA"])
        ax.set_ylabel("95th percentile of request latency (ms)")
        ax.set_ylim(0, subset.p95_ms.max() * 1.2)
        ax.set_title(title, loc="left", weight="bold")
        ax.text(
            0,
            -0.23,
            "Bars: median of 3 rounds  |  Dots: individual rounds  |  One client",
            transform=ax.transAxes,
            fontsize=9,
            color="#4b5563",
        )
        save(fig, name)

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for x, device in enumerate(("cpu", "cuda")):
        values = training[training.device == device].elapsed_s
        ax.bar(x, values.median(), color=COLORS[device], width=0.55)
        ax.scatter([x] * 3, values, color="#172334", zorder=3, s=26)
        ax.text(x, values.median() + 7, f"{values.median():.1f} s", ha="center", weight="bold")
    ax.set(
        xticks=[0, 1],
        xticklabels=["CPU", "One A100 GPU"],
        ylabel="Training stage duration (s)",
        ylim=(0, training.elapsed_s.max() * 1.2),
    )
    ax.set_title("Training the ride classifier on the same worker", loc="left", weight="bold")
    ax.text(
        0,
        -0.23,
        "Bars: median of 3 runs  |  Dots: runs  |  Excludes golden evaluation",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    save(fig, "training_duration")

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for x, device in enumerate(("cpu", "cuda")):
        values = api[(api.route == "/v1/predict/window") & (api.device == device)]
        rss = values.api_peak_rss_mib.iloc[0]
        ax.bar(x, rss, color=COLORS[device], width=0.55)
        ax.text(x, rss + 35, f"{rss:,.0f} MiB", ha="center", weight="bold")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Classifier on CPU", "Classifier on CUDA"],
        ylabel="Peak API process RSS (MiB)",
        ylim=(0, 1550),
    )
    ax.set_title("Host memory used by the API serving both models", loc="left", weight="bold")
    ax.text(
        0,
        -0.23,
        "CUDA process RSS includes the GPU runtime; this is not GPU memory.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    save(fig, "api_host_memory")

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    allocations = [
        api[(api.route == "/v1/predict/window") & (api.device == "cuda")].peak_allocated_vram_mib.iloc[0],
        training[training.device == "cuda"].peak_allocated_vram_mib.max(),
    ]
    bars = ax.bar([0, 1], allocations, color=["#60a5fa", "#e8a51b"], width=0.55)
    for bar, value in zip(bars, allocations, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f} MiB", ha="center", weight="bold")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Joint API: classifier\non CUDA", "Classifier training\non CUDA"],
        ylabel="Peak allocated GPU tensor memory (MiB)",
        ylim=(0, max(allocations) * 1.25),
    )
    ax.set_title("Measured GPU allocation during inference and training", loc="left", weight="bold")
    ax.text(
        0,
        -0.23,
        "PyTorch allocator only: excludes reserved memory, CUDA runtime and other usage.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    save(fig, "gpu_tensor_allocation")

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for x, device in enumerate(("cpu", "cuda")):
        values = training[training.device == device].golden_macro_f1
        ax.scatter([x] * len(values), values, color=COLORS[device], s=65)
        ax.plot([x - 0.12, x + 0.12], [values.median()] * 2, color="#172334", lw=2)
    ax.set(xticks=[0, 1], xticklabels=["CPU", "One A100 GPU"], ylabel="Golden ride macro-F1", ylim=(0, 1))
    ax.set_title("Classifier quality varied between GPU training runs", loc="left", weight="bold")
    ax.text(
        0,
        -0.23,
        "Each dot: one training run; horizontal mark: median. Same 100 golden rides.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    save(fig, "training_quality")

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for device in ("cpu", "cuda"):
        values = []
        for run in range(1, 4):
            frame = pd.read_csv(API / device / f"round-{run}.csv")
            values.extend(frame.loc[frame.route == "/v1/predict/window", "latency_ms"])
        ax.hist(
            values,
            bins=30,
            histtype="step",
            linewidth=2,
            color=COLORS[device],
            label="CPU classifier" if device == "cpu" else "CUDA classifier",
        )
    ax.set(xlabel="Live request latency (ms)", ylabel="Number of requests")
    ax.set_title("Distribution of live request latency", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.text(
        0,
        -0.23,
        "600 requests per device across 3 rounds; the same 100 rides are replayed.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    save(fig, "live_latency_distribution")

    load_dir = RESULTS / "vertex-2985610094491729920"
    load = []
    for run, clients in enumerate((1, 2, 4), 1):
        frame = pd.read_csv(load_dir / f"round-{run}.csv")
        meta = json.loads((load_dir / f"round-{run}.json").read_text())
        assert len(frame) == 300 and (frame.http_status == 200).all() and meta["clients"] == clients
        live = frame.loc[frame.route == "/v1/predict/window", "latency_ms"]
        assert len(live) == 200
        load.append(
            {"clients": clients, "live_p95_ms": live.quantile(0.95), "mixed_requests_s": len(frame) / meta["elapsed_s"]}
        )
    pd.DataFrame(load).to_csv(HERE / "mixed-load-results.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, metric, label in (
        (axes[0], "live_p95_ms", "Live request p95 (ms)"),
        (axes[1], "mixed_requests_s", "Completed mixed requests / s"),
    ):
        ax.plot([r["clients"] for r in load], [r[metric] for r in load], "o-", color="#2563a8", lw=2)
        ax.set(xlabel="Concurrent clients", ylabel=label, xticks=[1, 2, 4], ylim=(0, None))
    fig.suptitle("Mixed-load stress test on one CPU worker", weight="bold")
    fig.text(
        0.05,
        -0.05,
        "One run per level; generator and API share the host. Not real-time ride capacity.",
        fontsize=9,
        color="#4b5563",
    )
    save(fig, "mixed_load")

    raw_path = HERE.parent / "optimization" / "results" / "latencies.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        expected = 3 * 2 * 4 * 200
        if len(raw) != expected:
            raise ValueError(f"Expected {expected} optimization timings, got {len(raw)}")
        opt = raw.groupby(["runtime", "round", "batch"], as_index=False).agg(
            mean_ms=("latency_ms", "mean"),
            p95_ms=("latency_ms", lambda x: x.quantile(0.95)),
            process_rss_mib=("process_rss_mib", "max"),
        )
        opt["windows_per_s"] = opt.batch * 1000 / opt.mean_ms
        opt.to_csv(HERE.parent / "optimization" / "results" / "summary.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
        for ax, batch in zip(axes, (1, 128), strict=True):
            part = opt[opt.batch == batch]
            for x, runtime in enumerate(("pytorch", "onnx")):
                values = part[part.runtime == runtime].mean_ms
                ax.bar(x, values.median(), color=("#60a5fa", "#34a58a")[x], width=0.55)
                ax.scatter([x] * 3, values, color="#172334", zorder=3, s=24)
                ax.text(
                    x, values.max() + part.mean_ms.max() * 0.04, f"{values.median():.3f} ms", ha="center", weight="bold"
                )
            ax.set(
                xticks=[0, 1],
                xticklabels=["PyTorch", "ONNX Runtime"],
                ylabel="Mean network call latency (ms)",
                ylim=(0, part.mean_ms.max() * 1.25),
            )
            ax.set_title(f"Batch of {batch} window{'s' if batch > 1 else ''}")
        fig.suptitle("CPU network optimization: three paired rounds", weight="bold")
        fig.text(
            0.05,
            -0.04,
            "Prepared inputs; classifier network only. Dots: rounds; bars: median.",
            fontsize=9,
            color="#4b5563",
        )
        save(fig, "onnx_latency")

        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        for runtime, color, label in (("pytorch", "#60a5fa", "PyTorch"), ("onnx", "#34a58a", "ONNX Runtime")):
            medians = opt[opt.runtime == runtime].groupby("batch").windows_per_s.median()
            ax.plot(medians.index, medians.values, "o-", color=color, linewidth=2, label=label)
        ax.set(
            xlabel="Windows per batch", ylabel="Classifier windows per second", xticks=[1, 8, 32, 128], ylim=(0, None)
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 32, 128], ["1", "8", "32", "128"])
        ax.set_title("CPU network throughput by batch size", loc="left", weight="bold")
        ax.legend(frameon=False)
        ax.text(
            0,
            -0.23,
            "Median of 3 rounds; prepared windows, not API requests.",
            transform=ax.transAxes,
            fontsize=9,
            color="#4b5563",
        )
        save(fig, "onnx_throughput")


if __name__ == "__main__":
    api, training = analyze()
    plot(api, training)
    print(f"Validated {len(api)} API route-rounds and {len(training)} training runs.")
    print(f"Figures saved to {OUTPUT}")
