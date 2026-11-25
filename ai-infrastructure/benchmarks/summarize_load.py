"""Validate and summarize the three Vertex mixed-load plateaus."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, quantiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--figure", type=Path, help="Save a live p95 and throughput plot")
    args = parser.parse_args()
    release = json.loads((args.directory / "replay.json").read_text())
    print(f"API peak RSS across all plateaus: {release['api_peak_rss_mib']:.1f} MiB")
    print("| Clients | Wall time (s) | Requests/s | Route | Calls | Mean (ms) | p95 (ms) |")
    print("|---:|---:|---:|---|---:|---:|---:|")
    points = []
    for number, clients in enumerate((1, 2, 4), start=1):
        name = f"round-{number}"
        summary = json.loads((args.directory / f"{name}.json").read_text())
        with (args.directory / f"{name}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if summary["clients"] != clients or len(rows) != 300 or any(r["http_status"] != "200" for r in rows):
            raise ValueError(f"Incomplete plateau {name}")
        seconds = summary["elapsed_s"]
        for route, count in (("/v1/predict", 100), ("/v1/predict/window", 200)):
            times = [float(row["latency_ms"]) for row in rows if row["route"] == route]
            if len(times) != count:
                raise ValueError(f"Unexpected count for {route}")
            p95 = quantiles(times, n=100, method="inclusive")[94]
            if route.endswith("/window"):
                points.append((clients, p95, len(rows) / seconds))
            print(
                f"| {clients} | {seconds:.2f} | {len(rows) / seconds:.2f} | {route} | "
                f"{count} | {mean(times):.1f} | {p95:.1f} |"
            )
    if args.figure:
        import matplotlib.pyplot as plt

        levels, latencies, rates = zip(*points, strict=True)
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), layout="constrained")
        axes[0].plot(levels, latencies, "o-", color="#176676")
        axes[0].axhline(200, linestyle="--", color="#904221", label="Project target (200 ms)")
        axes[0].legend(frameon=False)
        axes[0].set(ylabel="Live p95 latency (ms)")
        axes[1].plot(levels, rates, "o-", color="#176676")
        axes[1].set(ylabel="Completed mixed requests / second")
        for axis in axes:
            axis.set(xlabel="Concurrent clients", xticks=levels, ylim=(0, None))
            axis.spines[["top", "right"]].set_visible(False)
        fig.savefig(args.figure, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
