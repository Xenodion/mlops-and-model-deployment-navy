"""Summarize paired classifier benchmark rounds from raw timing CSVs."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("round-*-*.csv")):
        _, round_number, device = path.stem.split("-")
        data = pd.read_csv(path)
        for batch, group in data.groupby("batch"):
            if len(group) != 200:
                raise ValueError(f"Incomplete measurements in {path} batch {batch}")
            rows.append({"round": int(round_number), "device": device, "batch": batch,
                         "mean_ms": group.ms.mean(), "p95_ms": group.ms.quantile(0.95),
                         "windows_per_s": batch * 1000 / group.ms.mean(),
                         "rss_mib": group.sampled_rss_mib.max(),
                         "vram_mib": group.peak_allocated_vram_mib.max(),
                         "agreement": group.window_agreement.iloc[0],
                         "ride_macro_f1": group.ride_macro_f1.iloc[0]})
    result = pd.DataFrame(rows)
    if len(result) != 50:
        raise ValueError(f"Expected 50 groups, got {len(result)}")
    print(result.groupby(["device", "batch"])[["mean_ms", "p95_ms", "windows_per_s", "rss_mib", "vram_mib", "agreement", "ride_macro_f1"]].median().to_string())


if __name__ == "__main__":
    main()
