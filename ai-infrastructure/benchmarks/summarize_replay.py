"""Summarize three completed sequential joint API replay rounds."""

import argparse
import csv
from pathlib import Path
from statistics import mean, quantiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    routes = {"/v1/predict": 100, "/v1/predict/window": 200}
    print("round,route,requests,mean_ms,p95_ms")
    for number in range(1, 4):
        with (args.directory / f"round-{number}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != sum(routes.values()) or any(row["http_status"] != "200" for row in rows):
            raise ValueError(f"Incomplete or failed requests in round {number}")
        for route, expected in routes.items():
            times = [float(row["latency_ms"]) for row in rows if row["route"] == route]
            if len(times) != expected:
                raise ValueError(f"Unexpected request count for {route} in round {number}")
            p95 = quantiles(times, n=100, method="inclusive")[94]
            print(f"{number},{route},{expected},{mean(times):.3f},{p95:.3f}")


if __name__ == "__main__":
    main()
