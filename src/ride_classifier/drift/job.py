"""Offline drift report for per-window feature records (JSONL or parquet)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.metrics import f1_score

from ride_classifier.config import ID_COLUMN, LABELS, SENSOR_COLUMNS, TIME_COLUMN, load_params
from ride_classifier.data.clean import clean_rides
from ride_classifier.drift.psi import psi
from ride_classifier.features.window_stats import FEATURES, window_stats


def feature_records(current: pd.DataFrame) -> pd.DataFrame:
    """Accept feature rows or raw sensor rows grouped by a unique window_id."""
    if "event" in current:
        current = current[current["event"] == "window_prediction"].copy()
    if set(FEATURES).issubset(current.columns):
        return current
    required = {"window_id", TIME_COLUMN, *SENSOR_COLUMNS}
    if not required.issubset(current.columns) or current["window_id"].isna().any():
        raise ValueError("Supply eight feature columns or sensor rows with a non-null window_id and time_index")
    rows = []
    for window_id, window in current.groupby("window_id", sort=False):
        clean = clean_rides(window.assign(**{ID_COLUMN: str(window_id)}))
        row = window_stats(clean)
        for column in ("predicted_label", "true_label"):
            if column in window:
                values = window[column].dropna().unique()
                if len(values) > 1:
                    raise ValueError(f"{column} must be constant within each window")
                row[column] = values[0] if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def make_report(reference: dict, current: pd.DataFrame, config: dict, floor: float, reference_path: str) -> dict:
    """Compare feature records and, when available, prediction and delayed-label records."""
    if current.empty or not set(FEATURES).issubset(current.columns):
        raise ValueError("Current input must contain at least one row and all eight window features")
    features = {}
    reasons = []
    for name in FEATURES:
        ref = reference["features"][name]
        value = psi(ref["shares"], current[name].to_numpy(dtype=float), ref["edges"])
        level = "retrain" if value >= config["psi_retrain"] else "alert" if value >= config["psi_alert"] else "ok"
        features[name] = {"psi": value, "ks_pvalue": None, "level": level}
    if any(item["level"] == "retrain" for item in features.values()):
        reasons.append("input_drift")

    prediction_mix = None
    performance = None
    if "predicted_label" in current:
        predicted = current["predicted_label"].dropna()
        if not predicted.isin(LABELS).all():
            raise ValueError("Unknown predicted category")
        if not predicted.empty:
            counts = predicted.value_counts().reindex(LABELS, fill_value=0).to_numpy(dtype=float)
            shares = np.array([reference["prediction_mix"][label] for label in LABELS], dtype=float)
            if not np.isfinite(shares).all() or (shares < 0).any() or shares.sum() <= 0:
                raise ValueError("Invalid reference prediction mix")
            # Zero expected categories are retained with epsilon so an observed new category is not ignored.
            shares = np.maximum(shares, 1e-6)
            shares /= shares.sum()
            pvalue = float(chisquare(counts, f_exp=shares * counts.sum()).pvalue)
            prediction_mix = {
                "chi2_pvalue": pvalue,
                "n_predictions": len(predicted),
                "current": dict(zip(LABELS, (counts / counts.sum()).tolist(), strict=True)),
                "reference": reference["prediction_mix"],
            }
            if pvalue < config["chi2_pvalue"]:
                reasons.append("prediction_drift")
        if "true_label" in current:
            labelled = current.dropna(subset=["predicted_label", "true_label"])
            if not labelled["true_label"].isin(LABELS).all():
                raise ValueError("Unknown true category")
            score = None
            if len(labelled) >= config["min_labels"]:
                score = float(
                    f1_score(
                        labelled["true_label"],
                        labelled["predicted_label"],
                        labels=LABELS,
                        average="macro",
                        zero_division=0,
                    )
                )
                if score < floor:
                    reasons.append("performance")
            performance = {"n_labels": len(labelled), "macro_f1": score, "floor": floor}
    status = (
        "retrain_recommended"
        if reasons
        else ("alert" if any(item["level"] == "alert" for item in features.values()) else "ok")
    )
    return {
        "model_version": reference.get("model_version"),
        "reference": reference_path,
        "n_windows": len(current),
        "features": features,
        "prediction_mix": prediction_mix,
        "performance": performance,
        "status": status,
        "reasons": reasons,
        "notes": [
            "KS unavailable: the reference stores binned counts rather than individual values.",
            "Chi-square is approximate when expected category counts are small.",
            "The reference prediction mix is per ride; current predictions must have a comparable sampling unit.",
        ],
    }


def write_report(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    rows = [
        "# Drift report",
        "",
        f"Status: **{report['status']}**",
        f"Windows: {report['n_windows']}",
        f"Reasons: {', '.join(report['reasons']) or 'none'}",
        "",
        "| Feature | PSI | Level |",
        "|---|---:|---|",
    ]
    rows += [f"| {name} | {item['psi']:.4f} | {item['level']} |" for name, item in report["features"].items()]
    rows += [
        "",
        "## Prediction mix",
        "",
        f"`{json.dumps(report['prediction_mix'])}`",
        "",
        "## Performance",
        "",
        f"`{json.dumps(report['performance'])}`",
        "",
        *report["notes"],
    ]
    (out / "drift_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if args.current.suffix == ".jsonl":
            current = pd.read_json(args.current, lines=True)
        elif args.current.suffix == ".parquet":
            current = pd.read_parquet(args.current)
        else:
            raise ValueError("Current input must be JSONL or parquet")
        params = load_params()
        report = make_report(
            reference,
            feature_records(current),
            params["drift"],
            params["evaluate"]["gate"]["min_macro_f1"],
            str(args.reference),
        )
        write_report(report, args.out)
        return 3 if report["reasons"] else 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Drift report failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
