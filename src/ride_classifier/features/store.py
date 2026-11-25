"""Lightweight local Feature Store: offline Parquet + SQLite registry & online lookup.

Materializes versioned feature views once from raw/cleaned readings into:
- an offline store (Parquet per view/version) for reproducible training and batch jobs;
- an online store (SQLite indexed by view, version, entity_id) for low-latency lookup;
- a metadata registry (schema, feature list, source fingerprint, timestamp).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ride_classifier.config import ID_COLUMN, ROOT
from ride_classifier.features.stats import ride_features
from ride_classifier.features.window_stats import window_stats

DEFAULT_STORE_ROOT = ROOT / "artifacts/feature_store"


@dataclass(frozen=True)
class FeatureView:
    """Definition of a reusable, versioned feature view."""

    name: str
    version: str
    entity_column: str
    description: str
    compute: Callable[[pd.DataFrame], pd.DataFrame]


def _compute_window_drift(readings: pd.DataFrame) -> pd.DataFrame:
    """Compute the 8 drift summary features per ride/window entity."""
    entity_col = "window_id" if "window_id" in readings.columns else ID_COLUMN
    rows = {entity_id: window_stats(group) for entity_id, group in readings.groupby(entity_col, sort=True)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(entity_col)


def default_views() -> tuple[FeatureView, ...]:
    """Built-in feature views for ride-level statistics and window-level drift features."""
    return (
        FeatureView(
            name="ride_stats",
            version="v1",
            entity_column=ID_COLUMN,
            description="Per-ride time, jerk, distribution and Welch spectral statistics (~170 features).",
            compute=ride_features,
        ),
        FeatureView(
            name="window_drift_stats",
            version="v1",
            entity_column=ID_COLUMN,
            description="Eight standard-deviation features (6 channels + 2 norms) used for input drift monitoring.",
            compute=_compute_window_drift,
        ),
    )


def _table_schema(frame: pd.DataFrame) -> dict[str, str]:
    return {str(col): str(dtype) for col, dtype in frame.dtypes.items()}


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()[:16]


class FeatureStore:
    """File-backed offline Parquet + SQLite online feature store."""

    def __init__(self, root: Path = DEFAULT_STORE_ROOT, views: Sequence[FeatureView] = ()):
        self.root = Path(root)
        self.offline_dir = self.root / "offline"
        self.db_path = self.root / "store.db"
        self._views: dict[tuple[str, str], FeatureView] = {}
        for view in views:
            self.register(view)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS feature_views (
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    entity_column TEXT NOT NULL,
                    description TEXT NOT NULL,
                    feature_columns TEXT NOT NULL,
                    schema_json TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    n_rows INTEGER NOT NULL,
                    offline_path TEXT NOT NULL,
                    materialized_at TEXT NOT NULL,
                    PRIMARY KEY (name, version)
                );
                CREATE TABLE IF NOT EXISTS online_features (
                    view_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (view_name, version, entity_id)
                );
                """
            )

    def register(self, view: FeatureView) -> None:
        """Register a FeatureView definition in memory for materialization."""
        self._views[(view.name, view.version)] = view

    def get_view(self, name: str, version: str = "v1") -> FeatureView:
        key = (name, version)
        if key not in self._views:
            raise KeyError(f"Unknown feature view {name}@{version}")
        return self._views[key]

    def materialize(self, name: str, raw_frame: pd.DataFrame, version: str = "v1") -> dict:
        """Compute `name@version` on `raw_frame` and persist to both offline Parquet and online SQLite."""
        view = self.get_view(name, version)
        computed = view.compute(raw_frame).copy()
        if computed.index.name != view.entity_column and view.entity_column in computed.columns:
            computed = computed.set_index(view.entity_column)
        computed.index = computed.index.astype(str)
        computed.index.name = view.entity_column

        self.offline_dir.mkdir(parents=True, exist_ok=True)
        offline_path = self.offline_dir / f"{view.name}_{view.version}.parquet"
        computed.to_parquet(offline_path, index=True)

        now = datetime.now(UTC).isoformat(timespec="seconds")
        fingerprint = _frame_fingerprint(computed)
        feature_columns = [str(c) for c in computed.columns]
        schema = _table_schema(computed)

        rows = [
            (view.name, view.version, str(entity_id), json.dumps(record), now)
            for entity_id, record in zip(computed.index, computed.to_dict(orient="records"), strict=True)
        ]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feature_views (
                    name, version, entity_column, description, feature_columns,
                    schema_json, fingerprint, n_rows, offline_path, materialized_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name, version) DO UPDATE SET
                    entity_column=excluded.entity_column,
                    description=excluded.description,
                    feature_columns=excluded.feature_columns,
                    schema_json=excluded.schema_json,
                    fingerprint=excluded.fingerprint,
                    n_rows=excluded.n_rows,
                    offline_path=excluded.offline_path,
                    materialized_at=excluded.materialized_at
                """,
                (
                    view.name,
                    view.version,
                    view.entity_column,
                    view.description,
                    json.dumps(feature_columns),
                    json.dumps(schema),
                    fingerprint,
                    len(computed),
                    str(offline_path),
                    now,
                ),
            )
            conn.execute(
                "DELETE FROM online_features WHERE view_name = ? AND version = ?",
                (view.name, view.version),
            )
            conn.executemany(
                """
                INSERT INTO online_features (view_name, version, entity_id, features_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

        return {
            "name": view.name,
            "version": view.version,
            "n_rows": len(computed),
            "n_features": len(feature_columns),
            "fingerprint": fingerprint,
            "offline_path": str(offline_path),
            "materialized_at": now,
        }

    def get_offline_features(
        self, name: str, version: str = "v1", entity_ids: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Read materialized features from the offline Parquet store."""
        meta = self.describe(name, version)
        frame = pd.read_parquet(meta["offline_path"])
        if entity_ids is not None:
            wanted = [str(e) for e in entity_ids]
            frame = frame.reindex(wanted)
        return frame

    def get_online_features(
        self, name: str, entity_ids: Sequence[str], version: str = "v1"
    ) -> dict[str, dict[str, float] | None]:
        """Low-latency lookup of precomputed feature vectors from SQLite by entity_id."""
        wanted = [str(e) for e in entity_ids]
        if not wanted:
            return {}
        placeholders = ",".join("?" for _ in wanted)
        with self._connect() as conn:
            found = {
                row["entity_id"]: json.loads(row["features_json"])
                for row in conn.execute(
                    f"""
                    SELECT entity_id, features_json FROM online_features
                    WHERE view_name = ? AND version = ? AND entity_id IN ({placeholders})
                    """,
                    (name, version, *wanted),
                )
            }
        return {entity_id: found.get(entity_id) for entity_id in wanted}

    def describe(self, name: str, version: str = "v1") -> dict:
        """Return metadata for a materialized feature view."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM feature_views WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
        if row is None:
            raise KeyError(f"Feature view {name}@{version} has not been materialized")
        return {
            "name": row["name"],
            "version": row["version"],
            "entity_column": row["entity_column"],
            "description": row["description"],
            "feature_columns": json.loads(row["feature_columns"]),
            "schema": json.loads(row["schema_json"]),
            "fingerprint": row["fingerprint"],
            "n_rows": row["n_rows"],
            "offline_path": row["offline_path"],
            "materialized_at": row["materialized_at"],
        }

    def list_views(self) -> list[dict]:
        """List all materialized feature views in the store."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, version, entity_column, n_rows, fingerprint, materialized_at "
                "FROM feature_views ORDER BY name, version"
            ).fetchall()
        return [dict(row) for row in rows]


def default_store(root: Path = DEFAULT_STORE_ROOT) -> FeatureStore:
    """Instantiate the project FeatureStore with built-in ride and drift feature views."""
    return FeatureStore(root=root, views=default_views())


def main() -> None:
    parser = argparse.ArgumentParser(description="Local ride-classifier Feature Store CLI")
    parser.add_argument("--root", type=Path, default=DEFAULT_STORE_ROOT, help="Feature store directory")
    sub = parser.add_subparsers(dest="command", required=True)

    mat = sub.add_parser("materialize", help="Compute and materialize a feature view from a parquet file")
    mat.add_argument("--input", type=Path, required=True, help="Input readings parquet")
    mat.add_argument("--view", default="ride_stats", help="Feature view name")
    mat.add_argument("--version", default="v1", help="Feature view version")

    online = sub.add_parser("online", help="Fetch online features for one or more entity IDs")
    online.add_argument("--view", default="ride_stats")
    online.add_argument("--version", default="v1")
    online.add_argument("--entity-id", action="append", required=True, dest="entity_ids")

    sub.add_parser("list", help="List materialized feature views")

    args = parser.parse_args()
    store = default_store(args.root)
    if args.command == "materialize":
        frame = pd.read_parquet(args.input)
        print(json.dumps(store.materialize(args.view, frame, version=args.version), indent=2))
    elif args.command == "online":
        print(json.dumps(store.get_online_features(args.view, args.entity_ids, version=args.version), indent=2))
    elif args.command == "list":
        print(json.dumps(store.list_views(), indent=2))


if __name__ == "__main__":
    main()
