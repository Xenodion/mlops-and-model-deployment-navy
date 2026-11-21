"""DVC stage `prepare_srv`: SRV3490 raw readings -> converted (rad/s to deg/s) and cleaned table.

A cache of `data.srv3490.load_clean_srv3490`, so training and studies do not redo it; the model still cleans every
ride it receives (`clean_rides` in the served model), so live data goes through the same step.
"""

from __future__ import annotations

import logging

import pyarrow as pa
import pyarrow.parquet as pq

from ride_classifier.config import ID_COLUMN, ROOT, load_params
from ride_classifier.data.srv3490 import load_clean_srv3490
from ride_classifier.pipeline.cli import setup

logger = logging.getLogger(__name__)
RIDES_PER_BATCH = 500  # ponytail: fixed batches keep peak RAM ~1 GB on the 12 GB laptop; no tuning needed


def main() -> None:
    setup()
    srv = load_params()["data"]["srv3490"]
    raw, out = ROOT / srv["raw_path"], ROOT / srv["clean_path"]
    ride_ids = sorted(pq.read_table(raw, columns=[ID_COLUMN]).column(ID_COLUMN).unique().to_pylist())
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for start in range(0, len(ride_ids), RIDES_PER_BATCH):
            batch = load_clean_srv3490(raw, srv["rotation_factor"], ride_ids[start : start + RIDES_PER_BATCH])
            table = pa.Table.from_pandas(batch, preserve_index=False)
            writer = writer or pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    logger.info("wrote %d rides to %s", len(ride_ids), out)


if __name__ == "__main__":
    main()
