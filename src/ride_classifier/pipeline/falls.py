"""Fall groups (a fall ride + its comparison rides): group-level split and pre-fall segments of SRV3490."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ride_classifier.config import ID_COLUMN, ROOT, TIME_COLUMN, load_params
from ride_classifier.data.srv3490 import CRASH_REF_COLUMN, TOPPLE_COLUMN, load_srv3490, ride_table

logger = logging.getLogger(__name__)

GROUP_COLUMN = "fall_group_id"
ROLE_COLUMN = "role"
GROUP_SPLIT_COLUMN = "group_split"
FALL, COMPARISON = "fall", "comparison"
DEVELOPMENT, HELD_OUT = "development", "held_out"


def split_fall_groups(rides: pd.DataFrame, held_out_share: float, seed: int) -> pd.DataFrame:
    """One row per ride: ride_id, fall_group_id, city, role, group_split.

    The split is made on whole groups, stratified by city (each city keeps about ``held_out_share`` of its
    groups held out, at least one when it has two or more). Groups are sorted before shuffling, so the same
    rides and seed always give the same split whatever the input order.
    """
    table = rides[[ID_COLUMN, GROUP_COLUMN, "city"]].sort_values(ID_COLUMN).reset_index(drop=True)
    table[ROLE_COLUMN] = np.where(table[ID_COLUMN] == table[GROUP_COLUMN], FALL, COMPARISON)
    rng = np.random.default_rng(seed)
    held_out: set[str] = set()
    for _, groups in table.groupby("city", sort=True)[GROUP_COLUMN]:
        ids = np.array(sorted(groups.unique()))
        n_held = round(held_out_share * len(ids))
        if len(ids) >= 2:
            n_held = max(n_held, 1)
        held_out.update(rng.permutation(ids)[:n_held].tolist())
    table[GROUP_SPLIT_COLUMN] = np.where(table[GROUP_COLUMN].isin(held_out), HELD_OUT, DEVELOPMENT)
    return table


def pre_fall_window(fall_time_index_ms: int, pre_fall_s: float, margin_s: float) -> tuple[int, int]:
    """Elapsed-time window [start_ms, end_ms) before a fall: ends margin_s before it, lasts pre_fall_s.

    The start is clipped at 0, so a fall earlier than pre_fall_s + margin_s gives a shorter window.
    """
    end = int(fall_time_index_ms - margin_s * 1000)
    start = max(0, int(end - pre_fall_s * 1000))
    return start, end


def segment(readings: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Readings of one ride with start_ms <= elapsed < end_ms, elapsed = time_index - first time_index.

    end_ms is exclusive and lies before the fall, so the impact is excluded. The same elapsed window can be
    cut on the comparison rides of the group.
    """
    elapsed = readings[TIME_COLUMN] - readings[TIME_COLUMN].min()
    return readings.loc[(elapsed >= start_ms) & (elapsed < end_ms)]


def main() -> None:
    """Split the SRV3490 fall groups and write the committed ride-id table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params()
    cfg = params["falls"]
    columns = [ID_COLUMN, CRASH_REF_COLUMN, "city", TIME_COLUMN, TOPPLE_COLUMN]
    readings = load_srv3490(ROOT / params["data"]["srv3490"]["raw_path"], columns)
    rides = ride_table(readings)
    del readings
    groups = split_fall_groups(rides, cfg["held_out_share"], cfg["seed"])
    path = ROOT / Path(cfg["groups_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    groups.to_csv(path, index=False)
    counts = groups[groups[ROLE_COLUMN] == FALL].groupby(["city", GROUP_SPLIT_COLUMN]).size()
    logger.info("fall groups per city and split:\n%s", counts.to_string())
    logger.info("wrote %d rides to %s", len(groups), path)

    falls = rides.merge(groups[[ID_COLUMN, ROLE_COLUMN]], on=ID_COLUMN)
    falls = falls[falls[ROLE_COLUMN] == FALL]
    too_early = (falls["fall_time_index"] < (cfg["pre_fall_s"] + cfg["margin_s"]) * 1000).sum()
    logger.info("falls earlier than pre_fall_s + margin_s into the ride: %d", too_early)


if __name__ == "__main__":
    main()
