"""The API contract (docs/PLAN.md §9.1): what a caller sends and what comes back.

Pydantic rejects anything outside the contract with a 422 that names the wrong field (lab 3): the caller's
mistake is reported as such, instead of a confident wrong answer or a 500.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ride_classifier.config import SAMPLING_PERIOD_MS, SENSOR_COLUMNS, TIME_COLUMN

MIN_READINGS = 300  # 30 s at 10 Hz: a shorter ride doesn't carry enough signal
MAX_READINGS = 20_000  # ~33 min: protects the service from huge requests

# Physical bounds: the accelerometer saturates around ±20 m/s², the gyroscope around ±300 deg/s
Acceleration = Annotated[float, Field(ge=-100, le=100, allow_inf_nan=False)]
Rotation = Annotated[float, Field(ge=-2000, le=2000, allow_inf_nan=False)]

ACCELERATION_HELP = "m/s², gravity included. null = missing reading, filled in like in training"
ROTATION_HELP = "deg/s. null = missing reading, filled in like in training"


def _example(n: int = MIN_READINGS) -> dict:
    """A valid 30-second ride for Swagger's "Try it out" (a still sensor, gravity on the y axis)."""
    return {
        "ride_id": "example-ride",
        "time_index": [i * SAMPLING_PERIOD_MS for i in range(n)],
        "ax": [-0.1] * n,
        "ay": [-9.2] * n,
        "az": [-2.3] * n,
        "rx": [0.0] * n,
        "ry": [0.0] * n,
        "rz": [0.0] * n,
    }


class Ride(BaseModel):
    """One ride's raw readings: one array per channel, all the same length.

    Columnar on purpose: a ride has ~3,000 readings, and validating 7 arrays is much faster than 3,000 small objects.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_example()]})

    ride_id: str | None = Field(default=None, max_length=128, description="Your identifier for the ride, sent back")
    time_index: list[Annotated[int, Field(ge=0)]] = Field(
        min_length=MIN_READINGS,
        max_length=MAX_READINGS,
        description=f"Milliseconds since the start of the ride: one reading every {SAMPLING_PERIOD_MS} ms (10 Hz)",
    )
    ax: list[Acceleration | None] = Field(description=f"Acceleration, x axis, {ACCELERATION_HELP}")
    ay: list[Acceleration | None] = Field(description=f"Acceleration, y axis (≈ vertical), {ACCELERATION_HELP}")
    az: list[Acceleration | None] = Field(description=f"Acceleration, z axis, {ACCELERATION_HELP}")
    rx: list[Rotation | None] = Field(description=f"Rotation, x axis, {ROTATION_HELP}")
    ry: list[Rotation | None] = Field(description=f"Rotation, y axis, {ROTATION_HELP}")
    rz: list[Rotation | None] = Field(description=f"Rotation, z axis, {ROTATION_HELP}")

    @model_validator(mode="after")
    def check_shape(self) -> Ride:
        lengths = {name: len(getattr(self, name)) for name in (TIME_COLUMN, *SENSOR_COLUMNS)}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"all arrays must have the same length, got {lengths}")
        if any(later < earlier for earlier, later in zip(self.time_index, self.time_index[1:], strict=False)):
            raise ValueError("time_index must never go backwards")
        return self

    def to_frame(self) -> pd.DataFrame:
        """The readings as the model expects them (null -> NaN, filled in by the model's cleaning)."""
        return pd.DataFrame(
            {
                TIME_COLUMN: np.asarray(self.time_index, dtype="int64"),
                **{column: np.asarray(getattr(self, column), dtype="float64") for column in SENSOR_COLUMNS},
            }
        )


class Prediction(BaseModel):
    ride_id: str | None = Field(description="The ride_id you sent")
    label: str = Field(description="Most likely category: Safe, Safe overall, Bad weather, Tandem or Reckless")
    probabilities: dict[str, float] = Field(description="One probability per category (they sum to 1)")
    reckless_score: float = Field(description="Probability of Reckless, compared with the threshold")
    is_reckless: bool = Field(description="reckless_score >= the service's threshold (see /health)")
    model_version: str = Field(description="Registry version of the model that answered")
    n_readings: int = Field(description="Readings received")


class Health(BaseModel):
    status: Literal["ok", "degraded"] = Field(description="degraded = no model loaded: predictions answer 503")
    model_loaded: bool
    model_version: str
    reckless_threshold: float
