"""1D CNN on fixed-length windows of a ride (docs/PLAN.md §6.3). Same contract as the baseline (baseline.py).

The filters slide along time over the 6 sensor channels and learn short movement patterns (bumps, sharp turns,
wobbling). Each window gets 5 probabilities; a ride's prediction is the mean over its windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from ride_classifier.config import ID_COLUMN, LABELS
from ride_classifier.features.windows import windows_by_ride
from ride_classifier.models.timeline import Timeline, cumulative_mean, window_end_times


class RideCNN(nn.Module):
    """N x (convolution -> batch norm -> ReLU -> max pooling), then the mean over time, dropout and a linear layer."""

    def __init__(self, n_channels: int, filters: list[int], kernel: int, dropout: float, n_classes: int = len(LABELS)):
        super().__init__()
        blocks, channels = [], n_channels
        for out_channels in filters:
            blocks += [
                nn.Conv1d(channels, out_channels, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channels, time) -> (batch, classes) scores."""
        return self.classify(self.features(x))


class CNNModel:
    """Windows -> standardised per channel (fitted on the training rides only) -> RideCNN -> mean per ride."""

    model_type = "cnn"

    def __init__(
        self,
        window_s: float,
        stride_s: float,
        filters: list[int],
        kernel: int,
        dropout: float,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        augment: bool,
        seed: int,
        class_weight: str = "balanced",
    ):
        self.window_s, self.stride_s = window_s, stride_s
        self.filters, self.kernel, self.dropout = list(filters), kernel, dropout
        self.epochs, self.batch_size = epochs, batch_size
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.augment, self.seed = augment, seed
        self.class_weight = class_weight

    def prepare(self, readings: pd.DataFrame) -> pd.Series:
        return windows_by_ride(readings, self.window_s, self.stride_s)

    def fit(self, inputs: pd.Series, labels: pd.Series) -> CNNModel:
        x = np.concatenate(inputs.to_list())
        y = np.concatenate([np.full(len(w), LABELS.index(labels.loc[ride_id])) for ride_id, w in inputs.items()])
        self.mean_ = x.mean(axis=(0, 2), keepdims=True)
        self.std_ = x.std(axis=(0, 2), keepdims=True) + 1e-6
        x = (x - self.mean_) / self.std_

        # Same seed -> same model, so the run can be rebuilt exactly (repro_check). On CPU this holds without
        # torch.use_deterministic_algorithms, which doubled the training time (checked: two trainings, 0 difference).
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        order_rng = np.random.default_rng(self.seed)

        self.net_ = RideCNN(x.shape[1], self.filters, self.kernel, self.dropout)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y, self.class_weight), dtype=torch.float32))
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        x_all, y_all = torch.from_numpy(x), torch.from_numpy(y)
        self.history_ = []  # mean training loss per epoch, logged to MLflow
        self.net_.train()
        for _ in range(self.epochs):
            order = torch.from_numpy(order_rng.permutation(len(y)))
            total = 0.0
            for start in range(0, len(y), self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) < 2:  # batch norm needs at least 2 windows
                    continue
                xb, yb = x_all[batch], y_all[batch]
                if self.augment:
                    xb = augment(xb, generator)
                optimizer.zero_grad()
                loss = loss_fn(self.net_(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            self.history_.append(total / len(y))
        self.net_.eval()
        return self

    def _window_probas(self, windows: np.ndarray) -> np.ndarray:
        """(n_windows, channels, time) -> (n_windows, n_labels): softmax of the network on normalised windows."""
        with torch.no_grad():
            x = torch.from_numpy(((windows - self.mean_) / self.std_).astype(np.float32))
            return torch.softmax(self.net_(x), dim=1).numpy()

    def predict_proba(self, inputs: pd.Series) -> pd.DataFrame:
        rows = {ride_id: self._window_probas(windows).mean(axis=0) for ride_id, windows in inputs.items()}
        return pd.DataFrame.from_dict(rows, orient="index", columns=LABELS).rename_axis(ID_COLUMN)

    def predict_timeline(self, inputs: pd.Series) -> dict[str, Timeline]:
        """One Timeline per ride: window probabilities, their running mean, and when each window is complete."""
        timelines = {}
        for ride_id, windows in inputs.items():
            probas = self._window_probas(windows)
            end_s = window_end_times(len(probas), self.window_s, self.stride_s)
            timelines[ride_id] = Timeline(ride_id, end_s, probas, cumulative_mean(probas))
        return timelines


def class_weights(y: np.ndarray, mode: str) -> np.ndarray:
    """How much each category's windows count in the loss, so rare categories aren't ignored.

    balanced: each category counts as much as the others (a Reckless window weighs ~70x a Safe one)
    sqrt:     square root of that (~8x): softer, avoids over-predicting the rare categories
    none:     every window counts the same
    """
    counts = np.bincount(y, minlength=len(LABELS))
    balanced = np.where(counts > 0, len(y) / (len(LABELS) * np.maximum(counts, 1)), 0.0)
    if mode == "balanced":
        return balanced
    if mode == "sqrt":
        return np.sqrt(balanced)
    if mode == "none":
        return (counts > 0).astype(float)
    raise ValueError(f"unknown class_weight {mode!r}, expected 'balanced', 'sqrt' or 'none'")


def augment(windows: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Training only: each window's channels scaled by ±10 % and a little noise, so the CNN learns the pattern,
    not the exact values of 400 rides."""
    scale = 1 + 0.1 * (2 * torch.rand(windows.shape[0], windows.shape[1], 1, generator=generator) - 1)
    return windows * scale + 0.05 * torch.randn(windows.shape, generator=generator)
