"""Convolutional autoencoder on ride windows: label-free danger score = reconstruction error (specs R1).

The encoder is E3's convolution blocks. The autoencoder and the danger score never see labels; only the optional
5-class head (logistic regression on the mean embedding of a ride) does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F

from ride_classifier.config import ID_COLUMN, LABELS
from ride_classifier.features.windows import windows_by_ride
from ride_classifier.models.cnn import class_weights
from ride_classifier.models.timeline import Timeline, cumulative_mean, window_end_times

INFERENCE_BATCH = 4096  # windows per forward pass outside training (bounds peak memory)


class RideAutoencoder(nn.Module):
    """Encoder: N x (Conv1d, BatchNorm, ReLU, MaxPool1d(2)). Decoder: mirror, Upsample(2) instead of pooling."""

    def __init__(self, n_channels: int, filters: list[int], kernel: int):
        super().__init__()
        encoder, channels = [], n_channels
        for out_channels in filters:
            encoder += [
                nn.Conv1d(channels, out_channels, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            channels = out_channels
        self.encoder = nn.Sequential(*encoder)
        decoder, sizes = [], [*reversed(filters), n_channels]
        for i, (in_channels, out_channels) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            decoder += [nn.Upsample(scale_factor=2), nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2)]
            if i < len(sizes) - 2:
                decoder += [nn.BatchNorm1d(out_channels), nn.ReLU()]
        self.decoder = nn.Sequential(*decoder)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channels, time) -> (batch, filters[-1]): encoder output averaged over time."""
        return self.encoder(x).mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction with the input's shape (cropped or edge-padded when the length is not a multiple of 2^N)."""
        out = self.decoder(self.encoder(x))[..., : x.shape[2]]
        return F.pad(out, (0, x.shape[2] - out.shape[2]), mode="replicate")


class DangerModel:
    """Windows -> standardised per channel -> autoencoder (no labels) -> z-scored reconstruction error per window."""

    model_type = "danger"

    def __init__(
        self,
        window_s: float,
        stride_s: float,
        filters: list[int],
        kernel: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        head_C: float,
        class_weight: str,
        seed: int,
    ):
        self.window_s, self.stride_s = window_s, stride_s
        self.filters, self.kernel = list(filters), kernel
        self.epochs, self.batch_size = epochs, batch_size
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.head_C, self.class_weight, self.seed = head_C, class_weight, seed
        self.head_: LogisticRegression | None = None

    def prepare(self, readings: pd.DataFrame) -> pd.Series:
        return windows_by_ride(readings, self.window_s, self.stride_s)

    def fit(self, inputs: pd.Series, labels: pd.Series | None = None) -> DangerModel:
        """Train the autoencoder on ALL windows (labels never used), then the head if labels is given."""
        x = np.concatenate(inputs.to_list())
        self.mean_ = x.mean(axis=(0, 2), keepdims=True)
        self.std_ = x.std(axis=(0, 2), keepdims=True) + 1e-6
        n_channels = x.shape[1]
        x -= self.mean_  # in place: x is a fresh copy, so no second array of all windows is made
        x /= self.std_
        x_all = torch.from_numpy(x.astype(np.float32, copy=False))
        del x

        torch.manual_seed(self.seed)
        order_rng = np.random.default_rng(self.seed)
        self.net_ = RideAutoencoder(n_channels, self.filters, self.kernel)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.history_ = []  # mean training MSE per epoch
        self.net_.train()
        for _ in range(self.epochs):
            order = torch.from_numpy(order_rng.permutation(len(x_all)))
            total = 0.0
            for start in range(0, len(x_all), self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) < 2:  # batch norm needs at least 2 windows
                    continue
                optimizer.zero_grad()
                loss = F.mse_loss(self.net_(x_all[batch]), x_all[batch])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            self.history_.append(total / len(x_all))
        self.net_.eval()

        errors = self._errors_standardised(x_all)
        del x_all
        self.error_mean_, self.error_std_ = float(errors.mean()), float(errors.std() + 1e-9)
        if labels is not None:
            self._fit_head(inputs, labels)
        return self

    def _fit_head(self, inputs: pd.Series, labels: pd.Series) -> None:
        """5-class logistic regression on the mean embedding per ride, only for rides present in labels."""
        ride_ids = [r for r in inputs.index if r in labels.index]
        features = np.stack([self._embeddings(inputs.loc[r]).mean(axis=0) for r in ride_ids])
        y = np.array([LABELS.index(labels.loc[r]) for r in ride_ids])
        weights = {k: float(v) for k, v in enumerate(class_weights(y, self.class_weight)) if k in set(y)}
        self.head_ = LogisticRegression(C=self.head_C, class_weight=weights, max_iter=5000).fit(features, y)

    def _normalise(self, windows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((windows - self.mean_) / self.std_).astype(np.float32))

    def _errors_standardised(self, x: torch.Tensor) -> np.ndarray:
        """Mean squared reconstruction error per window of already standardised windows, by batches."""
        with torch.no_grad():
            parts = [
                ((self.net_(b) - b) ** 2).mean(dim=(1, 2)) for b in torch.split(x, INFERENCE_BATCH)
            ]  # batches: the activations of all windows at once would not fit in memory
        return torch.cat(parts).numpy()

    def _errors(self, windows: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error per window, on standardised windows."""
        return self._errors_standardised(self._normalise(windows))

    def _embeddings(self, windows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            parts = [self.net_.embed(b) for b in torch.split(self._normalise(windows), INFERENCE_BATCH)]
        return torch.cat(parts).numpy()

    def danger_windows(self, windows: np.ndarray) -> np.ndarray:
        """(n_windows, channels, time) -> (n_windows,) reconstruction error z-scored on the training windows."""
        return (self._errors(windows) - self.error_mean_) / self.error_std_

    def _head_probas(self, windows: np.ndarray) -> np.ndarray:
        """Head applied to each window's own embedding: (n_windows, n_labels), 0 for categories unseen in training."""
        probas = np.zeros((len(windows), len(LABELS)))
        probas[:, self.head_.classes_] = self.head_.predict_proba(self._embeddings(windows))
        return probas

    def predict_proba(self, inputs: pd.Series) -> pd.DataFrame:
        """Head probabilities per ride from the mean embedding of its windows. Raises if fitted without labels."""
        if self.head_ is None:
            raise RuntimeError("no 5-class head: fit() was called without labels")
        rows = {}
        for ride_id, windows in inputs.items():
            row = np.zeros(len(LABELS))
            row[self.head_.classes_] = self.head_.predict_proba(self._embeddings(windows).mean(axis=0, keepdims=True))[
                0
            ]
            rows[ride_id] = row
        return pd.DataFrame.from_dict(rows, orient="index", columns=LABELS).rename_axis(ID_COLUMN)

    def predict_timeline(self, inputs: pd.Series) -> dict[str, Timeline]:
        """One Timeline per ride: danger per window and its running mean, plus category probabilities.

        The probabilities are the head applied to each window's own embedding (the ride's mean embedding would use
        windows not yet recorded); they are NaN when the model has no head.
        """
        timelines = {}
        for ride_id, windows in inputs.items():
            danger = self.danger_windows(windows)
            if self.head_ is None:
                probas = np.full((len(windows), len(LABELS)), np.nan)
            else:
                probas = self._head_probas(windows)
            end_s = window_end_times(len(windows), self.window_s, self.stride_s)
            timelines[ride_id] = Timeline(
                ride_id, end_s, probas, cumulative_mean(probas), danger, cumulative_mean(danger[:, None])[:, 0]
            )
        return timelines
