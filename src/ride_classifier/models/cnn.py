"""1D CNN on fixed-length windows of a ride (docs/PLAN.md §6.3). Same contract as the baseline (baseline.py).

The filters slide along time over the 6 sensor channels and learn short movement patterns (bumps, sharp turns,
wobbling). Each window gets 5 probabilities; a ride's prediction is the mean over its windows.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch import nn

from ride_classifier.config import ID_COLUMN, LABELS, SAMPLING_HZ
from ride_classifier.features.windows import windows_by_ride
from ride_classifier.models.timeline import Timeline, cumulative_mean, window_end_times


def export_onnx_bytes(net: nn.Module, n_channels: int, window_samples: int, *, quantize_int8: bool = False) -> bytes:
    """Export a fitted RideCNN network to ONNX bytes (FP32 or dynamic INT8)."""
    net.eval()
    sample = torch.zeros((1, n_channels, window_samples), dtype=torch.float32)
    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = Path(tmp) / "model.onnx"
        torch.onnx.export(
            net,
            sample,
            str(fp32_path),
            input_names=["windows"],
            output_names=["logits"],
            dynamic_axes={"windows": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        if not quantize_int8:
            return fp32_path.read_bytes()
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = Path(tmp) / "model_int8.onnx"
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
        return int8_path.read_bytes()


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

    def _get_ort_session(self):
        ort_session = getattr(self, "_ort_session", None)
        if ort_session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = getattr(self, "_onnx_threads", 2)
            options.inter_op_num_threads = 1
            ort_session = ort.InferenceSession(
                self._onnx_bytes, sess_options=options, providers=["CPUExecutionProvider"]
            )
            self._ort_session = ort_session
        return ort_session

    def enable_onnx(self, quantize_int8: bool = False, intra_op_threads: int = 2) -> CNNModel:
        """Compile the fitted CNN to ONNX Runtime (FP32 or dynamic INT8) for CPU inference."""
        if not hasattr(self, "net_") or not hasattr(self, "mean_"):
            raise RuntimeError("Fit the CNNModel before enabling ONNX inference")
        n_channels = int(self.mean_.shape[1])
        window_samples = round(self.window_s * SAMPLING_HZ)
        self._onnx_threads = intra_op_threads
        self._onnx_bytes = export_onnx_bytes(self.net_, n_channels, window_samples, quantize_int8=quantize_int8)
        self._ort_session = None
        self._get_ort_session()
        self._onnx_runtime = "onnx-int8" if quantize_int8 else "onnx"
        return self

    def disable_onnx(self) -> CNNModel:
        """Revert inference back to PyTorch."""
        self._onnx_bytes = None
        self._ort_session = None
        self._onnx_runtime = "pytorch"
        return self

    @property
    def inference_runtime(self) -> str:
        return getattr(self, "_onnx_runtime", "pytorch")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_ort_session", None)  # ONNX InferenceSession is recreated lazily from _onnx_bytes
        return state

    def _window_probas(self, windows: np.ndarray) -> np.ndarray:
        """(n_windows, channels, time) -> (n_windows, n_labels): softmax of the network on normalised windows."""
        x = ((windows - self.mean_) / self.std_).astype(np.float32)
        if getattr(self, "_onnx_bytes", None) is not None:
            logits = self._get_ort_session().run(None, {"windows": x})[0]
            return softmax(logits, axis=1)
        with torch.no_grad():
            return torch.softmax(self.net_(torch.from_numpy(x)), dim=1).numpy()

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
