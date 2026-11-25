"""Run the release API with its classifier network on the requested device."""

import json
import os
import signal
from contextlib import nullcontext
from pathlib import Path

import mlflow
import torch
import uvicorn


def main():
    device = os.environ["BENCH_DEVICE"]
    original_load = mlflow.pyfunc.load_model
    forwards = 0

    def load(*args, **kwargs):
        model = original_load(*args, **kwargs)
        estimator = model.unwrap_python_model().model
        if hasattr(estimator, "net_"):
            network = estimator.net_.to(device)
            original_forward = network.forward

            def forward(inputs):
                nonlocal forwards
                result = original_forward(inputs.to(device))
                if result.device.type != device:
                    raise RuntimeError("Classifier used the wrong device")
                forwards += 1
                return result.cpu()

            network.forward = forward
        return model

    mlflow.pyfunc.load_model = load
    server = uvicorn.Server(uvicorn.Config("ride_classifier.api.main:app", host="127.0.0.1", port=18172, access_log=False))
    server.capture_signals = nullcontext
    signal.signal(signal.SIGTERM, lambda *_: setattr(server, "should_exit", True))
    try:
        server.run()
    finally:
        stats = {"classifier_forwards": forwards, "device": device}
        if device == "cuda":
            stats.update(gpu=torch.cuda.get_device_name(), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        Path(os.environ["BENCH_DEVICE_REPORT"]).write_text(json.dumps(stats))


if __name__ == "__main__":
    main()
