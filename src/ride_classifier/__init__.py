"""Ride classifier: classify the IMU recording of a ride into 5 riding categories."""

import os

# Quieter MLflow, as in the labs: no "agent hint" on import, no progress bars in logs
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

# Download artifacts THROUGH the MLflow server. Otherwise MLflow 3.17 hands out direct S3 links built with the
# storage's Docker-internal name (http://s3:8333), unreachable from a laptop or CI: downloads hang and retry.
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
