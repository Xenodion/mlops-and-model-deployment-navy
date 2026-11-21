"""Run the MLflow server with its SQLite registry backed up to Cloud Storage.

Cloud Storage volumes cannot host SQLite (out-of-order journal writes), so the database lives on the container
disk: restored from the bucket at start, copied back (consistent snapshot) whenever it changed and at shutdown.
ponytail: one instance only; a hard kill can lose the last few seconds of writes. Cloud SQL would remove the limit
(disabled in the course project).
"""

import os
import signal
import sqlite3
import subprocess
import sys
import time

from google.cloud import storage

DB = "/tmp/mlflow.db"
SNAPSHOT = "/tmp/mlflow.snapshot.db"
POLL_S = 5
blob = storage.Client().bucket(os.environ["STATE_BUCKET"]).blob("mlflow-state/mlflow.db")


def backup() -> None:
    source, target = sqlite3.connect(DB), sqlite3.connect(SNAPSHOT)
    source.backup(target)  # consistent copy even if a request is writing
    source.close(), target.close()
    blob.upload_from_filename(SNAPSHOT)


if blob.exists():
    blob.download_to_filename(DB)
    print("registry restored from the bucket", flush=True)

server = subprocess.Popen(
    [
        sys.executable, "-m", "mlflow", "server", "--host", "0.0.0.0", "--port", os.environ.get("PORT", "8080"),
        "--workers", "1", "--backend-store-uri", f"sqlite:///{DB}",
        "--artifacts-destination", os.environ["ARTIFACT_ROOT"], "--serve-artifacts",
        "--allowed-hosts", "*.run.app,localhost:*,127.0.0.1:*",
    ]
)  # fmt: skip


def stop(*_: object) -> None:
    server.terminate()
    server.wait(timeout=20)
    if os.path.exists(DB):
        backup()
    sys.exit(0)


signal.signal(signal.SIGTERM, stop)
last = None
while server.poll() is None:
    time.sleep(POLL_S)
    if os.path.exists(DB) and (mtime := os.stat(DB).st_mtime_ns) != last:
        backup()
        last = mtime
sys.exit(server.returncode)
