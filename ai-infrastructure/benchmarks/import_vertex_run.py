"""Copy a completed Vertex benchmark run and its raw measurements to shared MLflow."""

import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from mlflow import MlflowClient
from mlflow.entities import Param


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=int)
    parser.add_argument("archive", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    target = MlflowClient(os.environ["MLFLOW_TRACKING_URI"])
    experiment = target.get_experiment_by_name("ai-infrastructure")
    experiment_id = experiment.experiment_id if experiment else target.create_experiment("ai-infrastructure")
    existing = target.search_runs([experiment_id], f"tags.vertex_job_id = '{args.job}'")
    with tempfile.TemporaryDirectory() as directory:
        with tarfile.open(args.archive) as archive:
            archive.extractall(directory, filter="data")
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        source = MlflowClient(Path(directory).as_uri())
        candidates = source.search_runs([e.experiment_id for e in source.search_experiments()])
        finished = [run for run in candidates if run.info.status == "FINISHED"]
        if not finished:
            raise ValueError("No finished worker runs found")
        measurements = Path(directory) / "measurements"
        measurements.mkdir()
        for path in args.results.rglob("*"):
            if path.is_file() and path.suffix in {".csv", ".json"}:
                destination = measurements / path.relative_to(args.results)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        for run in finished:
            copied = next((r for r in existing if r.data.tags.get("source_run_id") == run.info.run_id), None)
            if copied and copied.info.status == "FINISHED":
                print(copied.info.run_id)
                continue
            if copied is None:
                copied = target.create_run(
                    experiment_id,
                    start_time=run.info.start_time,
                    tags={"vertex_job_id": str(args.job), "source_run_id": run.info.run_id},
                    run_name=f"vertex-{args.job}-{run.info.run_id[:8]}",
                )
            run_id = copied.info.run_id
            try:
                history = [
                    point for key in run.data.metrics for point in source.get_metric_history(run.info.run_id, key)
                ]
                recorded = [point for key in run.data.metrics for point in target.get_metric_history(run_id, key)]
                target.log_batch(
                    run_id,
                    metrics=[p for p in history if p not in recorded],
                    params=[Param(k, v) for k, v in run.data.params.items()],
                )
                target.log_artifacts(run_id, str(measurements), "measurements")
                target.set_terminated(run_id, end_time=run.info.end_time)
            except Exception:
                target.set_terminated(run_id, status="FAILED")
                raise
            print(run_id)


if __name__ == "__main__":
    main()
