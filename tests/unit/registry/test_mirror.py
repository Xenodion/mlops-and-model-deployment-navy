"""make k8s-model: registered models copied between two MLflow servers (here two SQLite stores)."""

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from ride_classifier.config import ID_COLUMN, MODEL_NAME, SENSOR_COLUMNS, TIME_COLUMN
from ride_classifier.pipeline.train import PACKAGE_DIR
from ride_classifier.registry.mirror import RUN_TAG, mirror, mirror_all


def server(tmp_path, name: str) -> MlflowClient:
    """An empty MLflow with the experiment's files kept in tmp_path (not in ./mlruns)."""
    client = MlflowClient(f"sqlite:///{(tmp_path / f'{name}.db').as_posix()}")
    client.create_experiment("ride-classifier", artifact_location=(tmp_path / f"{name}-files").as_uri())
    return client


@pytest.fixture
def source(tmp_path, ride_model):
    """v1 (@production) and v2 (@staging), each from its own committed, gate-passing run."""
    client = server(tmp_path, "source")
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(client.tracking_uri)
    experiment = client.get_experiment_by_name("ride-classifier").experiment_id
    for score in (0.4, 0.5):
        with mlflow.start_run(experiment_id=experiment):
            mlflow.set_tags({"git_dirty": "False", "gate": "passed", "git_commit": "abc"})
            mlflow.log_param("C", 0.05)
            for step, loss in enumerate((1.0, 0.5)):
                mlflow.log_metric("train_loss", loss, step=step)
            mlflow.log_metric("golden_macro_f1", score)
            mlflow.log_text("scores", "report.txt")
            info = mlflow.pyfunc.log_model(
                name="model", python_model=ride_model, code_paths=[str(PACKAGE_DIR)], pip_requirements=["pandas"]
            )
        mlflow.register_model(info.model_uri, MODEL_NAME, tags={"golden_macro_f1": str(score)})
    client.set_registered_model_alias(MODEL_NAME, "production", "1")
    client.set_registered_model_alias(MODEL_NAME, "staging", "2")
    yield client
    mlflow.set_tracking_uri(previous)


def predict(client: MlflowClient, ride: pd.DataFrame) -> pd.DataFrame:
    mlflow.set_tracking_uri(client.tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production").predict(ride)


def test_versions_runs_and_aliases_are_copied_with_their_numbers(source, tmp_path, labelled_rides):
    target = server(tmp_path, "target")
    assert mirror(source, target) == {"1": "1", "2": "2"}

    aliases = target.get_registered_model(MODEL_NAME).aliases
    assert {alias: str(number) for alias, number in aliases.items()} == {"production": "1", "staging": "2"}
    original, copy = source.get_model_version(MODEL_NAME, "1"), target.get_model_version(MODEL_NAME, "1")
    run = target.get_run(copy.run_id)
    assert run.data.tags[RUN_TAG] == original.run_id
    assert run.data.params == {"C": "0.05"}
    assert [m.value for m in target.get_metric_history(copy.run_id, "train_loss")] == [1.0, 0.5]
    assert [a.path for a in target.list_artifacts(copy.run_id)] == ["report.txt"]
    assert copy.tags["golden_macro_f1"] == "0.4" and copy.tags["run_id"] == copy.run_id

    readings = labelled_rides[0]
    ride = readings[readings[ID_COLUMN] == "Tandem-1"][[ID_COLUMN, TIME_COLUMN, *SENSOR_COLUMNS]]
    pd.testing.assert_frame_equal(predict(target, ride), predict(source, ride))  # the same model, exactly


def test_running_it_again_copies_nothing_twice(source, tmp_path):
    target = server(tmp_path, "target")
    mirror(source, target)
    source.set_registered_model_alias(MODEL_NAME, "production", "2")  # promoted meanwhile

    assert mirror(source, target) == {"1": "1", "2": "2"}
    assert len(target.search_model_versions(f"name='{MODEL_NAME}'")) == 2
    assert len(target.search_runs([target.get_experiment_by_name("ride-classifier").experiment_id])) == 2
    assert str(target.get_registered_model(MODEL_NAME).aliases["production"]) == "2"


def test_versions_that_break_the_registration_rules_are_not_copied(source, tmp_path):
    source.set_tag(source.get_model_version(MODEL_NAME, "2").run_id, "git_dirty", "True")
    target = server(tmp_path, "target")

    with pytest.raises(SystemExit, match="registration rules"):
        mirror(source, target)
    assert target.search_model_versions(f"name='{MODEL_NAME}'") == []


def test_files_added_to_a_run_later_are_copied_on_the_next_run(source, tmp_path):
    """E.g. the drift reference, logged with model 1's run by the `reference` stage after the training."""
    target = server(tmp_path, "target")
    mirror(source, target)
    source.log_text(source.get_model_version(MODEL_NAME, "2").run_id, "{}", "drift_reference.json")

    mirror(source, target)
    copy = target.get_model_version(MODEL_NAME, "2")
    assert "drift_reference.json" in {a.path for a in target.list_artifacts(copy.run_id)}


def test_a_copy_can_be_the_source_without_duplicating_anything(source, tmp_path):
    """dev -> shared, then shared -> cluster: copies are matched by the id of the original run."""
    shared, cluster = server(tmp_path, "shared"), server(tmp_path, "cluster")
    mirror(source, cluster)
    mirror(source, shared)

    assert mirror(shared, cluster) == {"1": "1", "2": "2"}
    assert len(cluster.search_model_versions(f"name='{MODEL_NAME}'")) == 2
    assert len(cluster.search_runs([cluster.get_experiment_by_name("ride-classifier").experiment_id])) == 2


def test_model_2_is_optional_but_model_1_is_not(source, tmp_path):
    target = server(tmp_path, "target")
    assert set(mirror_all(source, target, [MODEL_NAME, "fall-risk"])) == {MODEL_NAME}  # no fall-risk: skipped

    with pytest.raises(SystemExit, match="No registered model"):
        mirror_all(server(tmp_path, "empty"), target, [MODEL_NAME])
