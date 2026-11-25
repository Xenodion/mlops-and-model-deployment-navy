import json

import mlflow
import pytest
from mlflow import MlflowClient

from ride_classifier.registry.export_model import META_FILE, export
from ride_classifier.registry.register import PRODUCTION, STAGING, promote, register, registration_problems

CLEAN = {"git_dirty": "False", "gate": "passed"}


def test_a_clean_run_that_passed_the_gate_can_be_registered():
    assert registration_problems(CLEAN, {"golden_macro_f1": 0.45}, production_score=None) == []


def test_a_run_on_uncommitted_code_is_refused():
    problems = registration_problems({**CLEAN, "git_dirty": "True"}, {"golden_macro_f1": 0.45}, None)
    assert any("uncommitted" in p for p in problems)


def test_a_run_that_failed_the_gate_is_refused():
    problems = registration_problems({**CLEAN, "gate": "failed"}, {"golden_macro_f1": 0.45}, None)
    assert any("gate" in p for p in problems)


def test_a_model_worse_than_production_is_refused_an_equal_one_is_not():
    assert registration_problems(CLEAN, {"golden_macro_f1": 0.40}, production_score=0.45)
    assert registration_problems(CLEAN, {"golden_macro_f1": 0.45}, production_score=0.45) == []


@pytest.fixture(scope="module")
def registry(tmp_path_factory, ride_model):
    """A local MLflow store + registry, and a function logging a trained model in a run with given scores."""
    root = tmp_path_factory.mktemp("registry")
    tracking_uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.create_experiment("test", artifact_location=(root / "mlruns").as_uri())
    model = ride_model

    def log_training(golden_macro_f1: float, tags: dict = CLEAN) -> tuple[str, str]:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(experiment_id=experiment) as run:
            mlflow.set_tags({**tags, "git_commit": "abc1234", "data_md5": "f7eb70"})
            mlflow.log_metrics({"cv_macro_f1": 0.37, "golden_macro_f1": golden_macro_f1})
            info = mlflow.pyfunc.log_model(name="model", python_model=model)
        return run.info.run_id, info.model_uri

    return MlflowClient(tracking_uri), log_training, root


def test_register_promote_and_export(registry):
    client, log_training, root = registry
    run_id, model_uri = log_training(golden_macro_f1=0.45)

    version = register(client, run_id, model_uri)
    assert client.get_model_version_by_alias("ride-classifier", STAGING).version == version.version
    assert register(client, run_id, model_uri).version == version.version  # same run: same version, no duplicate

    promote(client)
    assert client.get_model_version_by_alias("ride-classifier", PRODUCTION).version == version.version

    meta = export(client, PRODUCTION, destination=root / "model")
    assert (root / "model" / "MLmodel").exists()
    assert json.loads((root / "model" / META_FILE).read_text()) == meta
    assert meta["run_id"] == run_id and meta["git_commit"] == "abc1234"


def test_a_worse_model_than_production_is_not_registered(registry):
    client, log_training, _ = registry
    register(client, *log_training(golden_macro_f1=0.45))
    promote(client)  # production now scores 0.45 on the golden rides
    with pytest.raises(SystemExit, match="below the production model"):
        register(client, *log_training(golden_macro_f1=0.30))
