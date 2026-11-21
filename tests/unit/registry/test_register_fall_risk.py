import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from ride_classifier.models.fall_risk import FallRiskModel
from ride_classifier.models.fall_risk_pyfunc import FallRiskPyfunc
from ride_classifier.registry.register_fall_risk import register


def test_register_and_load_fall_risk(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.create_experiment("fall-risk-test", artifact_location=(tmp_path / "artifacts").as_uri())
    features = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    model = FallRiskModel(1.0, 0).fit(features, pd.Series([0, 0, 1, 1]))
    with mlflow.start_run(experiment_id=experiment) as run:
        mlflow.set_tags({"git_dirty": "False", "git_commit": "abc", "srv3490_md5": "123", "gate": "passed"})
        mlflow.log_metric("gate_passed", 1.0)
        info = mlflow.pyfunc.log_model(name="model", python_model=FallRiskPyfunc(model))
    client = MlflowClient(uri)
    version = register(client, run.info.run_id, info.model_uri)
    assert register(client, run.info.run_id, info.model_uri).version == version.version
    assert client.get_model_version_by_alias("fall-risk", "staging").version == version.version
    actual = mlflow.pyfunc.load_model("models:/fall-risk@staging").predict(features)["fall_risk"]
    assert actual.to_numpy() == pytest.approx(model.predict_risk(features))

    client.set_tag(run.info.run_id, "gate", "failed")
    with pytest.raises(SystemExit, match="passed its gate"):
        register(client, run.info.run_id, info.model_uri)
