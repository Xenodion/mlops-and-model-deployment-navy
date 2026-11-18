from ride_classifier.registry.tracking import dvc_md5, flatten, run_tags


def test_nested_params_become_dotted_names():
    assert flatten({"train": {"baseline": {"C": 0.05}, "model": "baseline"}, "seed": 42}) == {
        "train.baseline.C": 0.05,
        "train.model": "baseline",
        "seed": 42,
    }


def test_data_fingerprint_is_read_from_the_dvc_pointer(tmp_path):
    pointer = tmp_path / "rides.parquet.dvc"
    pointer.write_text("outs:\n- md5: f7eb702062e890c9f3c533e5eaa683b7\n  path: rides.parquet\n", encoding="utf-8")
    assert dvc_md5(pointer) == "f7eb702062e890c9f3c533e5eaa683b7"
    assert dvc_md5(tmp_path / "missing.dvc") == "unknown"


def test_every_run_records_which_code_and_which_data(tmp_path):
    tags = run_tags({"data_md5": tmp_path / "missing.dvc"})
    assert set(tags) == {"git_commit", "git_dirty", "data_md5"}
    assert tags["git_dirty"] in {"True", "False", "unknown"}
