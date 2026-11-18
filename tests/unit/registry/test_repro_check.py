from ride_classifier.config import ROOT, load_params
from ride_classifier.registry.repro_check import compare, data_problems
from ride_classifier.registry.tracking import dvc_md5

DATA = load_params()["data"]


def test_scores_are_compared_only_when_the_original_run_has_them():
    original = {"cv_macro_f1": 0.3726, "golden_macro_f1": 0.4773}
    rebuilt = {"cv_macro_f1": 0.3726, "golden_macro_f1": 0.4774, "new_metric": 1.0}
    deltas = compare(original, rebuilt)
    assert set(deltas) == {"cv_macro_f1", "golden_macro_f1"}
    assert deltas["cv_macro_f1"] == 0.0


def test_data_matches_when_the_run_recorded_the_same_fingerprints():
    tags = {
        "data_md5": dvc_md5(ROOT / f"{DATA['raw_rides']}.dvc"),
        "labels_md5": dvc_md5(ROOT / f"{DATA['raw_labels']}.dvc"),
    }
    assert data_problems(tags, DATA) == []


def test_other_data_is_reported_with_the_fix():
    tags = {"data_md5": "0" * 32, "labels_md5": dvc_md5(ROOT / f"{DATA['raw_labels']}.dvc")}
    problems = data_problems(tags, DATA)
    assert len(problems) == 1 and "make data" in problems[0]
