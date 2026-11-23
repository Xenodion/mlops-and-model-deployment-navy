from types import SimpleNamespace

import pytest

from ride_classifier.registry.retraining_trigger import record_trigger


class FakeClient:
    def __init__(self, *, commit="abc", dirty="False", status="FINISHED"):
        self.run = SimpleNamespace(
            info=SimpleNamespace(status=status), data=SimpleNamespace(tags={"git_dirty": dirty, "git_commit": commit})
        )
        self.tags = {}

    def get_run(self, run_id):
        return self.run

    def set_tag(self, run_id, key, value):
        self.tags[key] = value


def test_records_reason_and_evidence_for_a_new_clean_run():
    client = FakeClient()
    record_trigger(client, "new", "old", "abc", "input_drift", "report 123", "workflow/42")
    assert client.tags == {
        "trigger_reason": "input_drift",
        "trigger_evidence": "report 123",
        "trigger_workflow": "workflow/42",
    }


@pytest.mark.parametrize(
    ("run_id", "previous", "commit", "reason", "evidence", "client"),
    [
        ("old", "old", "abc", "manual", "", FakeClient()),
        ("new", "old", "abc", "input_drift", "", FakeClient()),
        ("new", "old", "other", "manual", "", FakeClient()),
        ("new", "old", "abc", "manual", "", FakeClient(dirty="True")),
        ("new", "old", "abc", "manual", "", FakeClient(status="FAILED")),
        ("new", "old", "abc", "unknown", "", FakeClient()),
    ],
)
def test_rejects_stale_or_untraceable_runs(run_id, previous, commit, reason, evidence, client):
    with pytest.raises(ValueError):
        record_trigger(client, run_id, previous, commit, reason, evidence, "")
    assert client.tags == {}
