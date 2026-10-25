"""Campaign registration, four-charter coverage and restart integrity."""
import datetime as dt
import json
import sqlite3

import pytest

from pod_platform.research.builtins import BuiltinContext, write_fixture_dataset
from pod_platform.strategies.base import Charter
from pod_platform.strategies.charter_campaign import run_charter_campaign

NOW = dt.datetime(2026, 10, 25, 12, tzinfo=dt.timezone.utc)


def test_all_charters_are_registered_before_labels_and_results_remain_test_only(tmp_path, monkeypatch):
    manifest = write_fixture_dataset(tmp_path / "inputs")
    state = tmp_path / "state"
    original = BuiltinContext.execute
    calls = []

    def checked(self, executor_id, request):
        with sqlite3.connect(state / "registry.sqlite") as db:
            assert db.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 4
            assert db.execute("SELECT COUNT(*) FROM experiment_specs").fetchone()[0] == 4
        calls.append(executor_id)
        return original(self, executor_id, request)

    monkeypatch.setattr(BuiltinContext, "execute", checked)
    report = run_charter_campaign(manifest, state, dataset_id="fixture_panel",
                                 research_asof=NOW, clock=lambda: NOW)
    assert {candidate["charter"] for candidate in report["candidates"]} == {c.value for c in Charter}
    assert len(calls) == 12
    assert report["fixture"] is True
    assert report["research_mode"] == "contemporary_research"
    assert all(c["status"] != "admitted" for c in report["candidates"])
    assert all(c["metrics"] for c in report["candidates"])
    replay = run_charter_campaign(manifest, state, dataset_id="fixture_panel",
                                 research_asof=NOW, clock=lambda: NOW)
    assert replay["report_hash"] == report["report_hash"]
    assert replay["trial_accounting"] == report["trial_accounting"]


def test_changed_implementation_cannot_reuse_a_registered_trial(tmp_path, monkeypatch):
    from pod_platform.research.builtins import executors
    from pod_platform.pods.runner import ExecutorRegistry, ExecutorError
    from .test_builtin_executors import make_request

    manifest = write_fixture_dataset(tmp_path / "inputs")
    registry = ExecutorRegistry()
    context = executors.register_executors(registry, manifest_path=manifest, state_dir=tmp_path / "state",
                                          clock=lambda: NOW, research_asof=NOW)
    result = registry.run(make_request())
    assert result.metrics
    monkeypatch.setattr(executors, "_implementation_hash", lambda: "0" * 64)
    with pytest.raises(ExecutorError):
        registry.run(make_request())


def test_closed_source_manifest_stops_before_any_research_results(tmp_path):
    manifest = write_fixture_dataset(tmp_path / "inputs")
    document = json.loads(manifest.read_text())
    document["datasets"]["fixture_panel"]["approved"] = False
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        run_charter_campaign(manifest, tmp_path / "state", dataset_id="fixture_panel",
                             research_asof=NOW, clock=lambda: NOW)
