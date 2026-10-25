"""Component evidence can test a native earnings mechanism with minimal financial data."""

import csv
import datetime as dt
import hashlib
import json
import random
import sqlite3

import pytest

from pod_platform.research.builtins.errors import BuiltinConfigError, SourceNotEligible, VariantBudgetExhausted
from pod_platform.research.builtins.fixtures import fixture_manifest
from pod_platform.research.builtins.manifest import load_manifest
from pod_platform.research.builtins.timeutil import month_end_date, month_index
from pod_platform.research.component_probe import ComponentProbeService, ComponentProbeSpec, ProbeBudgets


NOW = dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)


def minimal_panel(tmp_path, *, sign=1, missing=False, source_research_only=False):
    rng = random.Random(27)
    first = month_index(dt.date(2015, 1, 1))
    # Future earnings innovation is exactly a component known three months earlier.
    signal = {(m, i): rng.gauss(0, 1) for m in range(-3, 12) for i in range(30)}
    path = tmp_path / "earnings.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["permno", "eom", "niq_su", "sale_gr1"])
        for m in range(12):
            for i in range(30):
                writer.writerow([i, month_end_date(first + m),
                                 "" if missing else sign * signal[m - 3, i], signal[m, i]])
        # Protected values are intentionally invalid. Only the date may be inspected.
        writer.writerow(["PROTECTED", "2019-01-31", "DO NOT PARSE", "DO NOT PARSE"])
    manifest = fixture_manifest(path.name)
    ds = manifest["datasets"]["fixture_panel"]
    ds["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if source_research_only:
        ds["source_card"]["formation_eligible"] = False
        ds["source_card"]["status"] = "SAMPLED"
        ds["source_card"]["unresolved_questions"] = ["financial bridge not established"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    service = ComponentProbeService(manifest_path, tmp_path / "state", clock=lambda: NOW)
    spec = ComponentProbeSpec(namespace="earnings", probe_name="sales_probe", dataset_id="fixture_panel",
                              recipe_id="fund_earnings_innovation_v1", feature_names=["sale_gr1_rank"],
                              start=dt.date(2015, 1, 1), end=dt.date(2015, 12, 31), research_asof=NOW,
                              evidence_refs=["paper:mechanism:forecast_sales"], budgets=ProbeBudgets(max_names=20))
    return service, spec, manifest_path, path


def test_native_earnings_positive_without_return_sector_size_or_exposure_data(tmp_path):
    service, spec, _, _ = minimal_panel(tmp_path)
    plan = service.plan(spec)
    assert plan["required_columns"] == ["niq_su", "sale_gr1"]
    assert plan["promotable"] is False
    assert "ret_exc_lead1m" in plan["optional_additional_controls"]
    registered = service.register(spec)
    result = service.run(registered["probe_id"])
    assert result["status"] == "positive_signal"
    assert result["metrics"]["mean_feature_target_ic"] == pytest.approx(1)
    assert result["evaluated_months"] == 9
    assert result["loaded_names"] == 20
    assert result["loaded_rows"] == 240
    assert result["scope"] == "research_only"
    assert "UNTESTED" in result["original_mechanism"]
    assert all(key not in result for key in ("admission_id", "model", "forecast", "profitability"))
    assert service.run(registered["probe_id"]) == result
    assert ComponentProbeService(service.manifest_path, service.state_dir, lambda: NOW).run(registered["probe_id"]) == result
    with sqlite3.connect(service.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 1


def test_negative_is_completed_retained_evidence(tmp_path):
    service, spec, _, _ = minimal_panel(tmp_path, sign=-1)
    result = service.run(service.register(spec)["probe_id"])
    assert result["status"] == "negative"
    assert result["metrics"]["mean_feature_target_ic"] == pytest.approx(-1)
    assert service.run(result["probe_id"]) == result


def test_minimal_manifest_needs_no_unused_financial_declarations(tmp_path):
    service, spec, manifest_path, _ = minimal_panel(tmp_path)
    raw = json.loads(manifest_path.read_text())
    ds = raw['datasets']['fixture_panel']
    for name in ('label', 'evaluation', 'sector_column', 'market_cap_column', 'exposure_columns', 'column_aliases'):
        ds.pop(name, None)
    ds['columns'] = {k: v for k, v in ds['columns'].items() if k in ('niq_su', 'sale_gr1')}
    manifest_path.write_text(json.dumps(raw))
    assert service.run(service.register(spec)['probe_id'])['status'] == 'positive_signal'
    with pytest.raises(BuiltinConfigError, match='financial execution requires'):
        load_manifest(manifest_path)
    native_returns = spec.model_copy(update={'recipe_id': 'dynamics_reversal_liquidity_v1',
                                             'feature_names': ['ret_12_1_rank']})
    with pytest.raises(BuiltinConfigError, match='intrinsically needs'):
        service.plan(native_returns)


def test_missing_native_outcomes_are_inconclusive_not_zero_filled(tmp_path):
    service, spec, _, _ = minimal_panel(tmp_path, missing=True)
    result = service.run(service.register(spec)["probe_id"])
    assert result["status"] == "inconclusive"
    assert result["evaluated_pairs"] == 0
    assert result["metrics"]["mean_feature_target_ic"] is None


def test_feature_target_mode_measures_association_instead_of_baseline_difference(tmp_path):
    service, spec, manifest_path, path = minimal_panel(tmp_path)
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row['permno'] != 'PROTECTED':
            row['niq_su'] = row['permno']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    raw = json.loads(manifest_path.read_text())
    raw['datasets']['fixture_panel']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(raw))
    spec.feature_names = ['niq_su_rank']
    spec.component_kind = 'feature_target'
    result = service.run(service.register(spec)['probe_id'])
    assert result['primary_metric'] == 'feature_target_ic'
    assert result['status'] == 'positive_signal'
    assert result['metrics']['primary_mean'] == pytest.approx(1)
    assert result['metrics']['mean_incremental_ic'] == pytest.approx(0)


def test_plan_and_registration_do_not_parse_rows(tmp_path, monkeypatch):
    service, spec, _, _ = minimal_panel(tmp_path)
    import pod_platform.research.component_probe as module
    monkeypatch.setattr(module, "_iter_csv", lambda *_: pytest.fail("row access before counted trial"))
    service.plan(spec)
    service.register(spec)
    with sqlite3.connect(service.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM registrations").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 0


def test_counted_trial_exists_before_rows_and_failed_trials_retained(tmp_path, monkeypatch):
    service, spec, _, _ = minimal_panel(tmp_path)
    import pod_platform.research.component_probe as module
    def read(*_):
        with sqlite3.connect(service.path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM registrations").fetchone()[0] == 1
        raise RuntimeError("read failed")
    monkeypatch.setattr(module, "_iter_csv", read)
    result = service.run(service.register(spec)["probe_id"])
    assert result["status"] == "failed"
    assert service.run(result["probe_id"]) == result


def test_budget_namespace_is_persistent_and_registration_immutable(tmp_path):
    service, spec, _, _ = minimal_panel(tmp_path)
    spec.budgets.max_trials = 1
    first = service.register(spec)
    service.run(first["probe_id"])
    altered = spec.model_copy(deep=True)
    altered.feature_names = ["niq_su_rank"]
    with pytest.raises(BuiltinConfigError, match="immutable"):
        service.register(altered)
    altered.probe_name = "second"
    second = service.register(altered)
    with pytest.raises(VariantBudgetExhausted):
        service.run(second["probe_id"])
    altered.probe_name = "budget_bypass"
    altered.budgets.max_trials = 2
    with pytest.raises(VariantBudgetExhausted):
        service.register(altered)
    with sqlite3.connect(service.path) as conn:
        for table in ("namespaces", "registrations", "trials", "results"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(f"DELETE FROM {table}")


def test_data_and_manifest_changes_fail_after_counted_reservation(tmp_path):
    service, spec, manifest_path, data_path = minimal_panel(tmp_path)
    registered = service.register(spec)
    data_path.write_text(data_path.read_text().replace("DO NOT PARSE", "changed"))
    report = service.run(registered["probe_id"])
    assert report["status"] == "failed"
    assert "dataset hash" in report["error"]
    new = spec.model_copy(update={"probe_name": "changed_source"})
    registered = service.register(new)
    raw = json.loads(manifest_path.read_text())
    raw["datasets"]["fixture_panel"]["source_snapshot_id"] = "different"
    manifest_path.write_text(json.dumps(raw))
    report = service.run(registered["probe_id"])
    assert report["status"] == "failed"
    with sqlite3.connect(service.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 2


def test_research_source_opt_in_does_not_weaken_default_gate(tmp_path):
    service, spec, manifest_path, _ = minimal_panel(tmp_path, source_research_only=True)
    with pytest.raises(SourceNotEligible):
        load_manifest(manifest_path)
    result = service.run(service.register(spec)["probe_id"])
    assert result["status"] == "positive_signal"
    raw = json.loads(manifest_path.read_text())
    raw["datasets"]["fixture_panel"]["source_card"]["status"] = "REJECTED"
    manifest_path.write_text(json.dumps(raw))
    with pytest.raises(SourceNotEligible):
        service.plan(spec)


def test_protected_unknown_feature_and_missing_native_measurement_denied(tmp_path):
    service, spec, manifest_path, _ = minimal_panel(tmp_path)
    for updates in ({"partition": "check"}, {"end": "2019-01-31"}, {"feature_names": ["write_code"]}):
        with pytest.raises((ValueError, BuiltinConfigError)):
            service.plan({**spec.model_dump(mode="json"), **updates})
    raw = json.loads(manifest_path.read_text())
    del raw["datasets"]["fixture_panel"]["columns"]["sale_gr1"]
    manifest_path.write_text(json.dumps(raw))
    with pytest.raises(BuiltinConfigError, match="UNTESTED"):
        service.plan(spec)


def test_interrupted_trial_not_silently_rerun(tmp_path):
    service, spec, _, _ = minimal_panel(tmp_path)
    probe_id = service.register(spec)["probe_id"]
    with sqlite3.connect(service.path) as conn:
        conn.execute("INSERT INTO trials VALUES (?,?,?)", (probe_id, spec.namespace, NOW.isoformat()))
    report = service.run(probe_id)
    assert report["status"] == "failed"
    assert "interrupted" in report["error"]
    with sqlite3.connect(service.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 1
