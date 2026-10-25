"""Offline tests for the built-in charter executors (dataset manifest, strict loader, walk-forward engine, ledger).

Everything runs on a synthetic CSV fixture written under tmp_path. Parquet cases are skipped without pyarrow.
Nothing here calls a model or the network.
"""

import copy
import csv
import datetime as dt
import json
import math
import os
import sqlite3
from pathlib import Path

import pytest

from pod_platform.agents.durable_cache import DurableExtractionCache
from pod_platform.agents.provider import DeterministicMockProvider
from pod_platform.pods import (
    ExecutorRegistry,
    ExperimentPlan,
    ExperimentVariant,
    FeatureDefinition,
    PodResearchRunner,
    ResearchProposal,
    pod_spec,
)
from pod_platform.pods.runner import ExecutorError, ExperimentRequest
from pod_platform.research.builtins import (
    EXECUTOR_IDS,
    BuiltinConfigError,
    BuiltinLedger,
    DataIntegrityError,
    PartitionDenied,
    SourceNotEligible,
    catalog_summary,
    load_manifest,
    register_executors,
    write_fixture_dataset,
)
from pod_platform.research.builtins import stats
from pod_platform.research.builtins.fixtures import FIELDS, FIXTURE_DATASET_ID, generate_rows
from pod_platform.research.builtins.loader import load_panel
from pod_platform.research.builtins.timeutil import month_index
from pod_platform.research.memory import PodMemory
from pod_platform.strategies.base import Charter

UTC = dt.timezone.utc
ASOF = dt.datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
NOW = dt.datetime(2026, 10, 25, 13, 0, tzinfo=UTC)
DYN = EXECUTOR_IDS[Charter.MARKET_DYNAMICS.value]
FUND = EXECUTOR_IDS[Charter.FUNDAMENTALS_EXPECTATIONS.value]
SOURCE_ID = "fixture_synthetic_panel"


def clock():
    return NOW


# ------------------------------------------------------------------ helpers


def make_proposal(executor_id, variants):
    return ResearchProposal(
        title="Fixture test proposal",
        mechanism="A mechanism that is long enough to pass the schema validation.",
        falsifiable_target="Candidate rank IC is not above the declared baseline on design.",
        baseline="Simple baseline",
        features=[FeatureDefinition(name="f_one", definition="Rank of a panel column", timing="Known at month end", source_ids=[SOURCE_ID])],
        evidence_refs=[SOURCE_ID],
        experiment_plan=ExperimentPlan(
            executor_id=executor_id, primary_metric="belief_ic_improvement", data_partition="design",
            variants=variants, description="Run the registered built-in executor.",
        ),
        competing_explanation="Common style exposure explains it.",
        limitations=["Fixture data only"],
        evidence_quality="weak",
    )


def make_request(executor_id=DYN, name="v1", draft="draft1", ns="pod_dynamics", **params):
    variant = ExperimentVariant(name=name, params=params)
    return ExperimentRequest(ns, draft, executor_id, variant, make_proposal(executor_id, [variant]))


def setup(tmp_path, manifest=None, **kwargs):
    manifest = manifest or write_fixture_dataset(tmp_path / "data")
    registry = ExecutorRegistry()
    kwargs.setdefault("clock", clock)
    kwargs.setdefault("research_asof", ASOF)
    ctx = register_executors(registry, manifest_path=manifest, state_dir=tmp_path / "state", **kwargs)
    return registry, ctx


def edit_manifest(path, fn, name="edited_manifest.json"):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    fn(doc, doc["datasets"][FIXTURE_DATASET_ID])
    out = Path(path).with_name(name)
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out


def rewrite_csv(path, fn, extra_fields=(), drop_fields=()):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = [f for f in reader.fieldnames if f not in drop_fields]
        fields += [f for f in extra_fields if f not in fields]
        rows = list(reader)
    rows = fn(rows) or rows
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_metrics(registry, ctx, **kw):
    result = registry.run(make_request(**kw))
    return result, ctx.report(result.registration_id, kw.get("name", "v1"))


# ------------------------------------------------------------------ registration / env


def test_four_executors_registered_with_stable_ids(tmp_path):
    registry, _ = setup(tmp_path)
    assert registry.ids() == tuple(sorted(EXECUTOR_IDS.values()))
    assert set(EXECUTOR_IDS) == {c.value for c in Charter}
    assert EXECUTOR_IDS[Charter.MARKET_DYNAMICS.value] == "builtin_market_dynamics_v1"
    assert EXECUTOR_IDS[Charter.FUNDAMENTALS_EXPECTATIONS.value] == "builtin_fundamentals_expectations_v1"


def test_manifest_comes_only_from_argument_or_env(tmp_path, monkeypatch):
    manifest = write_fixture_dataset(tmp_path / "data")
    monkeypatch.delenv("POD_DATASET_MANIFEST", raising=False)
    with pytest.raises(BuiltinConfigError):
        register_executors(ExecutorRegistry(), state_dir=tmp_path / "s")
    monkeypatch.setenv("POD_DATASET_MANIFEST", str(manifest))
    monkeypatch.setenv("POD_BUILTIN_STATE_DIR", str(tmp_path / "envstate"))
    registry = ExecutorRegistry()
    ctx = register_executors(registry, clock=clock, research_asof=ASOF)
    assert len(registry.ids()) == 4
    assert ctx.ledger.path == tmp_path / "envstate" / "builtin_research.sqlite"
    monkeypatch.setenv("POD_DATASET_MANIFEST", str(tmp_path / "missing.json"))
    with pytest.raises(BuiltinConfigError):
        register_executors(ExecutorRegistry())


@pytest.mark.parametrize("charter", list(Charter))
def test_each_charter_has_a_runnable_evaluated_candidate(tmp_path, charter):
    registry, ctx = setup(tmp_path)
    executor_id = EXECUTOR_IDS[charter.value]
    result = registry.run(make_request(executor_id, ns=pod_spec(charter).namespace, dataset_id="fixture_panel"))
    assert result.executor_id == executor_id and result.data_partition == "design"
    assert result.registration_id == f"{pod_spec(charter).namespace}:draft1"
    assert len(result.spec_hash) == 64
    m = result.metrics
    for key in (
        "belief_ic_candidate", "belief_ic_baseline", "belief_ic_improvement", "belief_ic_improvement_t",
        "bridge_ic_candidate", "bridge_ic_baseline", "bridge_net_spread_candidate", "bridge_net_spread_baseline",
        "bridge_turnover_candidate", "coverage_eval_months", "coverage_names_per_month_min",
        "sector_long_max_share", "sector_neutral_belief_ic_candidate", "exposure_corr_candidate_me",
        "exposure_corr_baseline_ret_12_1", "trial_fold_evaluations",
    ):
        assert key in m and math.isfinite(m[key]), key
    assert m["coverage_eval_months"] >= 12
    assert 0.2 < m["bridge_short_eligible_share"] < 0.9
    assert "retrospective_design_partition_not_fresh_holdout" in result.notes
    report = ctx.report(result.registration_id, "v1")
    assert report["fixture"] is True and report["partition"] == "design"
    assert report["labels"]["proxy_labels"] and report["limitations"]
    assert report["limitations"][0].startswith("FIXTURE DATA")
    assert report["monthly"] and report["spec_hash"] == result.spec_hash
    assert report["data"]["rows_after_window_discarded_unparsed"] > 0


def test_sole_dataset_may_be_omitted_but_not_ambiguous(tmp_path):
    registry, ctx = setup(tmp_path)
    assert registry.run(make_request()).metrics  # sole approved dataset is the default
    manifest = ctx.manifest.path

    def second(doc, ds):
        doc["datasets"]["other_panel"] = copy.deepcopy(ds)

    registry2 = ExecutorRegistry()
    register_executors(registry2, manifest_path=edit_manifest(manifest, second), state_dir=tmp_path / "s2", clock=clock, research_asof=ASOF)
    with pytest.raises(ExecutorError, match="dataset_id is required"):
        registry2.run(make_request())


# ------------------------------------------------------------------ parameters cannot reach paths, code or labels


@pytest.mark.parametrize(
    "params",
    [
        {"path": "/etc/passwd"},
        {"code": "import os"},
        {"label_column": "ret_1_0"},
        {"dataset_id": "../panel"},
        {"dataset_id": "no_such_dataset"},
        {"recipe_id": "fund_earnings_innovation_v1"},  # recipe of another charter
        {"recipe_id": "not_a_recipe"},
        {"ridge_lambda": 3.0},
        {"method_id": "xgboost"},
        {"feature_set": "everything"},
        {"method_id": "rank_blend", "ridge_lambda": 0.05},
    ],
)
def test_variant_parameters_are_whitelisted(tmp_path, params):
    registry, ctx = setup(tmp_path)
    with pytest.raises(ExecutorError):
        registry.run(make_request(**params))
    assert ctx.ledger.variants_used("pod_dynamics:draft1") == 0


def test_protected_partitions_are_denied(tmp_path):
    registry, ctx = setup(tmp_path)
    for partition in ("check", "replay", "prospective", "final"):
        with pytest.raises(ExecutorError):
            registry.run(make_request(partition=partition))
    with pytest.raises(ExecutorError, match="PARTITION_ACCESS_DENIED"):
        registry.run(make_request(partition="check"))
    dataset = ctx.manifest.dataset("fixture_panel")
    with pytest.raises(PartitionDenied):
        load_panel(dataset, {"ret_1_0"}, partition="replay", research_asof=ASOF)


def test_label_not_mature_does_not_consume_budget(tmp_path):
    registry, ctx = setup(tmp_path, research_asof=dt.datetime(2018, 6, 30, tzinfo=UTC))
    with pytest.raises(ExecutorError, match="LABEL_NOT_MATURE"):
        registry.run(make_request())
    assert ctx.ledger.variants_used("pod_dynamics:draft1") == 0
    registry2, ctx2 = setup(tmp_path / "later", research_asof=NOW + dt.timedelta(days=5))
    with pytest.raises(ExecutorError, match="future"):
        registry2.run(make_request())


# ------------------------------------------------------------------ manifest: paths, approval, source gate, partitions


def test_manifest_rejects_traversal_and_escape(tmp_path):
    manifest = write_fixture_dataset(tmp_path / "data")
    with pytest.raises(BuiltinConfigError, match="traversal"):
        load_manifest(edit_manifest(manifest, lambda d, ds: ds.update(path="../panel.csv")))
    with pytest.raises(BuiltinConfigError, match="unsafe"):
        load_manifest(edit_manifest(manifest, lambda d, ds: ds.update(path="sub\\panel.csv")))
    with pytest.raises(BuiltinConfigError, match="unsafe"):
        load_manifest(edit_manifest(manifest, lambda d, ds: ds.update(path="~/panel.csv")))
    with pytest.raises(BuiltinConfigError, match="does not exist"):
        load_manifest(edit_manifest(manifest, lambda d, ds: ds.update(path="absent.csv")))

    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "panel.csv").write_text((tmp_path / "data" / "panel.csv").read_text(encoding="utf-8"), encoding="utf-8")
    absolute = str(other / "panel.csv")
    with pytest.raises(BuiltinConfigError, match="outside"):
        load_manifest(edit_manifest(manifest, lambda d, ds: ds.update(path=absolute)))
    ok = load_manifest(edit_manifest(manifest, lambda d, ds: (ds.update(path=absolute), d.update(allowed_roots=[str(other)]))))
    assert ok.dataset("fixture_panel").path == (other / "panel.csv").resolve()
    with pytest.raises(BuiltinConfigError, match="allowed_roots"):
        load_manifest(edit_manifest(manifest, lambda d, ds: d.update(allowed_roots=["relative/dir"])))

    link = tmp_path / "data" / "link.csv"
    try:
        link.symlink_to(other / "panel.csv")
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(BuiltinConfigError, match="outside"):
        load_manifest(edit_manifest(manifest, lambda d, ds: ds.update(path="link.csv")))
    card = tmp_path / "data" / "card.json"
    card.write_text("{}", encoding="utf-8")

    def card_traversal(d, ds):
        ds.pop("source_card")
        ds["source_card_path"] = "../data/card.json"

    with pytest.raises(BuiltinConfigError, match="traversal"):
        load_manifest(edit_manifest(manifest, card_traversal))


def test_dataset_must_be_operator_approved(tmp_path):
    manifest = write_fixture_dataset(tmp_path / "data")
    unapproved = edit_manifest(manifest, lambda d, ds: ds.update(approved=False))
    loaded = load_manifest(unapproved)
    assert loaded.datasets == {}
    with pytest.raises(BuiltinConfigError, match="not approved"):
        loaded.dataset("fixture_panel")
    with pytest.raises(BuiltinConfigError, match="not in the operator manifest"):
        loaded.dataset("ghost")
    registry = ExecutorRegistry()
    register_executors(registry, manifest_path=unapproved, state_dir=tmp_path / "s", clock=clock, research_asof=ASOF)
    with pytest.raises(ExecutorError, match="not approved"):
        registry.run(make_request(dataset_id="fixture_panel"))


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda c, ds: c.update(status="SAMPLED", formation_eligible=False), "not formation-eligible"),
        (lambda c, ds: c.update(status="ASSESSED"), "not APPROVED"),
        (lambda c, ds: c.update(access_confirmed=False, formation_eligible=False), "access is not confirmed"),
        (lambda c, ds: ds.update(source_snapshot_id="some_other_snapshot"), "snapshot"),
        (lambda c, ds: ds.update(source_id="some_other_source"), "source_id"),
    ],
)
def test_non_approved_sources_fail_closed(tmp_path, mutate, fragment):
    manifest = write_fixture_dataset(tmp_path / "data")

    def apply(doc, ds):
        mutate(ds["source_card"], ds)

    with pytest.raises(SourceNotEligible, match="SOURCE_NOT_FORMATION_ELIGIBLE") as info:
        load_manifest(edit_manifest(manifest, apply))
    assert fragment in str(info.value)


def test_design_partition_cannot_reach_protected_time(tmp_path):
    manifest = write_fixture_dataset(tmp_path / "data")
    with pytest.raises(BuiltinConfigError, match="before 2019"):
        load_manifest(edit_manifest(manifest, lambda d, ds: ds.update(partitions={"design": {"start": "2015-01-01", "end": "2019-06-30"}})))
    with pytest.raises(BuiltinConfigError, match="only the design partition"):
        load_manifest(edit_manifest(manifest, lambda d, ds: ds.update(
            partitions={"design": {"start": "2015-01-01", "end": "2018-12-31"}, "check": {"start": "2019-01-01", "end": "2020-12-31"}})))
    default = load_manifest(manifest).dataset("fixture_panel")
    assert default.design_window == (dt.date(2015, 1, 1), dt.date(2018, 12, 31))


def test_label_cannot_be_declared_as_feature_or_aliased(tmp_path):
    manifest = write_fixture_dataset(tmp_path / "data")
    with pytest.raises(BuiltinConfigError, match="label"):
        load_manifest(edit_manifest(manifest, lambda d, ds: ds.update(column_aliases={"ret_1_0": "ret_exc_lead1m"})))
    with pytest.raises(BuiltinConfigError):
        load_manifest(edit_manifest(manifest, lambda d, ds: ds["columns"]["ret_exc_lead1m"].update(dtype="str")))


# ------------------------------------------------------------------ strict loader


def test_declared_column_missing_from_file_and_undeclared_recipe_column(tmp_path):
    manifest = write_fixture_dataset(tmp_path / "data")
    registry, ctx = setup(tmp_path, manifest)
    # recipe needs ocf_at but the manifest does not declare it: no silent substitution
    undeclared = edit_manifest(manifest, lambda d, ds: ds["columns"].pop("ocf_at"))
    registry2 = ExecutorRegistry()
    register_executors(registry2, manifest_path=undeclared, state_dir=tmp_path / "s2", clock=clock, research_asof=ASOF)
    with pytest.raises(ExecutorError, match="does not declare"):
        registry2.run(make_request(EXECUTOR_IDS[Charter.VALUATION_RELATIVE_VALUE.value], ns="pod_valuation", recipe_id="valuation_cash_conversion_v1"))
    # declared but absent from the file
    rewrite_csv(tmp_path / "data" / "panel.csv", lambda rows: [{k: v for k, v in r.items() if k != "dolvol_126d"} for r in rows], drop_fields=("dolvol_126d",))
    with pytest.raises(ExecutorError, match="required columns missing from file"):
        registry.run(make_request())


def _row500(field, value):
    def fn(rows):
        rows[500][field] = value
    return fn


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda rows: rows.append(dict(rows[500])), "duplicate identity"),
        (_row500("ret_1_0", "inf"), "not finite"),
        (_row500("ret_12_1", "abc"), "non-numeric"),
        (_row500("eom", "2015-08-15"), "not a month-end"),
        (_row500("eom", "08/31/2015"), "unparseable"),
        (_row500("me", "-5"), "below declared minimum"),
        (_row500("permno", ""), "missing security identifier"),
        (_row500("gics", ""), "allow_missing=false"),
    ],
)
def test_data_integrity_failures_are_hard_and_replayed(tmp_path, mutate, fragment):
    manifest = write_fixture_dataset(tmp_path / "data")
    rewrite_csv(tmp_path / "data" / "panel.csv", mutate)
    registry, ctx = setup(tmp_path, manifest)
    with pytest.raises(ExecutorError, match=fragment):
        registry.run(make_request())
    slot = ctx.ledger.slot("pod_dynamics:draft1", "v1")
    assert slot["state"] == "failed_evidence"
    with pytest.raises(ExecutorError, match=fragment):  # replayed from the ledger, no new budget
        registry.run(make_request())
    assert ctx.ledger.variants_used("pod_dynamics:draft1") == 1
    assert ctx.ledger.trial_counts("pod_dynamics:draft1")["scored_variant"] == 1


def test_row_level_feature_availability_is_enforced(tmp_path):
    manifest = write_fixture_dataset(tmp_path / "data")

    def add_avail(rows):
        for row in rows:
            row["avail"] = row["eom"]
        late = dt.date.fromisoformat(rows[500]["eom"]) + dt.timedelta(days=1)
        rows[500]["avail"] = late.isoformat()

    rewrite_csv(tmp_path / "data" / "panel.csv", add_avail, extra_fields=("avail",))

    def declare(policy):
        def fn(doc, ds):
            ds["columns"]["avail"] = {"role": "context", "dtype": "date", "allow_missing": False}
            ds["availability"] = {
                "mode": "row_column", "column": "avail", "on_violation": policy,
                "attestation": "fixture row-level availability column for tests",
            }
        return fn

    registry, _ = setup(tmp_path, edit_manifest(manifest, declare("error"), "m_error.json"))
    with pytest.raises(ExecutorError, match="not available by the formation date"):
        registry.run(make_request())
    registry2, ctx2 = setup(tmp_path / "excl", edit_manifest(manifest, declare("exclude"), "m_exclude.json"))
    result, report = run_metrics(registry2, ctx2)
    assert report["data"]["rows_excluded_for_availability"] == 1 and result.metrics


def test_rows_after_the_design_window_are_never_parsed(tmp_path):
    clean_manifest = write_fixture_dataset(tmp_path / "clean")
    dirty_manifest = write_fixture_dataset(tmp_path / "dirty")

    def trash(rows):
        for row in rows:
            if row["eom"] >= "2019-01-01":
                for key in row:
                    if key not in ("permno", "eom"):
                        row[key] = "garbage"

    rewrite_csv(tmp_path / "dirty" / "panel.csv", trash)
    reg_a, ctx_a = setup(tmp_path / "a", clean_manifest)
    reg_b, ctx_b = setup(tmp_path / "b", dirty_manifest)
    res_a, rep_a = run_metrics(reg_a, ctx_a)
    res_b, rep_b = run_metrics(reg_b, ctx_b)
    assert res_a.metrics == res_b.metrics
    assert res_a.spec_hash != res_b.spec_hash  # the data file hash is part of the registered spec
    assert rep_b["data"]["rows_after_window_discarded_unparsed"] == 60 * 6
    assert rep_b["data"]["rows_before_window_discarded"] == 60
    assert rep_b["data"]["window"] == ["2015-01-01", "2018-12-31"]


def test_parquet_loader_matches_csv_and_discards_protected_rows(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = generate_rows()
    csv_manifest = write_fixture_dataset(tmp_path / "csv")
    reg_c, ctx_c = setup(tmp_path / "c", csv_manifest)
    expected = reg_c.run(make_request()).metrics
    for kind in ("date32", "timestamp", "string"):
        directory = tmp_path / kind
        directory.mkdir()
        cols = {}
        for name in FIELDS:
            values = [r[name] for r in rows]
            if name == "permno":
                cols[name] = pa.array([int(v) for v in values], type=pa.int64())
            elif name == "gics":
                cols[name] = pa.array(values, type=pa.string())
            elif name == "eom":
                days = [dt.date.fromisoformat(v) for v in values]
                if kind == "date32":
                    cols[name] = pa.array(days, type=pa.date32())
                elif kind == "timestamp":
                    cols[name] = pa.array([dt.datetime(d.year, d.month, d.day) for d in days], type=pa.timestamp("ns"))
                else:
                    cols[name] = pa.array(values, type=pa.string())
            else:
                cols[name] = pa.array([float(v) if v != "" else None for v in values], type=pa.float64())
        pq.write_table(pa.table(cols), str(directory / "panel.parquet"))
        manifest = edit_manifest(csv_manifest, lambda d, ds: ds.update(path=str(directory / "panel.parquet"), format="parquet") or d.update(allowed_roots=[str(directory)]), f"m_{kind}.json")
        reg, ctx = setup(tmp_path / f"run_{kind}", manifest)
        result, report = run_metrics(reg, ctx)
        for key, value in expected.items():
            assert result.metrics[key] == pytest.approx(value, rel=1e-9, abs=1e-12), (kind, key)
        assert report["data"]["format"] == "parquet"
        assert report["data"]["rows_in_window"] == 60 * 48


# ------------------------------------------------------------------ temporal integrity (mutation tests)


def _monthly(report):
    return {rec["month"]: rec for rec in report["monthly"]}


def test_mutating_future_labels_cannot_change_earlier_months(tmp_path):
    base_manifest = write_fixture_dataset(tmp_path / "base")
    mut_manifest = write_fixture_dataset(tmp_path / "mut")
    rewrite_csv(tmp_path / "mut" / "panel.csv", lambda rows: [{**r, "ret_exc_lead1m": "0.5"} if r["eom"].startswith("2017-06") else r for r in rows])
    ra, ca = setup(tmp_path / "a", base_manifest)
    rb, cb = setup(tmp_path / "b", mut_manifest)
    _, rep_a = run_metrics(ra, ca)
    _, rep_b = run_metrics(rb, cb)
    m0 = month_index(dt.date(2017, 6, 30))
    a, b = _monthly(rep_a), _monthly(rep_b)
    before = [m for m in a if m < m0]
    assert before and all(a[m] == b[m] for m in before)
    assert any(a[m] != b.get(m) for m in a if m >= m0)  # a now-degenerate evaluation month may disappear


def test_mutating_future_features_cannot_change_earlier_months(tmp_path):
    base_manifest = write_fixture_dataset(tmp_path / "base")
    mut_manifest = write_fixture_dataset(tmp_path / "mut")
    rewrite_csv(tmp_path / "mut" / "panel.csv", lambda rows: [{**r, "niq_su": "99"} if r["eom"].startswith("2017-06") else r for r in rows])
    ra, ca = setup(tmp_path / "a", base_manifest)
    rb, cb = setup(tmp_path / "b", mut_manifest)
    _, rep_a = run_metrics(ra, ca, executor_id=FUND, ns="pod_fundamentals")
    _, rep_b = run_metrics(rb, cb, executor_id=FUND, ns="pod_fundamentals")
    m0 = month_index(dt.date(2017, 6, 30))
    a, b = _monthly(rep_a), _monthly(rep_b)
    horizon = 3  # niq_su is also the lead-3m target of the month three rows earlier
    before = [m for m in a if m < m0 - horizon]
    assert before and all(a[m] == b[m] for m in before)
    assert any(a[m] != b.get(m) for m in a if m >= m0 - horizon)


def test_training_uses_only_matured_labels_and_purges_the_window_edge(tmp_path):
    registry, ctx = setup(tmp_path)
    result, report = run_metrics(registry, ctx)
    months = sorted(_monthly(report))
    assert months[0] >= month_index(dt.date(2015, 1, 31)) + 12  # warm-up: 12 matured training months
    assert months[-1] <= month_index(dt.date(2018, 11, 30))  # Dec-2018 label resolves in the protected 2019 window
    assert result.metrics["coverage_purged_label_rows"] >= 60
    assert result.metrics["coverage_warmup_months"] >= 12


# ------------------------------------------------------------------ evidence: power, null, thin data


def test_planted_signal_is_detected_and_null_is_not(tmp_path):
    strong = write_fixture_dataset(tmp_path / "strong", ret_signal=0.03)
    null = write_fixture_dataset(tmp_path / "null", ret_signal=0.0)
    rs, cs = setup(tmp_path / "s", strong)
    rn, cn = setup(tmp_path / "n", null)
    m_strong = rs.run(make_request()).metrics
    m_null = rn.run(make_request()).metrics
    assert m_strong["belief_ic_improvement"] > 0.05 and m_strong["belief_ic_improvement_t"] > 3.0
    assert m_strong["bridge_ic_candidate"] > m_strong["bridge_ic_baseline"] > 0.0
    assert m_strong["bridge_net_spread_candidate"] > 0.0
    assert abs(m_null["belief_ic_improvement_t"]) < 3.5
    assert abs(m_null["bridge_ic_candidate"]) < 0.1


def test_thin_data_is_a_failure_note_not_a_result(tmp_path):
    manifest = write_fixture_dataset(tmp_path / "data", n_securities=10)
    registry, ctx = setup(tmp_path, manifest)
    with pytest.raises(ExecutorError, match="INSUFFICIENT_HISTORY"):
        registry.run(make_request())
    assert ctx.ledger.slot("pod_dynamics:draft1", "v1")["state"] == "failed_evidence"


# ------------------------------------------------------------------ ledger: idempotency, budget, accounting


def test_variant_budget_idempotence_and_trial_counts(tmp_path):
    registry, ctx = setup(tmp_path, variant_budget=2)
    reg_id = "pod_dynamics:draft1"
    first = registry.run(make_request(name="a"))
    again = registry.run(make_request(name="a"))
    assert again == first and ctx.ledger.variants_used(reg_id) == 1
    registry.run(make_request(name="b", method_id="rank_blend"))
    with pytest.raises(ExecutorError, match="budget"):
        registry.run(make_request(name="c", feature_set="extended"))
    with pytest.raises(ExecutorError, match="different specification"):
        registry.run(make_request(name="a", method_id="rank_blend"))
    counts = ctx.ledger.trial_counts(reg_id)
    assert counts["scored_variant"] == 2 and counts["hyperparameter_config"] == 1
    assert counts["fold_evaluation"] > 0 and counts["design_partition_access"] == 2
    # a fresh process with the same state dir resumes the same slots
    registry2, ctx2 = setup(tmp_path, ctx.manifest.path, variant_budget=2)
    assert registry2.run(make_request(name="a")) == first
    assert ctx2.ledger.variants_used(reg_id) == 2


def test_interrupted_slot_resumes_and_is_counted(tmp_path):
    registry, ctx = setup(tmp_path)
    _, _, _, _, _, spec_hash = ctx.prepare(DYN, {"dataset_id": "fixture_panel"})
    ctx.ledger.admit("pod_dynamics:draft1", "v1", spec_hash, budget=8)  # simulate a crash after admission
    result = registry.run(make_request(dataset_id="fixture_panel"))
    assert result.spec_hash == spec_hash
    counts = ctx.ledger.trial_counts("pod_dynamics:draft1")
    assert counts["failed_infrastructure_attempt"] == 1 and counts["scored_variant"] == 1


def test_ledger_is_append_only(tmp_path):
    ledger = BuiltinLedger(tmp_path / "ledger", clock=clock)
    ledger.admit("r", "v", "a" * 64, budget=1)
    ledger.record_evidence_failure("r", "v", "a" * 64, {"type": "EvidenceInsufficient", "message": "x"})
    with pytest.raises(sqlite3.DatabaseError):
        ledger._conn.execute("UPDATE trials SET quantity = 99")
    with pytest.raises(sqlite3.DatabaseError):
        ledger._conn.execute("UPDATE variant_slots SET state = 'started'")
    with pytest.raises(sqlite3.DatabaseError):
        ledger._conn.execute("DELETE FROM variant_slots")
    naive = BuiltinLedger(None, clock=lambda: dt.datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="timezone-aware"):
        naive.admit("r", "v", "b" * 64, budget=1)


# ------------------------------------------------------------------ integration with the pod runner / discovery


def test_pod_runner_stores_builtin_result_as_evaluated_not_admitted(tmp_path):
    registry, ctx = setup(tmp_path)
    spec = pod_spec(Charter.MARKET_DYNAMICS)
    memory = PodMemory(tmp_path / "mem", spec.namespace)
    provider = DeterministicMockProvider(lambda request: "{}")
    runner = PodResearchRunner(spec, provider, DurableExtractionCache(tmp_path / "cache.sqlite"), memory)
    variant = ExperimentVariant(name="v1", params={"dataset_id": "fixture_panel", "method_id": "ridge", "ridge_lambda": 0.05})
    proposal = make_proposal(DYN, [variant])
    draft = memory.append("draft_proposal", {
        "status": "draft", "evaluated": False, "admission": "not_admitted", "title": proposal.title,
        "proposal": proposal.model_dump(mode="json"),
    })
    entries = runner.run_experiments(draft.entry_id, registry)
    assert len(entries) == 1 and entries[0].kind == "evaluated_result" and entries[0].evaluated
    content = entries[0].content
    assert content["admission"] == "not_admitted" and content["data_partition"] == "design"
    assert math.isfinite(content["metrics"]["belief_ic_candidate"])
    assert content["registration_id"] == f"{spec.namespace}:{draft.entry_id}"


def test_catalog_summary_exposes_ids_not_paths(tmp_path):
    manifest = write_fixture_dataset(tmp_path / "data")
    summary = catalog_summary(manifest)
    text = json.dumps(summary)
    assert str(tmp_path) not in text and "panel.csv" not in text
    assert set(summary["executors"]) == set(EXECUTOR_IDS.values())
    assert "fixture_panel" in summary["datasets"]
    assert "path" not in summary["allowed_variant_params"] and "dataset_id" in summary["allowed_variant_params"]
    assert catalog_summary(None)["datasets"] == {} if "POD_DATASET_MANIFEST" not in os.environ else True


# ------------------------------------------------------------------ numerics


def test_stats_primitives():
    assert stats.centered_rank([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]
    ranks = stats.centered_rank([3.0, 1.0, 2.0])
    assert ranks == pytest.approx([1 / 3, -1 / 3, 0.0])
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert stats.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None  # flat forecasts carry no skill
    assert stats.pearson([1, 2], [1, 2]) is None
    solution = stats.solve_linear([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    assert solution == pytest.approx([1.0, 3.0])
    assert stats.solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]) is None
    series = [float(i // 3) for i in range(30)]
    n = len(series)
    mu = sum(series) / n
    assert stats.hac_se(series, 0) == pytest.approx(math.sqrt(sum((v - mu) ** 2 for v in series)) / n)
    assert stats.hac_se(series, 3) > stats.hac_se(series, 0)  # positive autocorrelation widens the interval


def test_fixture_generator_is_deterministic_and_has_planted_dependence():
    a, b = generate_rows(seed=3, n_securities=8), generate_rows(seed=3, n_securities=8)
    assert a == b
    assert a != generate_rows(seed=4, n_securities=8)
    null = generate_rows(seed=3, n_securities=8, ret_signal=0.0)
    assert [r["ret_exc_lead1m"] for r in a] != [r["ret_exc_lead1m"] for r in null]
