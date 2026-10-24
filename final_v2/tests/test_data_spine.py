"""Part 2 acceptance: temporal correctness and identity tests (spec 26.1-26.2)."""

from __future__ import annotations

import datetime as dt

import pytest

from pod_platform.contracts import ErrorCode, PodPlatformError
from pod_platform.contracts.sources import AssessmentStatus, CardStatus, ScreenItem
from pod_platform.data import (
    AvailabilityRule,
    BitemporalStore,
    CatalogueManifest,
    DataBroker,
    IdentifierLink,
    IdentifierType,
    MappingProposal,
    Observation,
    ResolutionState,
    RuleKind,
    SecurityMaster,
    TrustedContext,
    UsageMode,
    compute_available_at,
)

from .test_contracts import _screen, _source_card

UTC = dt.timezone.utc
T0 = dt.datetime(2020, 1, 1, tzinfo=UTC)


def _when(**kw) -> dt.datetime:
    return T0 + dt.timedelta(**kw)


def _rule(**over) -> AvailabilityRule:
    base = dict(
        owner="data",
        created_at=T0,
        rule_id="rule_1",
        source_id="source_example",
        metric="spend_growth",
        rule_kind=RuleKind.SOURCE_PUBLICATION,
        lag_days=0,
        evidence_id="ev_1",
        revision_completeness="first_report_only",
        usage_mode=UsageMode.SOURCE_PUBLICATION_HISTORICAL,
    )
    base.update(over)
    return AvailabilityRule(**base)


def _obs(observation_id: str, **over) -> Observation:
    base = dict(
        owner="data",
        created_at=T0,
        observation_id=observation_id,
        source_id="source_example",
        metric="spend_growth",
        entity_id="sec_1",
        version=1,
        ingested_at=_when(days=3),
        version_known_at=_when(days=2),
        available_at=_when(days=2),
        value=0.05,
        unit="decimal_growth_rate",
    )
    base.update(over)
    return Observation(**base)


def _link(link_id: str, **over) -> IdentifierLink:
    base = dict(
        owner="data",
        created_at=T0,
        link_id=link_id,
        security_id="sec_1",
        identifier_type=IdentifierType.TICKER,
        identifier_value="AAA",
        valid_from=_when(days=0),
        valid_to=_when(days=3650),
        first_known_at=_when(days=0),
        resolution_state=ResolutionState.EXACT_DATED,
        evidence_id="ev_1",
    )
    base.update(over)
    return IdentifierLink(**base)


def _approved_card():
    return _source_card(
        status=CardStatus.APPROVED,
        access_confirmed=True,
        formation_eligible=True,
        screen=_screen(),
        unresolved_questions=[],
    )


def _broker():
    catalogue = CatalogueManifest()
    catalogue.import_card(_approved_card())
    store = BitemporalStore()
    master = SecurityMaster()
    master.add_link(_link("link_1", identifier_type=IdentifierType.ISSUER, identifier_value="sec_1"))
    rules = {"source_example:spend_growth": _rule()}
    broker = DataBroker(catalogue=catalogue, store=store, security_master=master, rules=rules)
    return broker, store, catalogue, master


def _ctx(**over) -> TrustedContext:
    base = dict(
        caller_role="strategy",
        namespace="strategy_1",
        research_asof=_when(days=3650),
        allowed_partitions=("design",),
        data_policy_version="policy_v1",
        request_id="req_1",
    )
    base.update(over)
    return TrustedContext(**base)


# ---------------------------------------------------------------- 26.1 temporal


def test_future_dated_observation_rejected():
    with pytest.raises(Exception):
        _obs("o1", available_at=_when(days=2), version_known_at=_when(days=1))


def test_revision_not_visible_before_its_publication():
    store = BitemporalStore()
    store.insert(_obs("o1", available_at=_when(days=2), version_known_at=_when(days=2)))
    store.insert(
        _obs(
            "o2",
            version=2,
            revision_of="o1",
            available_at=_when(days=30),
            version_known_at=_when(days=30),
            ingested_at=_when(days=31),
            value=-0.02,
        )
    )
    early = store.asof_versions(_when(days=5))
    assert [row.observation_id for row in early] == ["o1"]
    late = store.asof_versions(_when(days=40))
    assert [row.observation_id for row in late] == ["o2"]


def test_period_end_value_not_available_before_publication():
    store = BitemporalStore()
    store.insert(
        _obs(
            "o1",
            period_start=_when(days=0),
            period_end=_when(days=90),
            available_at=_when(days=100),
            version_known_at=_when(days=100),
            ingested_at=_when(days=101),
        )
    )
    assert store.asof_versions(_when(days=95)) == []
    assert len(store.asof_versions(_when(days=100))) == 1


def test_asof_selects_latest_eligible_not_latest_row():
    store = BitemporalStore()
    store.insert(_obs("o1", available_at=_when(days=2), version_known_at=_when(days=2)))
    store.insert(
        _obs(
            "o2",
            version=2,
            revision_of="o1",
            available_at=_when(days=50),
            version_known_at=_when(days=50),
            ingested_at=_when(days=51),
        )
    )
    assert [row.observation_id for row in store.asof_versions(_when(days=20))] == ["o1"]


def test_late_start_source_contributes_no_earlier_observations():
    _, store, catalogue, master = _broker()
    rules = {
        "source_example:spend_growth": _rule(
            earliest_start=_when(days=1000),
            usage_mode=UsageMode.SOURCE_PUBLICATION_HISTORICAL,
        )
    }
    broker = DataBroker(catalogue=catalogue, store=store, security_master=master, rules=rules)
    rows, report = broker.query_asof(
        _ctx(),
        source_id="source_example",
        metric="spend_growth",
        entity_id="sec_1",
        information_cutoff=_when(days=500),
        identity_at=_when(days=500),
        partition="design",
    )
    assert rows == []
    assert report.assertions["late_start_source_absent_before_start"] == "pass"


def test_broker_rejects_cutoff_beyond_research_clock():
    broker, *_ = _broker()
    with pytest.raises(PodPlatformError) as exc:
        broker.query_asof(
            _ctx(research_asof=_when(days=10)),
            source_id="source_example",
            metric="spend_growth",
            entity_id="sec_1",
            information_cutoff=_when(days=20),
            identity_at=_when(days=20),
            partition="design",
        )
    assert exc.value.response.error.code == ErrorCode.ASOF_VIOLATION


def test_broker_rejects_unapproved_source():
    catalogue = CatalogueManifest()
    catalogue.import_card(_source_card())
    store = BitemporalStore()
    master = SecurityMaster()
    master.add_link(_link("link_1", identifier_type=IdentifierType.ISSUER, identifier_value="sec_1"))
    broker = DataBroker(
        catalogue=catalogue,
        store=store,
        security_master=master,
        rules={"source_example:spend_growth": _rule()},
    )
    with pytest.raises(PodPlatformError) as exc:
        broker.query_asof(
            _ctx(),
            source_id="source_example",
            metric="spend_growth",
            entity_id="sec_1",
            information_cutoff=_when(days=5),
            identity_at=_when(days=5),
            partition="design",
        )
    assert exc.value.response.error.code == ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE


def test_broker_rejects_descriptive_only_metric():
    _, store, catalogue, master = _broker()
    rules = {"source_example:spend_growth": _rule(usage_mode=UsageMode.DESCRIPTIVE_ONLY)}
    broker = DataBroker(catalogue=catalogue, store=store, security_master=master, rules=rules)
    with pytest.raises(PodPlatformError) as exc:
        broker.query_asof(
            _ctx(),
            source_id="source_example",
            metric="spend_growth",
            entity_id="sec_1",
            information_cutoff=_when(days=5),
            identity_at=_when(days=5),
            partition="design",
        )
    assert exc.value.response.error.code == ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE


def test_broker_rejects_disallowed_partition():
    broker, *_ = _broker()
    with pytest.raises(PodPlatformError) as exc:
        broker.query_asof(
            _ctx(),
            source_id="source_example",
            metric="spend_growth",
            entity_id="sec_1",
            information_cutoff=_when(days=5),
            identity_at=_when(days=5),
            partition="check",
        )
    assert exc.value.response.error.code == ErrorCode.PARTITION_ACCESS_DENIED


def test_broker_eligibility_report_complete():
    broker, store, *_ = _broker()
    store.insert(_obs("o1", available_at=_when(days=2), version_known_at=_when(days=2), ingested_at=_when(days=2)))
    rows, report = broker.query_asof(
        _ctx(),
        source_id="source_example",
        metric="spend_growth",
        entity_id="sec_1",
        information_cutoff=_when(days=5),
        identity_at=_when(days=5),
        partition="design",
    )
    assert [row.observation_id for row in rows] == ["o1"]
    assert report.all_passed
    assert set(report.assertions) >= {
        "feature_availability_precedes_cutoff",
        "identity_links_valid_at_event_time",
        "source_approved_for_partition",
        "aggregation_windows_end_before_availability",
        "training_labels_resolved_before_cutoff",
        "late_start_source_absent_before_start",
    }


# ---------------------------------------------------------------- 26.2 identity


def test_reused_ticker_resolves_per_period():
    master = SecurityMaster()
    master.add_link(
        _link(
            "l1",
            security_id="sec_old",
            identifier_value="ZZZ",
            valid_from=_when(days=0),
            valid_to=_when(days=1000),
            first_known_at=_when(days=0),
        )
    )
    master.add_link(
        _link(
            "l2",
            security_id="sec_new",
            identifier_value="ZZZ",
            valid_from=_when(days=1000),
            valid_to=_when(days=3650),
            first_known_at=_when(days=1000),
        )
    )
    assert master.resolve(IdentifierType.TICKER, "ZZZ", _when(days=500)) == "sec_old"
    assert master.resolve(IdentifierType.TICKER, "ZZZ", _when(days=2000)) == "sec_new"


def test_ambiguous_mapping_stays_out_of_formation():
    master = SecurityMaster()
    master.add_link(
        _link("l1", security_id="sec_a", identifier_value="MMM", resolution_state=ResolutionState.AMBIGUOUS)
    )
    with pytest.raises(PodPlatformError) as exc:
        master.resolve(IdentifierType.TICKER, "MMM", _when(days=10))
    assert exc.value.response.error.code == ErrorCode.IDENTITY_AMBIGUOUS


def test_one_to_many_share_class_mapping_rejected():
    master = SecurityMaster()
    master.add_link(_link("l1", security_id="sec_a", identifier_value="BRK"))
    master.add_link(_link("l2", security_id="sec_b", identifier_value="BRK"))
    with pytest.raises(PodPlatformError) as exc:
        master.resolve(IdentifierType.TICKER, "BRK", _when(days=10))
    assert exc.value.response.error.code == ErrorCode.IDENTITY_AMBIGUOUS


def test_delisted_company_absent_from_current_map():
    master = SecurityMaster()
    master.add_link(
        _link(
            "l1",
            security_id="sec_dead",
            identifier_value="OLD",
            valid_from=_when(days=0),
            valid_to=_when(days=500),
            first_known_at=_when(days=0),
        )
    )
    assert master.resolve(IdentifierType.TICKER, "OLD", _when(days=100)) == "sec_dead"
    with pytest.raises(PodPlatformError):
        master.resolve(IdentifierType.TICKER, "OLD", _when(days=2000))


def test_link_unknown_at_query_time_not_used():
    master = SecurityMaster()
    master.add_link(
        _link(
            "l1",
            security_id="sec_1",
            identifier_value="AAA",
            valid_from=_when(days=0),
            valid_to=_when(days=3650),
            first_known_at=_when(days=100),
        )
    )
    with pytest.raises(PodPlatformError):
        master.resolve(IdentifierType.TICKER, "AAA", _when(days=50))
    assert master.resolve(IdentifierType.TICKER, "AAA", _when(days=200)) == "sec_1"


def test_mapping_proposal_admissibility():
    proposal = MappingProposal(
        source_identifier_type=IdentifierType.BRAND,
        source_identifier_value="brand_x",
        candidate_security_ids=["sec_1"],
        valid_from=T0,
        valid_to=_when(days=365),
        evidence_id="ev_1",
        confidence=0.8,
        review_status=ResolutionState.EXACT_DATED,
    )
    assert proposal.admissible_for_formation
    ambiguous = proposal.model_copy(
        update={"candidate_security_ids": ["sec_1", "sec_2"], "review_status": ResolutionState.AMBIGUOUS}
    )
    assert not ambiguous.admissible_for_formation


# --------------------------------------------------------------- availability


def test_compute_available_at_variants():
    source_rule = _rule()
    assert compute_available_at(source_rule, source_published_at=_when(days=7)) == _when(days=7)
    lag_rule = _rule(rule_kind=RuleKind.PERIOD_END_PLUS_LAG, lag_days=10)
    assert compute_available_at(lag_rule, period_end=_when(days=20)) == _when(days=30)
    delivery_rule = _rule(rule_kind=RuleKind.PLATFORM_DELIVERY)
    assert compute_available_at(delivery_rule, platform_delivered_at=_when(days=3)) == _when(days=3)
    with pytest.raises(Exception):
        compute_available_at(source_rule)


def test_period_end_lag_rule_requires_positive_lag():
    with pytest.raises(Exception):
        _rule(rule_kind=RuleKind.PERIOD_END_PLUS_LAG, lag_days=0)


# ------------------------------------------------------------------ catalogue


def test_catalogue_import_and_manifest(tmp_path):
    catalogue = CatalogueManifest()
    catalogue.import_card(_approved_card())
    catalogue.import_card(_source_card(source_id="source_b", topic="Demand"))
    assert len(catalogue) == 2
    manifest = catalogue.manifest("catalogue_20261024")
    assert manifest["entry_count"] == 2
    assert manifest["topic_counts"] == {"Contracts": 1, "Demand": 1}
    assert manifest["formation_eligible"] == ["source_example"]
    assert set(manifest["content_hashes"]) == {"source_example", "source_b"}


def test_catalogue_directory_import(tmp_path):
    import json

    card = _approved_card()
    (tmp_path / "card.json").write_text(json.dumps(card.model_dump(mode="json")), encoding="utf-8")
    catalogue = CatalogueManifest()
    assert catalogue.import_directory(tmp_path) == 1
    assert catalogue.card("source_example").dataset_id == "CA0093"


def test_catalogue_duplicate_source_rejected():
    catalogue = CatalogueManifest()
    catalogue.import_card(_approved_card())
    with pytest.raises(ValueError):
        catalogue.import_card(_approved_card())
