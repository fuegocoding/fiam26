"""Bounded alternative-source discovery: catalogue search, injected connectors, budgets, cache, proposals, review."""

import datetime as dt
import threading

import pytest

from pod_platform.application.namespaces import RoleContext
from pod_platform.contracts.errors import ErrorCode, PodPlatformError
from pod_platform.contracts.sources import AssessmentStatus, CardStatus, Screen, ScreenItem, SourceCard
from pod_platform.data.catalogue import CatalogueManifest
from pod_platform.data.discovery import CatalogueMeta, DiscoveryService

UTC = dt.timezone.utc
ASOF = dt.datetime(2024, 6, 1, tzinfo=UTC)


def ctx(role="strategy", ns="pod_a"):
    return RoleContext(caller_role=role, namespace=ns if role == "strategy" else role, run_id="run1", research_asof=ASOF)


def _card(source_id, topic, eligible, dataset=None):
    item = ScreenItem(status=AssessmentStatus.ESTABLISHED if eligible else AssessmentStatus.UNRESOLVED,
                      evidence_id="ev_x" if eligible else None)
    return SourceCard(owner="data", created_at=ASOF, source_id=source_id, dataset_id=dataset or f"ds_{source_id}", topic=topic,
                      snapshot_id="snap1", status=CardStatus.APPROVED if eligible else CardStatus.DOCUMENTED, access_confirmed=eligible,
                      formation_eligible=eligible, entity_grain=["security"],
                      screen=Screen(history=item, metric_lag=item, vintage=item, coverage=item),
                      unresolved_questions=[] if eligible else ["history unknown"])


def _candidates():
    base = dict(title="t", concepts=["web traffic"], fields=["visits"], entity_grain=["security"])
    return [
        dict(provider_ref="good_feed", availability="available", history_start=dt.date(2015, 1, 1), dataset_hint="vendor_ds_1", **base),
        dict(provider_ref="no_history", availability="available", **base),
        dict(provider_ref="future_feed", availability="available", history_start=dt.date(2030, 1, 1), **base),
        dict(provider_ref="no_adapter", availability="unsupported", **base),
        dict(provider_ref="paid_feed", availability="entitlement_blocked", **base),
        dict(provider_ref="Bad Ref!", availability="available", **base),  # invalid candidate is dropped, not trusted
        dict(provider_ref="in_catalogue", availability="available", history_start=dt.date(2015, 1, 1), dataset_hint="ds_CA0010", **base),
    ]


class FakeConnector:
    provider_id = "vendor_a"

    def __init__(self):
        self.discover_calls, self.sample_calls, self.fail, self.rows = [], [], False, None

    def discover(self, query, *, as_of, limit):
        self.discover_calls.append((query, as_of, limit))
        if self.fail:
            raise RuntimeError("https://vendor.example/?key=SECRET")
        return _candidates()

    def sample(self, provider_ref, *, as_of, max_rows):
        self.sample_calls.append((provider_ref, max_rows))
        return {"columns": ["day", "visits"], "rows": self.rows or [{"day": f"2024-01-{i % 28 + 1:02d}", "visits": i} for i in range(50)]}


@pytest.fixture()
def env(tmp_path):
    catalogue = CatalogueManifest()
    catalogue.import_card(_card("CA0010", "consumer_traffic", True))
    catalogue.import_card(_card("CA0020", "credit_markets", False))
    meta = [CatalogueMeta(source_id="CA0010", title="Store visits panel", concepts=["consumer demand", "foot traffic"], fields=["visits", "dwell_time"]),
            CatalogueMeta(source_id="CA0020", title="Loan spreads", concepts=["credit risk"], fields=["spread"])]
    connector = FakeConnector()

    def make():
        svc = DiscoveryService(tmp_path / "disc.sqlite", catalogue=catalogue, metadata=meta)
        svc.register_connector(connector)
        return svc

    svc = make()
    svc.set_provider_budget("pod_a", "vendor_a", 2)
    svc.set_provider_budget("pod_b", "vendor_a", 2)
    return svc, connector, catalogue, make


def _code(exc):
    return exc.value.response.error.code


def test_catalogue_search_matches_concepts_and_keeps_eligibility(env):
    svc, _, _, _ = env
    hits = svc.search_catalogue(ctx(), "consumer demand visits")
    assert [h["source_id"] for h in hits] == ["CA0010"]
    assert hits[0]["formation_eligible"] is True and hits[0]["availability"] == "available"
    credit = svc.search_catalogue(ctx(), "credit risk spreads")[0]
    assert credit["source_id"] == "CA0020" and credit["formation_eligible"] is False and credit["availability"] == "entitlement_unconfirmed"
    assert svc.search_catalogue(ctx(), "foot traffic")[0]["source_id"] == "CA0010"
    assert svc.search_catalogue(ctx(), "quantum gravity") == []
    assert svc.describe_catalogue(ctx(), "CA0020")["unresolved_questions"] == ["history unknown"]
    with pytest.raises(PodPlatformError):
        svc.describe_catalogue(ctx(), "NOPE")


@pytest.mark.parametrize("query", ["https://x.example/data", "see /etc/passwd", "api_key=abc", "x", "my secret feed"])
def test_query_rejects_urls_paths_and_credentials(env, query):
    svc, connector, _, _ = env
    with pytest.raises(PodPlatformError):
        svc.discover(ctx(), "pod_a", "vendor_a", query)
    assert connector.discover_calls == []


def test_discover_runs_callback_with_cache_budget_and_explicit_availability(env):
    svc, connector, _, _ = env
    first = svc.discover(ctx(), "pod_a", "vendor_a", "web  traffic")
    assert len(connector.discover_calls) == 1 and first["cached"] is False and first["queries_remaining"] == 1
    assert connector.discover_calls[0][:2] == ("web traffic", ASOF)
    avail = {c["provider_ref"]: c["availability"] for c in first["candidates"]}
    assert avail["good_feed"] == "available" and avail["no_history"] == "unresolved_history" and avail["future_feed"] == "unavailable_as_of"
    assert avail["no_adapter"] == "unsupported" and avail["paid_feed"] == "entitlement_blocked" and "Bad Ref!" not in avail
    assert first["dropped_candidates"] >= 1
    assert all(c["historical_coverage_established"] is False and c["formation_eligible"] is False for c in first["candidates"])
    again = svc.discover(ctx(), "pod_a", "vendor_a", "WEB TRAFFIC")  # same normalised query: cache, no spend, no callback
    assert again["cached"] is True and len(connector.discover_calls) == 1 and again["queries_remaining"] == 1
    svc.discover(ctx(), "pod_a", "vendor_a", "retail sales")
    assert svc.queries_remaining("pod_a", "vendor_a") == 0
    with pytest.raises(PodPlatformError) as exc:
        svc.discover(ctx(), "pod_a", "vendor_a", "credit spreads")
    assert _code(exc) == ErrorCode.PROVIDER_BUDGET_EXCEEDED and len(connector.discover_calls) == 2
    assert [r["status"] for r in svc.query_log("pod_a")].count("refused") == 1
    assert svc.queries_remaining("pod_b", "vendor_a") == 2  # budgets and caches are per namespace
    assert svc.discover(ctx(ns="pod_b"), "pod_b", "vendor_a", "web traffic")["cached"] is False


def test_unknown_provider_and_foreign_namespace_are_explicit_denials(env):
    svc, connector, _, _ = env
    out = svc.discover(ctx(), "pod_a", "vendor_zzz", "web traffic")
    assert out["status"] == "unsupported" and out["queries_spent"] == 0
    with pytest.raises(PodPlatformError) as exc:
        svc.discover(ctx(), "pod_b", "vendor_a", "web traffic")
    assert _code(exc) == ErrorCode.PARTITION_ACCESS_DENIED and connector.discover_calls == []
    with pytest.raises(PodPlatformError):
        svc.discover(ctx("integrity"), "integrity", "vendor_a", "web traffic")  # integrity does not spend discovery budget


def test_connector_failure_is_explicit_spent_and_leaks_nothing(env):
    svc, connector, _, _ = env
    connector.fail = True
    out = svc.discover(ctx(), "pod_a", "vendor_a", "web traffic")
    assert out["status"] == "connector_error" and out["error"] == "RuntimeError" and "SECRET" not in str(out)
    assert svc.queries_remaining("pod_a", "vendor_a") == 1
    assert "SECRET" not in str(svc.query_log("pod_a"))
    connector.fail = False
    assert svc.discover(ctx(), "pod_a", "vendor_a", "web traffic")["status"] == "ok"  # failures are not cached


def test_sample_is_bounded_cached_and_never_establishes_history(env):
    svc, connector, _, _ = env
    fp = svc.discover(ctx(), "pod_a", "vendor_a", "web traffic")["fingerprint"]
    out = svc.sample(ctx(), "pod_a", fp, "good_feed", max_rows=3)
    assert len(out["rows"]) == 3 and connector.sample_calls == [("good_feed", 3)]
    assert out["coverage"] == "sample_only" and out["historical_coverage_established"] is False and out["evidence_id"].startswith("sample:")
    assert svc.sample(ctx(), "pod_a", fp, "good_feed", max_rows=3)["cached"] is True and len(connector.sample_calls) == 1
    assert svc.queries_remaining("pod_a", "vendor_a") == 0  # discover + one sample
    for ref, status in (("no_adapter", "unsupported"), ("paid_feed", "entitlement_blocked"), ("no_history", "unresolved_history"), ("future_feed", "unavailable_as_of")):
        assert svc.sample(ctx(), "pod_a", fp, ref)["status"] == status
    with pytest.raises(PodPlatformError):
        svc.sample(ctx(), "pod_a", fp, "never_discovered")
    assert len(connector.sample_calls) == 1


def test_sample_with_future_dated_rows_is_rejected_and_not_cached(env):
    svc, connector, _, _ = env
    fp = svc.discover(ctx(), "pod_a", "vendor_a", "web traffic")["fingerprint"]
    connector.rows = [{"day": "2024-01-02", "visits": 1}, {"day": "2025-01-02", "visits": 2}]
    out = svc.sample(ctx(), "pod_a", fp, "good_feed")
    assert out["status"] == "rejected_asof" and "rows" not in out
    assert svc.queries_remaining("pod_a", "vendor_a") == 0  # the attempt is spent and logged
    with pytest.raises(PodPlatformError):  # budget gone: a retry cannot be served from cache because nothing was cached
        svc.sample(ctx(), "pod_a", fp, "good_feed")


def test_cache_and_budget_survive_restart(env):
    svc, connector, _, make = env
    svc.discover(ctx(), "pod_a", "vendor_a", "web traffic")
    restarted = make()
    again = restarted.discover(ctx(), "pod_a", "vendor_a", "web traffic")
    assert again["cached"] is True and len(connector.discover_calls) == 1 and restarted.queries_remaining("pod_a", "vendor_a") == 1
    with pytest.raises(ValueError):
        restarted.set_provider_budget("pod_a", "vendor_a", 99)  # a spent budget cannot be re-stated upward


def test_concurrent_discovery_never_exceeds_budget(env, tmp_path):
    svc, connector, _, make = env
    svc.set_provider_budget("pod_c", "vendor_a", 3)
    results = []

    def work(i):
        try:
            results.append(make().discover(ctx(ns="pod_c"), "pod_c", "vendor_a", f"topic number {i}")["status"])
        except PodPlatformError as exc:
            results.append(exc.response.error.code)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("ok") == 3 and results.count(ErrorCode.PROVIDER_BUDGET_EXCEEDED) == 5 and len(connector.discover_calls) == 3
    assert svc.queries_remaining("pod_c", "vendor_a") == 0


def _discovered(svc, ns="pod_a"):
    return svc.discover(ctx(ns=ns), ns, "vendor_a", "web traffic")["fingerprint"]


def test_proposals_are_own_namespace_pending_and_deduplicated(env):
    svc, _, catalogue, _ = env
    fp = _discovered(svc)
    prop = svc.propose_source(ctx(), "pod_a", fp, "good_feed", "Visits could proxy demand ahead of reports.")
    assert prop["status"] == "pending_review" and prop["kind"] == "new_source" and prop["formation_eligible"] is False and prop["duplicate"] is False
    assert svc.propose_source(ctx(), "pod_a", fp, "good_feed", "Visits could proxy demand ahead of reports.")["duplicate"] is True
    assert svc.propose_source(ctx(), "pod_a", fp, "no_adapter", "Needs a connector to be built later.")["kind"] == "connector_work_item"
    with pytest.raises(PodPlatformError) as exc:
        svc.propose_source(ctx(ns="pod_b"), "pod_a", fp, "good_feed", "Proposing into another pod namespace.")
    assert _code(exc) == ErrorCode.PARTITION_ACCESS_DENIED
    with pytest.raises(PodPlatformError):  # pod_b cannot reuse pod_a's discovery result
        svc.propose_source(ctx(ns="pod_b"), "pod_b", fp, "good_feed", "Borrowing another pod's discovery.")
    with pytest.raises(PodPlatformError):  # already in the catalogue under dataset ds_CA0010
        svc.propose_source(ctx(), "pod_a", fp, "in_catalogue", "Duplicate of an approved source card.")
    with pytest.raises(PodPlatformError):
        svc.propose_source(ctx(), "pod_a", fp, "invented", "Never returned by any discovery query.")
    assert [p["provider_ref"] for p in svc.list_proposals(ctx())] == ["good_feed", "no_adapter"]
    assert svc.list_proposals(ctx("integrity"), "pod_b") == [] and len(catalogue) == 2


def _accept_card(established=True, evidence="att_hist"):
    item = ScreenItem(status=AssessmentStatus.ESTABLISHED, evidence_id=evidence)
    return SourceCard(owner="integrity", created_at=ASOF, source_id="NEW0001", dataset_id="vendor_ds_1", topic="web_traffic", snapshot_id="s1",
                      status=CardStatus.APPROVED, access_confirmed=True, formation_eligible=True, entity_grain=["security"],
                      screen=Screen(history=item, metric_lag=item, vintage=item, coverage=item))


def test_review_requires_integrity_and_recorded_non_sample_evidence(env):
    svc, _, catalogue, _ = env
    fp = _discovered(svc)
    pid = svc.propose_source(ctx(), "pod_a", fp, "good_feed", "Visits could proxy demand ahead of reports.")["proposal_id"]
    sample = svc.sample(ctx(), "pod_a", fp, "good_feed")
    with pytest.raises(PodPlatformError) as exc:  # the proposing pod cannot approve itself
        svc.review_proposal(ctx(), pid, "accept", "self approval", _accept_card())
    assert _code(exc) == ErrorCode.PARTITION_ACCESS_DENIED
    integrity = ctx("integrity")
    for bad in (_accept_card(evidence="never_recorded"), _accept_card(evidence=sample["evidence_id"]), {"approved": True}, None):
        with pytest.raises(PodPlatformError):
            svc.review_proposal(integrity, pid, "accept", "no evidence", bad)
    with pytest.raises(PodPlatformError):  # a sample id cannot be recorded as an attestation either
        svc.record_evidence(integrity, pid, sample["evidence_id"], "sample", "a" * 64)
    svc.record_evidence(integrity, pid, "att_hist", "Vendor history statement, signed.", "a" * 64)
    with pytest.raises(PodPlatformError):  # evidence belongs to another proposal / is missing for other screens only if absent
        svc.review_proposal(integrity, "prop_unknown", "accept", "x", _accept_card())
    done = svc.review_proposal(integrity, pid, "accept", "Evidence checked.", _accept_card())
    assert done["status"] == "accepted_pending_import" and done["formation_eligible"] is False
    card = svc.reviewed_card(pid)
    assert card.source_id == "NEW0001" and card.verify_seal() and len(catalogue) == 2  # catalogue is not mutated
    with pytest.raises(PodPlatformError):
        svc.review_proposal(integrity, pid, "reject", "second review", None)


def test_reject_needs_no_card_and_discovery_alone_never_flips_eligibility(env):
    svc, _, _, _ = env
    fp = _discovered(svc)
    pid = svc.propose_source(ctx(), "pod_a", fp, "good_feed", "Visits could proxy demand ahead of reports.")["proposal_id"]
    assert svc.review_proposal(ctx("integrity"), pid, "reject", "Insufficient history.")["status"] == "rejected"
    assert svc.reviewed_card(pid) is None
