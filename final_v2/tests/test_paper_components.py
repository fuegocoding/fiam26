"""Source-grounded component research can use small evidence without claiming replication."""
import datetime as dt

import pytest

from pod_platform.papers import PaperService, PaperStore
from pod_platform.papers.components import ComponentEvidenceService, ComponentRejected
from pod_platform.papers.models import AccessStatus, PaperCard, sha256_text

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 25, 12, tzinfo=UTC)
CUTOFF = dt.datetime(2020, 6, 1, tzinfo=UTC)
DATE = dt.datetime(2020, 1, 1, tzinfo=UTC)
QUOTE = "Estimate the response using a rank transformation of the observed variable."


@pytest.fixture
def env(tmp_path):
    papers = PaperService(PaperStore(tmp_path / "papers.sqlite"), clock=lambda: NOW)
    papers.store.upsert_card(PaperCard(paper_id="arxiv:2001.12345", title="Synthetic component paper",
        arxiv_id="2001.12345", version=1, latest_version=2, version_key="v1", source="arxiv",
        first_public_date=DATE, version_date=DATE, abstract=QUOTE, abstract_hash=sha256_text(QUOTE),
        access_status=AccessStatus.ABSTRACT, text_status="abstract_only", retrieved_at=NOW))
    card = papers.add_claim("arxiv:2001.12345v1", kind="statistical_design", quote=QUOTE, source="abstract")
    service = ComponentEvidenceService(papers=papers, namespace="pod_valuation", clock=lambda: NOW)
    request = dict(namespace="pod_valuation", component_kind="method", component_name="rank response estimate",
        paper_id="arxiv:2001.12345v1", version_key="v1", claim_ids=[card.claims[0].claim_id],
        source="abstract", as_of=CUTOFF, unavailable_controls=["Paper's proprietary control sample unavailable."])
    return papers, service, request


def test_pinned_abstract_single_method_claim_is_usable_without_replication(env):
    papers, service, request = env
    result = service.resolve(request)
    assert result.label == "component_only" and result.replication_status == "not_replication"
    assert result.claims[0].quote == QUOTE and result.source_hash == sha256_text(QUOTE)
    assert result.version_date == DATE and result.retrieved_at == NOW
    assert result.evidence.availability_verified and result.evidence.available_at == DATE
    assert "Abstract only" in result.evidence.text and "Unavailable control" in result.evidence.text
    assert "Retrieved after as_of" in result.evidence.text
    assert service.verify(result.model_dump(mode="json")) == result
    # Same selected cached source after service restart has the same binding.
    restarted = ComponentEvidenceService(papers=PaperService(PaperStore(papers.store.path), clock=lambda: NOW),
        namespace="pod_valuation", clock=lambda: NOW)
    assert restarted.resolve(request).model_dump() == result.model_dump()
    strict = papers.create_replication_spec("arxiv:2001.12345v1", requested_label="METHOD_TRANSFER", as_of=CUTOFF)
    assert strict["label"] == "INCOMPLETE"  # component extraction does not loosen replication


def test_single_cached_native_component_excerpt_does_not_need_full_paper(env):
    papers, service, request = env
    card = papers.get(request["paper_id"])
    body = "Introduction\nThe mechanism responds to slow adjustment of the price.\n"
    card = papers._attach(card, body, sha256_text(body))
    papers.store.save_card(card.model_copy(update={"content_truncated": True}))
    card = papers.add_claim(request["paper_id"], kind="mechanism", quote=body.splitlines()[1])
    request.update(component_kind="mechanism", component_name="slow price adjustment", source="full_text",
                   claim_ids=[card.claims[0].claim_id])
    component = service.resolve(request)
    assert len(component.claims) == 1 and component.claims[0].kind == "mechanism"
    assert any("truncated" in x for x in component.limitations)
    assert component.evidence.evidence_id == component.component_id


@pytest.mark.parametrize("change", [
    {"label": "EXACT_REPLICATION"}, {"label": "whole_paper_replicated"},
    {"formation_valid": True}, {"approved_source": True}, {"approved_model": True},
    {"paper_id": "/tmp/paper.txt"}, {"paper_id": "https://arxiv.org/abs/2001.12345v1"},
    {"component_kind": "own_reasoning"},
])
def test_cannot_escalate_component_scope_or_forge_source(env, change):
    _, service, request = env
    with pytest.raises(ComponentRejected):
        service.resolve({**request, **change})


def test_quote_tamper_and_changed_source_are_denied(env):
    papers, service, request = env
    component = service.resolve(request)
    forged = component.model_dump(mode="json")
    forged["claims"][0]["quote"] = "Invented method superiority."
    with pytest.raises(ComponentRejected, match="component_tamper"):
        service.verify(forged)
    card = papers.get(request["paper_id"])
    claim = card.claims[0].model_copy(update={"quote": "Invented claim"})
    papers.store.save_card(card.model_copy(update={"claims": [claim]}))
    with pytest.raises(ComponentRejected, match="ungrounded_claims"):
        service.resolve(request)


@pytest.mark.parametrize("update,code", [
    ({"version_date": None}, "future_version"),
    ({"version_date": NOW}, "future_version"),
    ({"retrieved_at": NOW + dt.timedelta(seconds=1)}, "future_receipt"),
])
def test_unknown_or_future_versions_and_receipts_are_denied(env, update, code):
    papers, service, request = env
    papers.store.save_card(papers.get(request["paper_id"]).model_copy(update=update))
    with pytest.raises(ComponentRejected, match=code):
        service.resolve(request)


def test_unversioned_late_text_is_not_backdated(env, tmp_path):
    papers, service, request = env
    path = tmp_path / "component.txt"
    path.write_text(QUOTE)
    card = papers.ingest_local(path, title="Local excerpt", first_public_date=DATE, version_date=DATE)
    card = papers.add_claim(card.paper_id, kind="statistical_design", quote=QUOTE)
    request.update(paper_id=card.paper_id, version_key=card.version_key, source="full_text",
                   claim_ids=[card.claims[0].claim_id])
    with pytest.raises(ComponentRejected, match="late_text"):
        service.resolve(request)
    request["as_of"] = NOW
    assert service.resolve(request).source == "full_text"


def test_no_network_fallback_and_namespace_isolation(env):
    papers, service, request = env
    papers.online = True
    papers._fetch = lambda *args: pytest.fail("component reads must never fetch")
    with pytest.raises(ComponentRejected, match="version_mismatch"):
        service.resolve({**request, "version_key": "v2"})
    with pytest.raises(ComponentRejected, match="namespace_mismatch"):
        service.resolve({**request, "namespace": "pod_events"})


def test_select_one_exact_cached_quote_without_network_or_full_text(env):
    papers, service, request = env
    papers._fetch = lambda *a: pytest.fail('quote selection must stay offline')
    card = service.quote(paper_id=request['paper_id'], version_key='v1', source='abstract',
                         kind='mechanism', quote=QUOTE)
    assert any(c.kind == 'mechanism' and c.quote == QUOTE for c in card.claims)
    with pytest.raises(ComponentRejected, match='ungrounded'):
        service.quote(paper_id=request['paper_id'], version_key='v1', source='abstract',
                      kind='mechanism', quote='Fabricated claim absent from the paper.')
    with pytest.raises(ComponentRejected, match='version_mismatch'):
        service.quote(paper_id=request['paper_id'], version_key='v2', source='abstract', kind='mechanism', quote=QUOTE)
