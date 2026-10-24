"""Source-card and hypothesis registration acceptance (spec 8.4, 21.3-21.4, 9.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pod_platform.contracts import AssessmentStatus, CardStatus, HypothesisSpec, ResearchMode
from pod_platform.contracts.sources import SourceCard

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
CARDS = CONFIGS / "source_cards"
HYPOS = CONFIGS / "hypotheses"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _card(name: str) -> SourceCard:
    return SourceCard.model_validate(_load(CARDS / name))


def test_all_s1_source_cards_validate():
    cards = sorted(CARDS.glob("*.json"))
    assert len(cards) == 8
    for path in cards:
        card = _card(path.name)
        assert card.dataset_id.startswith(("CA", "COMP"))
        assert card.owner == "data"


def test_competition_panel_is_formation_eligible():
    panel = _card("COMP_chars_panel.json")
    assert panel.formation_eligible is True
    assert panel.screen.vintage.status == AssessmentStatus.ESTABLISHED
    filings = _card("COMP_8k_filings.json")
    assert filings.formation_eligible is False
    assert any("survivorship" in q.lower() for q in filings.unresolved_questions)


def test_no_carbon_card_is_formation_eligible_yet():
    for path in sorted(CARDS.glob("CA*.json")):
        card = _card(path.name)
        assert card.formation_eligible is False, f"{card.dataset_id} claims formation eligibility without full screens"
        assert card.broker_rejects_formation() is True


def test_binding_questions_are_unresolved_not_hidden():
    for path in sorted(CARDS.glob("CA*.json")):
        card = _card(path.name)
        screens = card.screen
        unresolved = [
            name
            for name, item in (
                ("history", screens.history),
                ("metric_lag", screens.metric_lag),
                ("vintage", screens.vintage),
                ("coverage", screens.coverage),
            )
            if item.status != AssessmentStatus.ESTABLISHED
        ]
        assert unresolved, f"{card.dataset_id} claims everything established"
        assert card.unresolved_questions, f"{card.dataset_id} hides its open questions"
        for question in card.unresolved_questions:
            assert len(question) > 20


def test_ca0056_vintage_reweighting_is_recorded():
    card = _card("CA0056_card_spend.json")
    assert card.screen.vintage.status == AssessmentStatus.UNRESOLVED
    assert any("reweight" in q.lower() for q in card.unresolved_questions)
    assert card.screen.metric_lag.status == AssessmentStatus.UNRESOLVED


def test_ca0097j_has_the_best_clocks_but_still_open_vintage():
    card = _card("CA0097J_job_postings.json")
    assert card.screen.history.status == AssessmentStatus.ESTABLISHED
    assert card.screen.metric_lag.status == AssessmentStatus.ESTABLISHED
    assert card.screen.vintage.status == AssessmentStatus.UNRESOLVED
    assert any("last_updated" in q or "vendor edits" in q.lower() or "mutated" in q.lower() for q in card.unresolved_questions)


def test_identity_and_coverage_caveats_recorded():
    for name in ("CA0056_card_spend.json", "CA0097J_job_postings.json"):
        card = _card(name)
        assert any("ticker" in q.lower() or "identity" in q.lower() or "resolution" in q.lower() for q in card.unresolved_questions)


def test_established_screens_carry_evidence():
    for path in sorted(CARDS.glob("*.json")):
        card = _card(path.name)
        for item in (card.screen.history, card.screen.metric_lag, card.screen.vintage, card.screen.coverage):
            if item.status == AssessmentStatus.ESTABLISHED:
                assert item.evidence_id and len(item.evidence_id) > 10


def test_charter1_hypothesis_validates():
    hypo = HypothesisSpec.model_validate(_load(HYPOS / "hyp_s1_card_spend_revenue_001.json"))
    assert hypo.owner == "strategy_1"
    assert hypo.research_mode == ResearchMode.CONTEMPORARY_RESEARCH
    assert hypo.mechanism_family == "demand_measurement"
    assert hypo.variant_budget == 3
    assert len(hypo.falsification) >= 4


def test_hypothesis_targets_fundamentals_not_returns():
    hypo = HypothesisSpec.model_validate(_load(HYPOS / "hyp_s1_card_spend_revenue_001.json"))
    assert "revenue" in hypo.target_id
    assert "return" not in hypo.target_id


def test_hypothesis_declares_the_bridge_separately():
    hypo = HypothesisSpec.model_validate(_load(HYPOS / "hyp_s1_card_spend_revenue_001.json"))
    assert any("bridge" in f.lower() for f in hypo.falsification)


def test_hypothesis_has_kill_conditions_not_just_success():
    hypo = HypothesisSpec.model_validate(_load(HYPOS / "hyp_s1_card_spend_revenue_001.json"))
    joined = " ".join(hypo.falsification).lower()
    for word in ("baseline", "composition", "sector", "cost"):
        assert word in joined


def test_ca0032_edgar_clocks_established():
    card = _card("CA0032_edgar_filings.json")
    assert card.screen.history.status == AssessmentStatus.ESTABLISHED
    assert card.screen.metric_lag.status == AssessmentStatus.ESTABLISHED
    assert card.screen.vintage.status == AssessmentStatus.ESTABLISHED
    assert card.screen.coverage.status == AssessmentStatus.SAMPLE_ONLY
    assert any("extract" in q.lower() for q in card.unresolved_questions)


def test_ca0092_drop_partition_vintage_evidence():
    card = _card("CA0092_securities_ratings.json")
    assert card.screen.vintage.status == AssessmentStatus.ESTABLISHED
    assert "ca_drop_partition" in card.screen.vintage.evidence_id
    assert any("backfill" in q.lower() for q in card.unresolved_questions)
    assert card.screen.metric_lag.status == AssessmentStatus.ESTABLISHED


def test_ca0035_events_pin_tagging_question():
    card = _card("CA0035_benzinga_events.json")
    assert card.screen.metric_lag.status == AssessmentStatus.ESTABLISHED
    assert card.screen.vintage.status == AssessmentStatus.UNRESOLVED
    assert any("tag" in q.lower() or "retag" in q.lower() for q in card.unresolved_questions)


def test_blocked_panels_record_the_rewrite_evidence():
    card = _card("CA0056_card_spend.json")
    assert card.screen.vintage.status == AssessmentStatus.UNRESOLVED
    assert any("reweight" in q.lower() for q in card.unresolved_questions)
    traffic = _card("CA0060_foot_traffic.json")
    assert traffic.screen.vintage.status == AssessmentStatus.UNRESOLVED


def test_status_doc_exists_and_covers_all_cards():
    doc = (CONFIGS.parent / "CARBON_DATA_STATUS.md").read_text(encoding="utf-8")
    for dataset_id in ("CA0032", "CA0035", "CA0056", "CA0060", "CA0092", "CA0097J"):
        assert dataset_id in doc
    for section in ("What we can use", "What we can't use", "Identity and coverage"):
        assert section in doc
