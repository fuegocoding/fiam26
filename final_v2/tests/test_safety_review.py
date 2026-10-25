"""Regression tests for the final safety review (replay accounting, chronology,
provider bounds, queue scope, memory integrity, historical paper evidence)."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from pod_platform.agents.provider import (
    CLAUDE_IGNORED_PARAMS,
    ClaudeCodeProvider,
    GenerationParams,
    IncompleteOutputError,
    LLMRequest,
    ProviderError,
    call_structured,
)
from pod_platform.contracts.errors import ErrorCode, PodPlatformError
from pod_platform.orchestration.replay import (
    ControlledReplay,
    ResolvedOutcomes,
    load_fixture,
    utc_close,
)
from pod_platform.papers.models import PapersError
from pod_platform.pods.runner import ExecutorError, ExecutorRegistry
from pod_platform.pods.schema import normalise_evidence
from pod_platform.research.jobs import QueueError
from pod_platform.research.memory import PodMemory

from .test_papers import NOW as PAPERS_NOW
from .test_papers import Fake, local_card, svc_for
from .test_pod_research import good_executor, make_queue, make_runner, propose
from .test_replay import ENDS, FORMATIONS, _caps, _liq, _replay, code_of, make_fixture

UTC = dt.timezone.utc


# ------------------------------------------------------------ replay accounting


def _open_period(replay):
    replay.rebalance_to_targets(
        {"AAA": Decimal("0.5"), "BBB": Decimal("-0.5")}, at=utc_close(FORMATIONS[0]), cost_bp_per_side=Decimal(10)
    )


def _advance(replay, resolved):
    return replay.advance(
        period_end_at=utc_close(ENDS[0]), rf_monthly=0.003, resolved=resolved, market_caps=_caps("AAA", "BBB")
    )


def test_impossible_total_price_fails_closed_and_leaves_ledger_untouched():
    replay = _replay()
    _open_period(replay)
    nav_before, cash_before = replay.nav(), replay.ledger.cash
    with pytest.raises(PodPlatformError) as err:  # -150% excess: previously floored to 1e-12 and reported
        _advance(replay, ResolvedOutcomes(returns={"AAA": -1.5, "BBB": 0.0}, basis={}))
    assert code_of(err) == ErrorCode.ACCOUNTING_RECONCILIATION_FAILED and "AAA" in str(err.value)
    assert replay.nav() == nav_before and replay.ledger.cash == cash_before  # nothing accrued before the check


def test_exact_minus_100_percent_needs_dated_liquidation_evidence():
    # rf 0.003 + excess -1.003 => total price exactly zero
    with pytest.raises(PodPlatformError) as err:
        replay = _replay()
        _open_period(replay)
        _advance(replay, ResolvedOutcomes(returns={"AAA": -1.003, "BBB": 0.0}, basis={}))
    assert code_of(err) == ErrorCode.ACCOUNTING_RECONCILIATION_FAILED

    replay = _replay()
    _open_period(replay)
    evidence = _liq(when=dt.datetime(2024, 2, 15, tzinfo=UTC), value=-1.003)
    result = _advance(
        replay, ResolvedOutcomes(returns={"AAA": -1.003, "BBB": 0.0}, liquidated={"AAA": evidence}, basis={})
    )
    assert result.liquidated == ["AAA"] and "AAA" not in replay.held()
    assert replay.nav() > 0


# ------------------------------------------------------------ forecast / certificate chronology


def _rewrite(root, name, fn):
    path = root / name
    docs = json.loads(path.read_text())
    for doc in docs:
        fn(doc)
    path.write_text(json.dumps(docs))


def test_forecast_published_after_decision_is_never_used(tmp_path):
    root = make_fixture(tmp_path / "fx")

    def late(doc):  # information_cutoff stays old; only the publication time moves past the decision
        if doc["forecast_id"].startswith("fc_2024-02-29"):
            doc["created_at"] = "2024-03-20T00:00:00+00:00"

    _rewrite(root, "forecasts.json", late)
    with pytest.raises(PodPlatformError) as err:
        ControlledReplay(load_fixture(root), tmp_path / "out").run()
    assert "no active approved forecasts at 2024-02-29" in str(err.value)


def test_certificate_must_cover_target_horizon_and_predate_decision(tmp_path):
    root = make_fixture(tmp_path / "fx_target")
    _rewrite(root, "certificates.json", lambda d: d.update(target_id="some_other_target"))
    with pytest.raises(PodPlatformError) as err:
        ControlledReplay(load_fixture(root), tmp_path / "o1").run()
    assert code_of(err) == ErrorCode.ADMISSION_CERTIFICATE_INVALID and "target" in str(err.value)

    root = make_fixture(tmp_path / "fx_horizon")
    _rewrite(root, "certificates.json", lambda d: d.update(horizon_trading_days=5))
    with pytest.raises(PodPlatformError) as err:
        ControlledReplay(load_fixture(root), tmp_path / "o2").run()
    assert code_of(err) == ErrorCode.ADMISSION_CERTIFICATE_INVALID

    root = make_fixture(tmp_path / "fx_backdated")  # effective_start is old but the certificate was issued later
    _rewrite(root, "certificates.json", lambda d: d.update(created_at="2024-02-01T00:00:00+00:00"))
    with pytest.raises(PodPlatformError) as err:
        ControlledReplay(load_fixture(root), tmp_path / "o3").run()
    assert code_of(err) == ErrorCode.ADMISSION_CERTIFICATE_INVALID and "after the decision" in str(err.value)


# ------------------------------------------------------------ Claude CLI provider


class Answer(BaseModel):
    label: str
    score: float


def _runner(stdout=b'{"label":"a","score":1}', calls=None):
    def run(cmd, **kwargs):
        if calls is not None:
            calls.append(cmd)
        return SimpleNamespace(stdout=stdout, stderr=b"", returncode=0)

    return run


def _request(**params):
    return LLMRequest(system="sys", user="{}", params=GenerationParams(**params))


def test_claude_cli_reports_unapplied_params_in_provenance():
    provider = ClaudeCodeProvider("claude-x", runner=_runner())
    response = provider.complete(_request(max_tokens=256, temperature=0.5, seed=3))
    assert response.ignored_params == CLAUDE_IGNORED_PARAMS
    assert response.prompt_tokens is None and response.completion_tokens is None  # usage is not invented
    record = call_structured(provider, _request(max_tokens=256), Answer).record
    assert record.ignored_params == list(CLAUDE_IGNORED_PARAMS)


def test_claude_cli_enforces_max_tokens_after_the_fact_and_input_bound():
    with pytest.raises(IncompleteOutputError):  # 23 bytes cannot be <= 1 token
        ClaudeCodeProvider(runner=_runner()).complete(_request(max_tokens=1))
    calls: list = []
    with pytest.raises(ProviderError, match="input limit"):
        ClaudeCodeProvider(max_input_bytes=10, runner=_runner(calls=calls)).complete(_request())
    assert calls == []  # nothing was launched


# ------------------------------------------------------------ queue / memory


def test_claim_recovery_is_namespace_scoped(tmp_path):
    queue, clk = make_queue(tmp_path)
    a = queue.submit("pod_valuation", "propose", {"n": 1}, max_attempts=3)
    b = queue.submit("pod_events", "propose", {"n": 2}, max_attempts=1)
    queue.claim("w1", lease_seconds=10, namespace="pod_valuation")
    queue.claim("w2", lease_seconds=10, namespace="pod_events")
    clk.advance(seconds=11)
    again = queue.claim("w3", namespace="pod_valuation")
    assert again.job_id == a.job_id and again.attempts == 2  # attempts carry over a crash
    assert queue.get(b.job_id).state == "running"  # another pod's expired lease is not touched
    assert queue.recover_expired("pod_events") == [b.job_id]
    assert queue.get(b.job_id).state == "failed"


def test_memory_entries_are_hash_verified_and_evaluated_needs_a_stored_draft(tmp_path):
    memory = PodMemory(tmp_path, "pod_valuation")
    entry = memory.append("note", {"title": "original"})
    assert memory.get(entry.entry_id).content == {"title": "original"}
    conn = sqlite3.connect(memory.path, isolation_level=None)
    conn.execute("DROP TRIGGER entries_no_update")
    conn.execute("UPDATE entries SET content_json = ?", (json.dumps({"title": "forged"}),))
    conn.close()
    with pytest.raises(QueueError, match="content hash"):
        memory.get(entry.entry_id)

    fresh = PodMemory(tmp_path / "other", "pod_valuation")
    claim = {"registration_id": "r", "executor_id": "e", "metrics": {"ic": 0.1}}
    with pytest.raises(QueueError, match="draft_proposal"):
        fresh.append("evaluated_result", dict(claim, draft_entry_id="mem_missing"))
    draft = fresh.append("draft_proposal", {"title": "d"})
    with pytest.raises(QueueError, match="admission"):
        fresh.append("evaluated_result", dict(claim, draft_entry_id=draft.entry_id, admission="admitted"))
    with pytest.raises(QueueError, match="finite"):
        fresh.append("evaluated_result", dict(claim, draft_entry_id=draft.entry_id, metrics={"ic": "great"}))
    ok = fresh.append("evaluated_result", dict(claim, draft_entry_id=draft.entry_id))
    assert ok.evaluated


def test_colliding_memory_key_is_not_reported_as_an_executor_result(tmp_path):
    runner, _provider, memory, _clk = make_runner(tmp_path)
    draft_id = propose(runner).memory_entry.entry_id
    memory.append("note", {"title": "planted"}, idempotency_key=f"eval:{draft_id}:v0")
    executors = ExecutorRegistry()
    executors.register("exec_a", good_executor)
    with pytest.raises(ExecutorError, match="not an executor result"):
        runner.run_experiments(draft_id, executors)
    assert memory.evaluated_results() == []


# ------------------------------------------------------------ evidence time boundary


def test_card_revised_after_research_date_is_not_old_evidence():
    card = {"paper_id": "P9", "title": "Old paper", "abstract": "Rewritten abstract.",
            "published_at": "2019-01-01", "updated_at": "2025-06-01"}
    (item,) = normalise_evidence([card], "paper_card")
    assert item.available_at == dt.datetime(2025, 6, 1, 23, 59, 59, 999999, tzinfo=UTC)


# ------------------------------------------------------------ papers store


def test_modern_openalex_abstract_is_not_served_as_old_evidence(tmp_path):
    s = svc_for(tmp_path, Fake())
    modern = s.get("doi:10.1000/xyz")
    assert modern.abstract == "Hello world" and modern.retrieved_at == PAPERS_NOW
    old = s.get("doi:10.1000/xyz", as_of="2020-01-01")  # dated 2019, but the text was fetched in 2026
    assert old.abstract is None and old.abstract_hash is None
    assert old.access_status.value == "metadata_only" and old.text_status.startswith("incomplete")
    assert s.ingest_full_text("doi:10.1000/xyz", as_of="2020-01-01").abstract is None
    assert s.get("doi:10.1000/xyz").abstract == "Hello world"  # the stored row was not damaged


def test_stored_content_and_cards_are_hash_verified_on_read(tmp_path):
    s = svc_for(tmp_path, None, online=False)
    card = local_card(tmp_path, s)
    assert s.store.get_content(card.content_hash)
    s.store._db.execute("UPDATE contents SET text = text || ' tampered'")
    with pytest.raises(PapersError, match="content hash"):
        s.store.get_content(card.content_hash)
    with pytest.raises(PapersError, match="content hash"):
        s.text_of(card)
    s.store._db.execute("UPDATE versions SET card_json = replace(card_json, 'Local drift', 'Forged title')")
    with pytest.raises(PapersError, match="card hash"):
        s.store.rows(card.paper_id)
