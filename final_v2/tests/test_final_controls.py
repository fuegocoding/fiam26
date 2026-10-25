"""Regression tests for the two final controls: actual-fill risk review and namespace-scoped worker claims.

Offline only; no network and no model call.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from pod_platform.contracts.errors import ErrorCode, PodPlatformError
from pod_platform.contracts.portfolio import RiskDecision
from pod_platform.orchestration.replay import ControlledReplay, load_fixture
from pod_platform.pods import PodWorker
from pod_platform.research.jobs import ResearchQueue

from .test_pod_research import Clock, job_payload, make_runner
from .test_replay import FORMATIONS, LONGS, SHORTS, code_of, make_fixture

# ------------------------------------------------------------ actual-fill review


class _Book:
    """Minimal stand-in: the independent engine reads only the book id; replay copies it."""

    def __init__(self, target_book_id: str):
        self.target_book_id = target_book_id

    def model_copy(self, update):
        return _Book(update["target_book_id"])


def _replay(tmp_path, **risk_over):
    fixture = load_fixture(make_fixture(tmp_path / "fx"))
    fixture.config["risk"].update(risk_over)
    runner = ControlledReplay(fixture, tmp_path / "out")
    engine, risk_cfg = runner._risk_engine()
    return runner, engine, risk_cfg, fixture.market[FORMATIONS[0]], FORMATIONS[0]


def _review(runner, engine, risk_cfg, rows, weights, formation):
    from pod_platform.orchestration.replay import utc_close

    return runner._review_actual_fills(
        engine, risk_cfg, _Book("book_x_postrisk"), weights, rows, formation.isoformat(), utc_close(formation)
    )


def _weights(long_total: str, short_total: str) -> dict[str, Decimal]:
    out = {sid: Decimal(long_total) / len(LONGS) for sid in LONGS}
    out.update({sid: Decimal(short_total) / len(SHORTS) for sid in SHORTS})
    return out


def test_uneven_fills_pass_official_but_breach_stricter_internal_net(tmp_path):
    runner, engine, risk_cfg, rows, formation = _replay(tmp_path, max_abs_net=0.05)
    weights = _weights("0.35", "-0.15")  # net +0.20: inside official +-0.5, outside internal 0.05
    caps = {sid: row.market_cap_usd for sid, row in rows.items()}
    report = runner.compliance.record(formation.isoformat(), "post_trade_actual", {}, weights, caps)
    assert report.official_pass
    with pytest.raises(PodPlatformError) as err:
        _review(runner, engine, risk_cfg, rows, weights, formation)
    assert code_of(err) == ErrorCode.PORTFOLIO_INFEASIBLE and "actual fills" in str(err.value)
    entry = runner.actual_review_log[-1]
    assert entry["stage"] == "actual_fill" and entry["reviewed_kind"] == "actual" and entry["decision"] == "BLOCK"
    assert entry["post_weights"] and any(c["name"] == "abs_net" and c["status"] == "fail" for c in entry["limit_checks"])
    # failure evidence is written, separate from the target-book review log
    runner._write_failure("PORTFOLIO_INFEASIBLE", "test")
    partial = json.loads((tmp_path / "out" / "risk_reviews_actual_fill.partial.json").read_text())
    assert partial[-1]["decision"] == "BLOCK" and runner.review_log == []


def test_balanced_actual_fills_pass_and_are_logged_separately(tmp_path):
    runner, engine, risk_cfg, rows, formation = _replay(tmp_path, max_abs_net=0.05)
    review = _review(runner, engine, risk_cfg, rows, _weights("0.5", "-0.5"), formation)
    assert review.decision == RiskDecision.PASS and review.reviewed_kind == "actual"
    assert review.reviewed_book_id == "book_x_postrisk_actualfill"
    assert runner.actual_review_log[-1]["decision"] == "PASS" and runner.review_log == []


def test_actual_beta_uses_measured_exposure_and_soft_limit_is_not_blocking(tmp_path):
    runner, engine, risk_cfg, rows, formation = _replay(tmp_path, max_abs_beta=0.1)
    review = _review(runner, engine, risk_cfg, rows, _weights("0.35", "-0.15"), formation)
    assert float(review.measured_exposures["abs_beta"]) == pytest.approx(0.2)  # fixture betas are 1.0
    assert review.decision == RiskDecision.PASS_WITH_REMEDY  # beta is a soft limit in the shared policy


def test_actual_review_without_measurement_evidence_fails_closed_and_is_logged(tmp_path):
    runner, engine, risk_cfg, rows, formation = _replay(tmp_path, max_abs_beta=0.1)
    weights = _weights("0.5", "-0.5")
    weights["GHOST"] = Decimal("0.01")  # held name with no market/beta evidence
    with pytest.raises(PodPlatformError) as err:
        _review(runner, engine, risk_cfg, rows, weights, formation)
    assert code_of(err) == ErrorCode.RISK_REVIEW_REQUIRED
    assert runner.actual_review_log[-1]["decision"] == "INVALID" and "GHOST" in runner.actual_review_log[-1]["post_weights"]


def test_full_run_records_actual_fill_review_for_every_period(tmp_path):
    fixture = load_fixture(make_fixture(tmp_path / "fx"))
    out = tmp_path / "out"
    ControlledReplay(fixture, out).run()
    actual = json.loads((out / "risk_reviews_actual_fill.json").read_text())
    assert [r["decision"] for r in actual] == ["PASS"] * 4 and all(r["reviewed_kind"] == "actual" for r in actual)
    assert len(json.loads((out / "risk_reviews.json").read_text())) == 4
    periods = json.loads((out / "periods.json").read_text())
    assert all(p["actual_fill_risk_decision"] == "PASS" and p["actual_fill_risk_review_digest"] for p in periods)


# ------------------------------------------------------------- namespace claims


def _two_pod_worker(tmp_path, configured=("valuation", "events")):
    clk = Clock()
    queue = ResearchQueue(tmp_path / "q.sqlite", clock=clk)
    runners = {}
    for pod in configured:
        (tmp_path / pod).mkdir()
        runner, *_ = make_runner(tmp_path / pod, pod=pod, clk=clk)
        runners[runner.pod.namespace] = runner
    return PodWorker(queue, runners, lease_seconds=60, retry_delay_s=0.0), queue, runners, clk


def test_empty_runner_mapping_claims_nothing(tmp_path):
    queue = ResearchQueue(tmp_path / "q.sqlite", clock=Clock())
    job = queue.submit("pod_valuation", "propose", job_payload())
    assert PodWorker(queue, {}).run_once() is None
    after = queue.get(job.job_id)
    assert after.state == "queued" and after.attempts == 0 and after.lease_token is None


def test_unconfigured_namespace_lease_recovery_is_not_touched(tmp_path):
    worker, queue, runners, clk = _two_pod_worker(tmp_path, configured=("valuation",))
    foreign = queue.submit("pod_events", "propose", job_payload(), max_attempts=3)
    claimed = queue.claim("other-worker", lease_seconds=60, namespace="pod_events")
    assert claimed.job_id == foreign.job_id
    clk.advance(seconds=120)  # foreign lease is now expired
    own = queue.submit("pod_valuation", "propose", job_payload())
    outcome = worker.run_once()
    assert outcome.job_id == own.job_id and outcome.state == "succeeded"
    after = queue.get(foreign.job_id)
    assert after.state == "running" and after.lease_token == claimed.lease_token and after.attempts == claimed.attempts
    assert worker.run_once() is None
    assert queue.get(foreign.job_id).state == "running"


def test_configured_pods_are_served_fairly_and_deterministically(tmp_path):
    worker, queue, runners, _ = _two_pod_worker(tmp_path)
    for n in range(2):
        for ns in ("pod_valuation", "pod_events"):
            queue.submit(ns, "propose", job_payload(question=f"Question {n} for {ns}?"))
    served = [queue.get(worker.run_once().job_id).namespace for _ in range(4)]
    assert served == ["pod_events", "pod_valuation", "pod_events", "pod_valuation"]
    assert worker.run_once() is None
