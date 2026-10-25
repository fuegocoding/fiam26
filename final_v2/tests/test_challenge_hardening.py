"""Hardening tests: sealed evidence, measured net-of-cost bridge, out-of-training calibration, native units, authority restart.

Synthetic fixture only; nothing here is investment evidence.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import pickle
import random

import pytest

from pod_platform.application.artifacts import ArtifactIndex
from pod_platform.application.integrity import IntegrityPolicy, IntegrityService, Outcome, net_spread_series
from pod_platform.application.namespaces import RoleContext
from pod_platform.application.publication import PublicationService
from pod_platform.contracts.errors import PodPlatformError
from pod_platform.research import challenge_bridge as bridge_module
from pod_platform.research.bridge_economics import CalibrationError, calibrate_return_bridge
from pod_platform.research.challenge_bridge import ChallengeBridge, CheckAuthority
from pod_platform.research.evaluation import ForecastOutcome
from pod_platform.research.registry import Registry

from .test_builtin_executors import ASOF
from .test_challenge_bridge import EVAL_ASOF, HOUR, authorize, build_world, challenge, make_service, summary

UTC = dt.timezone.utc


class Capture(IntegrityService):
    """Records the sealed bundle the bridge hands to integrity."""

    bundles: list = []

    def evaluate(self, *a, **k):
        Capture.bundles.append(k["evidence"])
        return super().evaluate(*a, **k)


# ------------------------------------------------------------------ 1. evidence is producer-sealed and bound


def test_unbound_observations_cannot_issue_research_or_formation_certificates(tmp_path):
    w = build_world(tmp_path)
    record = w.bridge.candidate_record(w.bctx, w.frozen[0])
    loose = make_service(w, tag="_loose", allow_unbound_fixture=True)
    for scope in ("research", "formation"):  # caller observations never support a real-scope certificate
        with pytest.raises(PodPlatformError) as exc:
            loose.evaluate(record, [], [], ctx=w.integrity, evaluation_asof=EVAL_ASOF, certificate_scope=scope)
        assert "unbound observations" in summary(exc)
    strict = make_service(w, tag="_strict")
    with pytest.raises(PodPlatformError):  # even fixture compatibility is switched off for hosts running real challenges
        strict.evaluate(record, [], [], ctx=w.integrity, evaluation_asof=EVAL_ASOF)
    # the bridge refuses a service that could accept unbound observations or holds another authority's seal
    for service in (loose, make_service(w, tag="_foreign", evidence_seal=CheckAuthority().evidence_seal())):
        with pytest.raises(PodPlatformError) as exc:
            challenge(w, service)
        assert "evidence seal" in summary(exc)
    assert w.bridge.family_accesses(w.frozen[0].mechanism_family) == 0


def test_sealed_bundle_is_bound_to_candidate_model_scope_and_content(tmp_path):
    w = build_world(tmp_path, variants=(("v1", {}), ("v2", {"feature_set": "extended"})))
    Capture.bundles = []
    first = challenge(w, make_service(w, Capture, tag="_cap"))
    assert first.decision.outcome == Outcome.ADMITTED and first.decision.labels["evidence_binding"].startswith("producer_sealed:ev_")
    bundle = Capture.bundles[0]
    assert first.decision.evidence["evidence_bundle_id"] == bundle.bundle_id
    rec0, rec1 = (w.bridge.candidate_record(w.bctx, f) for f in w.frozen)
    probe = make_service(w, tag="_probe")
    obs = list(bundle.check_observations)
    tampered_obs = (obs[0].model_copy(update={"prediction": obs[0].prediction + 5.0}),) + tuple(obs[1:])
    cases = {
        "forged": (rec0, dataclasses.replace(bundle, seal="0" * 64), {}),
        "tampered observation": (rec0, dataclasses.replace(bundle, check_observations=tampered_obs), {}),
        "tampered cost": (rec0, dataclasses.replace(bundle, cost_bps_one_way=0.0), {}),
        "replayed for another candidate": (rec1, bundle, {}),
        "wrong scope": (rec0, bundle, {"certificate_scope": "research"}),
        "caller observations differ": (rec0, bundle, {}),
    }
    for name, (record, evidence, extra) in cases.items():
        supplied = (obs[:3], obs[:3]) if name == "caller observations differ" else ((), ())
        with pytest.raises(PodPlatformError):
            probe.evaluate(record, *supplied, ctx=w.integrity, evaluation_asof=EVAL_ASOF, evidence=evidence, **extra)
    foreign = make_service(w, tag="_foreign2", evidence_seal=CheckAuthority().evidence_seal())
    with pytest.raises(PodPlatformError):  # a bundle sealed by another authority is unknown here
        foreign.evaluate(rec0, ctx=w.integrity, evaluation_asof=EVAL_ASOF, evidence=bundle)
    with pytest.raises(PodPlatformError):  # host role checks remain
        probe.evaluate(rec0, ctx=RoleContext("strategy", w.frozen[0].namespace, "r", EVAL_ASOF, ("design", "check")),
                       evaluation_asof=EVAL_ASOF, evidence=bundle)
    with pytest.raises(PodPlatformError):  # only integrity can mint a bundle
        w.authority.evidence_seal().issue(RoleContext("strategy", w.frozen[0].namespace, "r", EVAL_ASOF), record=rec0, scope="fixture",
                                          provenance={}, design_observations=[], check_observations=[], design_portfolio=[],
                                          check_portfolio=[], bridge_sign=1, tail_fraction=0.2, cost_bps_one_way=1.0, cost_source="x",
                                          issued_at=EVAL_ASOF)


# ------------------------------------------------------------------ 2. measured net-of-cost bridge


def _obs(day: int, scores: dict[str, float], rets: dict[str, float]) -> list[ForecastOutcome]:
    at = dt.datetime(2020, 1, day, tzinfo=UTC)
    return [ForecastOutcome(security_id=s, formation_at=at, feature_available_at=at, label_resolves_at=at + dt.timedelta(days=30),
                            prediction=p, baseline_prediction=0.0, target=0.0, net_forward_return=rets[s]) for s, p in scores.items()]


def _members(day: int, longs, shorts):
    return {"formation_at": dt.datetime(2020, 1, day, tzinfo=UTC).isoformat(), "longs": longs, "shorts": shorts}


def test_net_spread_prices_declared_turnover_and_rejects_inconsistent_membership():
    scores = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}
    day1, day2 = _obs(1, scores, {"A": .05, "B": 0, "C": 0, "D": 0, "E": -.03}), _obs(2, scores, {"A": .02, "B": 0, "C": 0, "D": 0, "E": 0})
    port = [_members(1, ["A"], ["E"]), _members(2, ["A"], ["E"])]
    rows = net_spread_series(day1 + day2, port, bridge_sign=1, tail_fraction=0.2, cost_bps_one_way=10.0)
    assert [round(r["turnover"], 9) for r in rows] == [2.0, 0.0]  # full entry, then no trading
    assert rows[0]["gross"] == pytest.approx(0.08) and rows[0]["net"] == pytest.approx(0.08 - 0.002)
    assert rows[1]["net"] == pytest.approx(0.02)
    # sign -1: the long leg is the LOWEST prediction and raw return = -net_forward_return
    flipped = _obs(1, scores, {"A": .05, "B": 0, "C": 0, "D": 0, "E": -.03})
    neg = net_spread_series(flipped, [_members(1, ["E"], ["A"])], bridge_sign=-1, tail_fraction=0.2, cost_bps_one_way=0.0)
    assert neg[0]["gross"] == pytest.approx(0.03 - (-0.05))
    # membership that is not the declared top tail (or has unknown names) is unmeasured, never assumed
    assert net_spread_series(day1, [_members(1, ["B"], ["E"])], bridge_sign=1, tail_fraction=0.2, cost_bps_one_way=1.0) is None
    assert net_spread_series(day1, [_members(1, ["A"], ["ZZ"])], bridge_sign=1, tail_fraction=0.2, cost_bps_one_way=1.0) is None


def test_measured_net_spread_is_required_and_present_in_the_decision(tmp_path):
    w = build_world(tmp_path)
    decision = challenge(w).decision
    m = decision.metrics
    assert decision.outcome == Outcome.ADMITTED and decision.criteria["net_bridge_met"]
    assert m["check_gross_spread"] > m["check_net_spread"] > 0 and m["check_turnover"] > 0 and m["design_net_spread"] > 0
    assert m["cost_bps_one_way_effective"] == 10.0  # the manifest's declared cost; host floor was 0


def test_high_cost_but_positive_signal_is_not_admitted(tmp_path):
    policy = IntegrityPolicy(policy_id="fixture_costly_v1", frozen_at=ASOF + HOUR, cost_bps_floor_one_way=1000.0)  # frozen before outcomes
    w = build_world(tmp_path, policy=policy)
    result = challenge(w)
    d = result.decision
    assert d.outcome == Outcome.SHADOW and d.certificate_id is None, d.reasons
    assert d.criteria["direction_met"] and d.criteria["improvement_met"] and not d.criteria["net_bridge_met"]
    assert d.metrics["check_gross_spread"] > 0 > d.metrics["check_net_spread"] and d.metrics["cost_bps_one_way_effective"] == 1000.0
    assert "criterion not met: net_bridge_met" in d.reasons
    frozen = w.frozen[0]
    with pytest.raises(PodPlatformError):
        w.bridge.publish_forecasts(RoleContext("strategy", frozen.namespace, "r", EVAL_ASOF), w.bctx, make_service(w),
                                   PublicationService(tmp_path / "p.sqlite"), ArtifactIndex(tmp_path / "a.sqlite"), frozen.candidate_id,
                                   authorize(w), published_at=EVAL_ASOF + 24 * HOUR)


# ------------------------------------------------------------------ 3. out-of-training calibration, native units


def _design_rows(slope: float, *, noise=0.01, months=14, names=40, seed=3, resolved="2018-12-31T23:59:59+00:00"):
    rng = random.Random(seed)
    return [{"m": 2000 * 12 + m, "s": f"S{i}", "p": (p := rng.uniform(-0.5, 0.5)), "r": slope * p + noise * rng.gauss(0, 1), "x": resolved}
            for m in range(months) for i in range(names)]


def test_calibration_measures_slope_and_residual_from_oos_rows_and_refuses_leaky_or_thin_rows():
    kw = dict(bridge_sign=1, design_end=dt.date(2018, 12, 31), min_months=6)
    planted = calibrate_return_bridge(_design_rows(0.02), **kw)
    assert planted["slope"] == pytest.approx(0.02, abs=0.006) and planted["residual_sd"] == pytest.approx(0.01, abs=0.002)
    assert planted["in_sample_fit"] is False and planted["n_months"] == 14 and planted["slope_se"] > 0
    assert abs(calibrate_return_bridge(_design_rows(0.0), **kw)["slope"]) < 0.01  # no skill -> no conversion magnitude
    flipped = calibrate_return_bridge([{**r, "r": -r["r"]} for r in _design_rows(0.02)], **{**kw, "bridge_sign": -1})
    assert flipped["slope"] == pytest.approx(planted["slope"])  # bridge sign aligns the return with the prediction
    with pytest.raises(CalibrationError, match="leak"):
        calibrate_return_bridge(_design_rows(0.02, resolved="2019-03-31T23:59:59+00:00"), **kw)
    with pytest.raises(CalibrationError, match="usable design months"):
        calibrate_return_bridge(_design_rows(0.02, months=4), **kw)
    with pytest.raises(CalibrationError):
        calibrate_return_bridge([{**r, "r": None} for r in _design_rows(0.02)], **kw)


def test_publication_uses_native_target_units_and_oos_slope_only_for_return(tmp_path, monkeypatch):
    w = build_world(tmp_path)
    frozen, service = w.frozen[0], make_service(w)
    auth = authorize(w)
    assert challenge(w, service, auth=auth).decision.outcome == Outcome.ADMITTED
    publication = PublicationService(tmp_path / "pub.sqlite", registry=Registry(tmp_path / "pub_registry.sqlite"))
    strategy = RoleContext("strategy", frozen.namespace, "run1", EVAL_ASOF)
    at = EVAL_ASOF + dt.timedelta(days=1)
    w.bridge.publish_forecasts(strategy, w.bctx, service, publication, ArtifactIndex(tmp_path / "artifacts.sqlite"), frozen.candidate_id, auth,
                               published_at=at)
    published = publication.export_forecasts(strategy)
    art = json.loads(w.bctx.candidate_path(frozen.spec_hash).read_text(encoding="utf-8"))
    cal = calibrate_return_bridge(art["predictions"], bridge_sign=frozen.bridge_sign, design_end=dt.date.fromisoformat(frozen.design_window[1]),
                                  min_months=6)
    assert cal["slope"] > 0
    assert all(f.target.unit == "centered_rank" and f.return_forecast.unit == "decimal_return" for f in published)
    native = [f.target.prediction for f in published]
    centre = sum(native) / len(native)
    assert all(abs(float(n)) < 1.0 for n in native), "native centred-rank-scale predictions are not rescaled to returns"
    for f in published:  # return forecast = slope x (native prediction - cross-sectional mean); the native value is untouched
        expected = frozen.bridge_sign * cal["slope"] * float(f.target.prediction - centre)
        assert float(f.return_forecast.expected_excess_return) == pytest.approx(expected, rel=1e-6, abs=1e-12)
    assert max(abs(float(f.return_forecast.expected_excess_return)) for f in published) < max(abs(float(n)) for n in native)
    # no measured bridge -> no publication (proxy targets without one stay shadow)
    def no_bridge(*a, **k):
        raise CalibrationError("synthetic: unsupported proxy target")
    monkeypatch.setattr(bridge_module, "calibrate_return_bridge", no_bridge)
    with pytest.raises(PodPlatformError) as exc:
        w.bridge.publish_forecasts(strategy, w.bctx, service, publication, ArtifactIndex(tmp_path / "artifacts2.sqlite"), frozen.candidate_id,
                                   auth, published_at=at)
    assert "no honest return bridge" in summary(exc)


# ------------------------------------------------------------------ 4. restart-safe authority key


def test_authority_key_file_is_private_stable_and_never_exposed(tmp_path):
    key_path = tmp_path / "private" / "check_authority.key"
    first = CheckAuthority.load_or_init(key_path)
    assert oct(key_path.stat().st_mode & 0o777) == "0o600" and (key_path.parent / ".gitignore").read_text().strip() == "*"
    key_hex = key_path.read_text().strip()
    assert key_hex not in repr(first) and key_hex not in str(first.evidence_seal()) and "hidden" in repr(first)
    for secret_holder in (first, first.evidence_seal()):
        with pytest.raises(TypeError):
            pickle.dumps(secret_holder)
    again = CheckAuthority.load_or_init(key_path)
    assert key_path.read_text().strip() == key_hex and again.evidence_seal().fingerprint == first.evidence_seal().fingerprint
    assert CheckAuthority.load_or_init(tmp_path / "other" / "k.key").evidence_seal().fingerprint != first.evidence_seal().fingerprint
    os.chmod(key_path, 0o644)
    with pytest.raises(PodPlatformError) as exc:
        CheckAuthority.load_or_init(key_path)
    assert "mode 0600" in summary(exc)
    os.chmod(key_path, 0o600)
    key_path.write_text("not a key\n")
    with pytest.raises(PodPlatformError) as exc:
        CheckAuthority.load_or_init(key_path)
    assert "malformed" in summary(exc)
    link = tmp_path / "private" / "link.key"
    os.symlink(key_path, link)
    with pytest.raises(PodPlatformError):
        CheckAuthority.load_or_init(link)


def test_restart_verifies_prior_authorization_without_resetting_quota(tmp_path):
    key_path = tmp_path / "private" / "check_authority.key"
    w = build_world(tmp_path, variants=(("v1", {}), ("v2", {"feature_set": "extended"})), authority=CheckAuthority.load_or_init(key_path))
    old, new = w.frozen
    auth = authorize(w, old, budget=1)
    first = challenge(w, make_service(w), old, auth)
    assert first.check_opened and w.bridge.family_accesses(old.mechanism_family) == 1
    w.bridge.close()
    # "restart": new process objects, same private key file and same bridge store
    authority2 = CheckAuthority.load_or_init(key_path)
    bridge2 = ChallengeBridge(tmp_path / "bridge", authority=authority2)
    assert authority2.verify(auth) and not CheckAuthority().verify(auth)
    assert bridge2.family_accesses(old.mechanism_family) == 1, "the persisted quota survives a restart"
    w2 = w
    w2.authority, w2.bridge = authority2, bridge2
    resumed = challenge(w2, make_service(w2, tag="_restart"), old, auth, bridge=bridge2)
    assert resumed.decision.as_dict() == first.decision.as_dict() and bridge2.family_accesses(old.mechanism_family) == 1
    assert len(bridge2.events(old.candidate_id)) >= 2
    with pytest.raises(PodPlatformError) as exc:  # a fresh budget=1 authorization from the restarted authority cannot open a second look
        challenge(w2, make_service(w2, tag="_restart"), new, authorize(w2, new, budget=1, authority=authority2), bridge=bridge2)
    assert "budget" in summary(exc) and bridge2.family_accesses(old.mechanism_family) == 1
    assert key_path.read_text().strip() not in " ".join(d for _, _, d in bridge2.events())
