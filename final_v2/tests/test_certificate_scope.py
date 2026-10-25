"""Machine-readable certificate scope, validation and controlled-replay wiring.

Offline only. Certificates here are test fixtures: nothing in this file claims
that a fixture certificate authorizes a real competition book.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from pod_platform.contracts.errors import ErrorCode, PodPlatformError
from pod_platform.orchestration.replay import ControlledReplay, load_fixture, utc_close
from pod_platform.research.certificates import (
    CertificateEnvironment,
    CertificatePolicy,
    CertificateRevocation,
    CertificateScope,
    RevocationLedger,
    ScopeLimits,
    check_scope_limits,
    resolve_policy,
    validate_certificate,
)
from pod_platform.research.promotion import AdmissionCertificate, PromotionCriteria, PromotionOutcome

from .test_replay import ENDS, FORMATIONS, T0, _certificate, _forecast, code_of, make_fixture

UTC = dt.timezone.utc
VALID_UNTIL = dt.datetime(2025, 1, 1, tzinfo=UTC)
SNAPSHOT = "snap_src_signal_2024_01"  # the snapshot id make_fixture's source card carries
SNAPS = {"src_signal": SNAPSHOT}
FORECAST = _forecast(FORMATIONS[0], ENDS[0], "AAA")
DECISION = utc_close(FORMATIONS[0])


def scope(**over) -> CertificateScope:
    base = dict(
        environment=CertificateEnvironment.FORMATION,
        valid_until=VALID_UNTIL,
        allowed_source_ids=["src_signal"],
        allowed_source_snapshots={"src_signal": [SNAPSHOT]},
    )
    base.update(over)
    return CertificateScope(**base)


def cert(sc="default", **over) -> AdmissionCertificate:
    base = dict(
        owner="integrity",
        created_at=T0,
        certificate_id="cert_a",
        model_id="model_a",
        model_hash=hashlib.sha256(b"model_a").hexdigest(),
        feature_hashes={"f": "h"},
        data_hashes={"d": "h"},
        approved_universe="test_universe",
        target_id="t1",
        horizon_trading_days=21,
        effective_start=T0,
        approved_mode="controlled_replay",
        expiry_policy="annual",
        promotion_evidence_ids=["ev_promotion_1"],
        scope=scope() if sc == "default" else sc,
    )
    base.update(over)
    return AdmissionCertificate(**base).sealed()


def check(c=None, **kw):
    kw.setdefault("decision_at", DECISION)
    kw.setdefault("source_snapshots", SNAPS)
    kw.setdefault("policy", "production")
    return validate_certificate(FORECAST, c if c is not None else cert(), **kw)


def rejects(match: str, c=None, **kw) -> str:
    with pytest.raises(PodPlatformError) as err:
        check(c, **kw)
    assert code_of(err) == ErrorCode.ADMISSION_CERTIFICATE_INVALID
    assert match in str(err.value), str(err.value)
    return str(err.value)


# ------------------------------------------------------------------ scope model


def test_production_accepts_a_complete_formation_certificate():
    result = check()
    assert result.authoritative and result.environment == "formation" and not result.legacy_unverified
    assert result.policy == "production" and result.as_dict()["authoritative"] is True


@pytest.mark.parametrize(
    "bad",
    [
        dict(valid_until=dt.datetime(2025, 1, 1)),  # naive
        dict(valid_from=VALID_UNTIL, valid_until=VALID_UNTIL),  # empty window
        dict(allowed_source_ids=[]),
        dict(allowed_source_ids=["a", "a"]),
        dict(allowed_source_snapshots={"other": ["s"]}),  # pin outside allowed ids
        dict(allowed_source_snapshots={}),
        dict(universe_ids=[""]),
    ],
)
def test_scope_rejects_malformed_values(bad):
    with pytest.raises(ValueError):
        scope(**bad)
    with pytest.raises(ValueError):
        scope(limits=ScopeLimits(max_gross_exposure=Decimal("0")))


def test_limits_reject_non_finite_negative_and_unknown_fields():
    for bad in ({"max_nav_usd": Decimal("NaN")}, {"max_abs_net_exposure": Decimal("-1")}, {"max_position_count": 0}):
        with pytest.raises(ValueError):
            ScopeLimits(**bad)
    with pytest.raises(ValidationError):
        ScopeLimits(max_leverage=3)  # extra="forbid"
    assert ScopeLimits(max_abs_net_exposure=Decimal("0")).declared() == {"max_abs_net_exposure": Decimal("0")}


def test_scope_is_part_of_the_seal_and_survives_json():
    sealed = cert()
    assert sealed.verify_seal()
    again = AdmissionCertificate.model_validate(sealed.model_dump(mode="json"))
    assert again.verify_seal() and again.scope == sealed.scope
    stretched = sealed.model_copy(update={"scope": scope(valid_until=dt.datetime(2030, 1, 1, tzinfo=UTC))})
    assert not stretched.verify_seal()  # extending expiry after sealing is detected
    assert cert(scope(environment=CertificateEnvironment.TEST)).content_hash != sealed.content_hash


def test_certificates_sealed_before_scope_existed_still_verify():
    legacy = _certificate()
    payload = legacy.model_dump(mode="json")
    assert payload["scope"] is None
    old = {k: v for k, v in payload.items() if k != "scope"}  # what an old file contains
    loaded = AdmissionCertificate.model_validate(old)
    assert loaded.scope is None and loaded.verify_seal() and loaded.content_hash == legacy.content_hash


def test_certificate_window_and_is_valid_at():
    sealed = cert(scope(valid_from=T0 + dt.timedelta(days=10)))
    assert not sealed.is_valid_at(T0 + dt.timedelta(days=9))
    assert sealed.is_valid_at(T0 + dt.timedelta(days=10))
    assert sealed.is_valid_at(VALID_UNTIL - dt.timedelta(seconds=1)) and not sealed.is_valid_at(VALID_UNTIL)
    with pytest.raises(ValueError):  # expiry before the certificate can start
        cert(scope(valid_until=T0 - dt.timedelta(days=1)), effective_start=T0)


def test_promotion_issues_sealed_certificate_with_scope():
    criteria = PromotionCriteria(
        min_coverage="c", improvement_vs_primary_baseline="i", direction_and_economic_size="d",
        stability_requirement="s", return_translation_requirement="r", implementation_requirement="m",
        permitted_statistical_uncertainty="u", multiple_testing_treatment="t",
    )
    outcome = PromotionOutcome(
        registration_id="reg_9", evaluated_at=T0 + dt.timedelta(days=1), criteria=criteria, coverage_met=True,
        improvement_met=True, direction_met=True, stability_met=True, implementation_met=True,
        uncertainty_met=True, multiple_testing_met=True,
    )
    issued = outcome.certificate_or_raise(
        "model_a", hashlib.sha256(b"m").hexdigest(), feature_hashes={"f": "h"}, data_hashes={"d": "h"},
        approved_universe="u", target_id="t1", horizon_trading_days=21, effective_start=T0 + dt.timedelta(days=1),
        approved_mode="formation", expiry_policy="annual", promotion_evidence_ids=["ev"],
        scope=scope(universe_ids=["u"]),
    )
    assert issued.verify_seal() and issued.scope.environment == CertificateEnvironment.FORMATION


# ------------------------------------------------------- legacy / free-text only


def test_legacy_free_text_certificate_fails_closed_in_production():
    legacy = _certificate()
    message = rejects("free-text", legacy, source_snapshots=None)
    assert "machine-readable scope" in message


def test_fixture_policy_accepts_legacy_but_labels_it_unverified():
    result = check(_certificate(), policy="fixture_compat", source_snapshots=None)
    assert result.legacy_unverified and not result.authoritative and result.environment is None
    assert any("legacy_unverified" in w for w in result.warnings)
    other_mode = _certificate().model_copy(update={"approved_mode": "formation"})
    other_mode = other_mode.sealed()
    rejects("not accepted by policy", other_mode, policy="fixture_compat", source_snapshots=None)


def test_unknown_policy_name_is_rejected():
    with pytest.raises(PodPlatformError) as err:
        resolve_policy("lenient")
    assert code_of(err) == ErrorCode.ADMISSION_CERTIFICATE_INVALID
    assert resolve_policy(None).name == "fixture_compat" and resolve_policy(None, default="production").name == "production"


# ------------------------------------------------------------ environment


@pytest.mark.parametrize("env", [CertificateEnvironment.TEST, CertificateEnvironment.SHADOW])
def test_test_and_shadow_certificates_never_authorize_formation(env):
    sealed = cert(scope(environment=env))
    for policy in ("production", "fixture_compat"):
        rejects("cannot authorize a formation run", sealed, policy=policy)
    assert check(sealed, environment=CertificateEnvironment.TEST).environment == env.value
    assert not check(sealed, environment=CertificateEnvironment.TEST).authoritative


def test_shadow_policy_accepts_shadow_but_not_test():
    check(cert(scope(environment=CertificateEnvironment.SHADOW)), policy="shadow")
    rejects("cannot authorize a shadow run", cert(scope(environment=CertificateEnvironment.TEST)), policy="shadow")
    check(cert(), policy="shadow")  # a formation certificate may run in shadow


# ------------------------------------------------------------ time


def test_expiry_not_yet_valid_and_naive_clock():
    sealed = cert(scope(valid_until=DECISION + dt.timedelta(days=1)))
    check(sealed, decision_at=DECISION)
    rejects("expired", sealed, decision_at=DECISION + dt.timedelta(days=1))  # exactly at valid_until
    rejects("not yet valid", cert(scope(valid_from=DECISION + dt.timedelta(days=1))))
    rejects("timezone-aware", decision_at=dt.datetime(2024, 1, 31, 21))
    rejects("no valid_until", cert(scope(valid_until=None)))
    no_expiry = check(cert(scope(valid_until=None)), policy=CertificatePolicy.production().replace(require_expiry=False))
    assert any("does not expire" in w for w in no_expiry.warnings)


def test_back_dated_and_post_decision_certificates_are_rejected():
    late = cert(created_at=DECISION + dt.timedelta(days=1))  # issued after the decision, claims an old start
    rejects("after the decision", late)
    rejects("after the decision", late, policy="fixture_compat")
    backdated = cert(created_at=T0 + dt.timedelta(days=30), effective_start=T0)  # issued later, claims an earlier start
    rejects("back-dated", backdated)
    check(backdated, policy="fixture_compat")  # fixture policy keeps the historic leniency, scope still checked
    published_earlier = cert(created_at=FORECAST.created_at + dt.timedelta(hours=1), effective_start=FORECAST.created_at + dt.timedelta(hours=1))
    rejects("published before certificate", published_earlier, decision_at=DECISION)


def test_revocation_is_effective_from_its_time():
    ledger = RevocationLedger(
        [
            CertificateRevocation(
                certificate_id="cert_a", revoked_at=DECISION + dt.timedelta(days=1),
                recorded_at=DECISION + dt.timedelta(days=2), reason="data incident", authority="integrity",
            )
        ]
    )
    check(revocations=ledger)  # decision precedes the revocation
    rejects("revoked", revocations=ledger, decision_at=DECISION + dt.timedelta(days=1))
    assert len(ledger) == 1 and ledger.effective("cert_other", DECISION + dt.timedelta(days=9)) is None
    with pytest.raises(ValueError):
        CertificateRevocation(certificate_id="c", revoked_at=dt.datetime(2024, 1, 1), recorded_at=DECISION, reason="r", authority="a")


# ------------------------------------------------- target / horizon / model / seal


def test_target_horizon_model_and_seal_checks():
    rejects("target", cert(target_id="other"))
    rejects("21d", cert(horizon_trading_days=5))
    rejects("does not cover", cert(model_id="model_b"))
    rejects("unsealed", cert().model_copy(update={"expiry_policy": "never"}))
    with pytest.raises(PodPlatformError) as err:
        validate_certificate(FORECAST, None, decision_at=DECISION, policy="production")
    assert "unknown certificate" in str(err.value)
    with pytest.raises(PodPlatformError):
        validate_certificate(FORECAST, cert(certificate_id="cert_zzz"), decision_at=DECISION, source_snapshots=SNAPS)


# -------------------------------------------------- sources, universe, namespace


def test_certificate_can_only_narrow_source_authority():
    rejects("does not authorize source", cert(scope(allowed_source_ids=["src_other"], allowed_source_snapshots={"src_other": ["s"]})))
    rejects("does not authorize source", source_ids=["src_extra"])  # an extra consumed source
    rejects("does not restrict source ids", cert(scope(allowed_source_ids=None, allowed_source_snapshots=None)))
    open_ids = cert(scope(allowed_source_ids=None, allowed_source_snapshots=None))
    result = check(open_ids, policy="fixture_compat")
    assert any("does not restrict sources" in w for w in result.warnings)


def test_snapshot_pins_are_enforced_and_must_be_supplied():
    rejects("not pinned", source_snapshots={"src_signal": "snap_other"})
    rejects("was not supplied", source_snapshots={})
    rejects("does not pin source snapshots", cert(scope(allowed_source_snapshots=None)))
    rejects("no snapshot pin for source", cert(scope(allowed_source_ids=["src_signal", "src_x"], allowed_source_snapshots={"src_x": ["s"]})))


def test_universe_and_namespace_scope_fail_closed_when_unsupplied():
    sealed = cert(scope(universe_ids=["us_equities"], namespaces=["pod_valuation"]))
    check(sealed, universe="us_equities", namespace="pod_valuation")
    rejects("universe", sealed, universe="eu_equities", namespace="pod_valuation")
    rejects("universe", sealed, namespace="pod_valuation")  # restriction declared, context absent
    rejects("namespace", sealed, universe="us_equities", namespace="pod_events")
    rejects("namespace", sealed, universe="us_equities")
    rejects("approved_universe", universe="something_else")  # falls back to the legacy approved_universe


# --------------------------------------------------------------------- limits


LIMITS = ScopeLimits(
    max_gross_exposure=Decimal("1.5"), max_abs_net_exposure=Decimal("0.2"), max_name_weight=Decimal("0.1"),
    max_nav_usd=Decimal("2000000"), max_position_count=20,
)
OK_EXPOSURE = {"gross": 1.0, "net": -0.05, "max_name_weight": 0.08, "nav_usd": 1_000_000, "position_count": 12}


def test_numeric_limits_are_checked_against_measured_exposure():
    sealed = cert(scope(limits=LIMITS))
    check_scope_limits(sealed, OK_EXPOSURE)
    assert check(sealed, exposures=OK_EXPOSURE).authoritative
    for key, value, limit_name in (
        ("gross", 1.6, "max_gross_exposure"),
        ("net", -0.3, "max_abs_net_exposure"),
        ("max_name_weight", 0.2, "max_name_weight"),
        ("nav_usd", 3_000_000, "max_nav_usd"),
        ("position_count", 25, "max_position_count"),
    ):
        rejects(limit_name, sealed, exposures={**OK_EXPOSURE, key: value})


def test_declared_limit_without_measurement_or_with_nan_fails_closed():
    sealed = cert(scope(limits=LIMITS))
    missing = {k: v for k, v in OK_EXPOSURE.items() if k != "nav_usd"}
    rejects("was not measured", sealed, exposures=missing)
    rejects("not finite", sealed, exposures={**OK_EXPOSURE, "gross": float("nan")})
    unevaluated = check(sealed)  # no exposures given: warns instead of silently passing
    assert any("limits_not_evaluated" in w for w in unevaluated.warnings)
    check_scope_limits(_certificate(), {})  # legacy certificate: nothing machine-checkable


# ------------------------------------------------------ controlled replay wiring


def _fixture_with(tmp_path, name, certificate=None, config=None):
    root = make_fixture(tmp_path / name)
    if certificate is not None:
        (root / "certificates.json").write_text(json.dumps([certificate.model_dump(mode="json")]))
    if config:
        path = root / "config.json"
        doc = json.loads(path.read_text())
        doc.update(config)
        path.write_text(json.dumps(doc))
    return root


def _replay(root, out, **kw):
    return ControlledReplay(load_fixture(root), out, **kw)


def test_replay_default_policy_labels_legacy_certificates_unverified(tmp_path):
    outcome = _replay(_fixture_with(tmp_path, "fx"), tmp_path / "out").run()
    summary = outcome["summary"]
    assert summary["freeze_status"] == "FROZEN"  # baseline fixture behaviour is preserved
    assert summary["certificate_policy"] == "fixture_compat"
    checks = summary["certificate_checks"]
    assert checks and all(c["legacy_unverified"] and not c["authoritative"] for c in checks)
    assert summary["certificates_authorize_competition_book"] is False


def test_replay_production_policy_with_scoped_certificate(tmp_path):
    root = _fixture_with(tmp_path, "fx", cert())
    outcome = _replay(root, tmp_path / "out", certificate_policy="production").run()
    summary = outcome["summary"]
    assert summary["freeze_status"] == "FROZEN", summary["freeze_reasons"]
    assert summary["certificate_policy"] == "production"
    assert [c["authoritative"] for c in summary["certificate_checks"]] == [True]
    assert summary["certificates_authorize_competition_book"] is False


def test_replay_production_rejects_legacy_certificate_and_writes_failure(tmp_path):
    root = _fixture_with(tmp_path, "fx")
    with pytest.raises(PodPlatformError) as err:
        _replay(root, tmp_path / "out", certificate_policy="production").run()
    assert code_of(err) == ErrorCode.ADMISSION_CERTIFICATE_INVALID and "free-text" in str(err.value)
    failure = json.loads((tmp_path / "out" / "failure.json").read_text())
    assert failure["error_code"] == "ADMISSION_CERTIFICATE_INVALID" and failure["frozen"] is False


def test_replay_policy_can_come_from_config_and_unknown_policy_fails(tmp_path):
    root = _fixture_with(tmp_path, "fx_cfg", config={"certificate_policy": "production"})
    with pytest.raises(PodPlatformError):
        _replay(root, tmp_path / "o1").run()
    root = _fixture_with(tmp_path, "fx_bad", config={"certificate_policy": "anything_goes"})
    with pytest.raises(PodPlatformError) as err:
        _replay(root, tmp_path / "o2")
    assert code_of(err) == ErrorCode.ADMISSION_CERTIFICATE_INVALID


@pytest.mark.parametrize("env", [CertificateEnvironment.TEST, CertificateEnvironment.SHADOW])
def test_test_or_shadow_certificate_cannot_freeze_a_formation_run(tmp_path, env):
    root = _fixture_with(tmp_path, "fx", cert(scope(environment=env)))
    with pytest.raises(PodPlatformError) as err:
        _replay(root, tmp_path / "out").run()
    assert code_of(err) == ErrorCode.ADMISSION_CERTIFICATE_INVALID and "cannot authorize a formation run" in str(err.value)
    # an explicitly labelled research-only run may use it, and it can never freeze
    outcome = _replay(root, tmp_path / "out_research", research_only=True).run()
    assert outcome["summary"]["freeze_status"] == "NOT_FROZEN"
    assert outcome["summary"]["source_policy"]["run_label"] == "RESEARCH_ONLY_NOT_FREEZABLE"


def test_replay_stops_when_certificate_expires_mid_run(tmp_path):
    expiring = cert(scope(valid_until=dt.datetime(2024, 3, 1, tzinfo=UTC)))
    root = _fixture_with(tmp_path, "fx", expiring)
    with pytest.raises(PodPlatformError) as err:
        _replay(root, tmp_path / "out", certificate_policy="production").run()
    assert "expired" in str(err.value)
    failure = json.loads((tmp_path / "out" / "failure.json").read_text())
    assert failure["periods_completed"] == ["2024-01-31", "2024-02-29"]


def test_replay_honours_revocation_ledger(tmp_path):
    root = _fixture_with(tmp_path, "fx", cert())
    ledger = RevocationLedger(
        [
            CertificateRevocation(
                certificate_id="cert_a", revoked_at=dt.datetime(2024, 2, 15, tzinfo=UTC),
                recorded_at=dt.datetime(2024, 2, 16, tzinfo=UTC), reason="model incident", authority="integrity",
            )
        ]
    )
    with pytest.raises(PodPlatformError) as err:
        _replay(root, tmp_path / "out", certificate_policy="production", revocations=ledger).run()
    assert "revoked" in str(err.value)
    failure = json.loads((tmp_path / "out" / "failure.json").read_text())
    assert failure["periods_completed"] == ["2024-01-31"]


def test_replay_enforces_source_and_snapshot_restrictions_of_the_certificate(tmp_path):
    narrow = cert(scope(allowed_source_ids=["src_prices"], allowed_source_snapshots={"src_prices": ["s"]}))
    with pytest.raises(PodPlatformError) as err:
        _replay(_fixture_with(tmp_path, "fx1", narrow), tmp_path / "o1", certificate_policy="production").run()
    assert "does not authorize source" in str(err.value)
    wrong_snapshot = cert(scope(allowed_source_snapshots={"src_signal": ["snap_from_another_vintage"]}))
    with pytest.raises(PodPlatformError) as err:
        _replay(_fixture_with(tmp_path, "fx2", wrong_snapshot), tmp_path / "o2", certificate_policy="production").run()
    assert "not pinned" in str(err.value)


def test_replay_enforces_certificate_exposure_limits_on_the_book(tmp_path):
    tight = cert(scope(limits=ScopeLimits(max_gross_exposure=Decimal("0.5"))))
    with pytest.raises(PodPlatformError) as err:
        _replay(_fixture_with(tmp_path, "fx", tight), tmp_path / "out", certificate_policy="production").run()
    assert code_of(err) == ErrorCode.ADMISSION_CERTIFICATE_INVALID and "max_gross_exposure" in str(err.value)
    roomy = cert(scope(limits=ScopeLimits(max_gross_exposure=Decimal("2.0"), max_nav_usd=Decimal("10000000"))))
    outcome = _replay(_fixture_with(tmp_path, "fx_ok", roomy), tmp_path / "out_ok", certificate_policy="production").run()
    assert outcome["summary"]["freeze_status"] == "FROZEN"


def test_replay_universe_scope_uses_config_universe(tmp_path):
    sealed = cert(scope(universe_ids=["us_equities"]))
    with pytest.raises(PodPlatformError) as err:  # config declares no universe: restriction cannot be evaluated
        _replay(_fixture_with(tmp_path, "fx1", sealed), tmp_path / "o1", certificate_policy="production").run()
    assert "universe" in str(err.value)
    outcome = _replay(
        _fixture_with(tmp_path, "fx2", sealed, config={"universe_id": "us_equities"}), tmp_path / "o2",
        certificate_policy="production",
    ).run()
    assert outcome["summary"]["freeze_status"] == "FROZEN"
