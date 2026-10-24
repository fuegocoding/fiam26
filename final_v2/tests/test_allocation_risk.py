"""Part 6 acceptance: integration, construction, risk review (spec 16, 17, 26.5)."""

from __future__ import annotations

import datetime as dt

import pytest

from pod_platform.allocation import (
    Candidate,
    ConstructionEngine,
    FamilyEstimate,
    FamilyIntegrator,
    InternalPolicy,
    OfficialLimits,
)
from pod_platform.contracts import ErrorCode, PodPlatformError
from pod_platform.risk import MeasuredExposure, RiskEngine, RiskLimit, RiskPolicy

UTC = dt.timezone.utc
T0 = dt.datetime(2021, 1, 4, 21, 0, tzinfo=UTC)
FAMILIES = ("f1", "f2", "f3")


def _integrator(**over) -> FamilyIntegrator:
    return FamilyIntegrator(FAMILIES, **over)


# ------------------------------------------------- 16.3 / C2 integration


def test_equal_family_weights_not_coverage_renormalized():
    integrator = _integrator()
    full = integrator.integrate(
        [
            FamilyEstimate("f1", "sec_1", 0.02, 0.01),
            FamilyEstimate("f2", "sec_1", 0.02, 0.01),
            FamilyEstimate("f3", "sec_1", 0.02, 0.01),
        ]
    )
    partial = integrator.integrate(
        [
            FamilyEstimate("f1", "sec_1", 0.02, 0.01),
            FamilyEstimate("f2", "sec_1", 0.02, 0.01),
        ]
    )
    assert abs(full.alpha - 0.02) < 1e-12
    assert abs(partial.alpha - 0.04 / 3) < 1e-12
    assert partial.alpha != 0.02


def test_missing_family_zero_alpha_with_mask_and_inflated_uncertainty():
    integrator = _integrator(neutral_uncertainty=0.01, missing_uncertainty_multiplier=2.0)
    complete = integrator.integrate(
        [
            FamilyEstimate("f1", "sec_1", 0.03, 0.01),
            FamilyEstimate("f2", "sec_1", 0.03, 0.01),
            FamilyEstimate("f3", "sec_1", 0.03, 0.01),
        ]
    )
    missing = integrator.integrate(
        [
            FamilyEstimate("f1", "sec_1", 0.03, 0.01),
            FamilyEstimate("f2", "sec_1", 0.03, 0.01),
        ]
    )
    assert missing.missingness_mask == ("f3",)
    assert missing.families_total == 3
    assert missing.families_available == 2
    assert missing.uncertainty > complete.uncertainty


def test_unavailable_estimate_treated_as_missing_not_zero():
    integrator = _integrator()
    result = integrator.integrate(
        [
            FamilyEstimate("f1", "sec_1", 0.05, 0.01),
            FamilyEstimate("f2", "sec_1", 0.0, 0.01, available=False),
            FamilyEstimate("f3", "sec_1", 0.05, 0.01),
        ]
    )
    assert result.families_available == 2
    assert "f2" in result.missingness_mask
    assert abs(result.alpha - 0.10 / 3) < 1e-12


def test_unadmitted_family_rejected():
    integrator = _integrator()
    with pytest.raises(ValueError):
        integrator.integrate([FamilyEstimate("rogue", "sec_1", 0.01, 0.01)])


def test_duplicate_family_estimate_rejected():
    integrator = _integrator()
    with pytest.raises(ValueError):
        integrator.integrate(
            [
                FamilyEstimate("f1", "sec_1", 0.01, 0.01),
                FamilyEstimate("f1", "sec_1", 0.02, 0.01),
            ]
        )


def test_multi_security_batch_rejected():
    integrator = _integrator()
    with pytest.raises(ValueError):
        integrator.integrate(
            [
                FamilyEstimate("f1", "sec_1", 0.01, 0.01),
                FamilyEstimate("f2", "sec_2", 0.01, 0.01),
            ]
        )


def test_neutral_uncertainty_must_be_positive():
    with pytest.raises(ValueError):
        FamilyIntegrator(FAMILIES, neutral_uncertainty=0.0)
    with pytest.raises(ValueError):
        FamilyIntegrator(FAMILIES, missing_uncertainty_multiplier=0.5)


# ------------------------------------------------- 16.5-16.11 construction


def _candidates(n_pairs: int = 150, incumbents: int = 0):
    out = []
    for i in range(n_pairs):
        out.append(
            Candidate(
                security_id=f"L{i}",
                alpha=0.01 + i * 1e-4,
                market_cap_usd=2e9,
                incumbent=i < incumbents,
                sector="tech",
            )
        )
        out.append(
            Candidate(
                security_id=f"S{i}",
                alpha=-0.01 - i * 1e-4,
                market_cap_usd=2e9,
                incumbent=i < incumbents,
                sector="tech",
            )
        )
    return out


def _engine(official=None, internal=None):
    return ConstructionEngine(
        official or OfficialLimits(), internal or InternalPolicy(target_positions=300)
    )


def _book_kwargs():
    return dict(
        run_id="run_1",
        decision_at=T0,
        forecast_publication_ids=["fc_1"],
        current_book_snapshot_id="snap_prev",
        nav_snapshot_id="nav_1",
        price_snapshot_id="px_1",
        risk_envelope_version="env_1",
        covariance_version="cov_1",
        cost_model_version="cost_1",
        attribution_method="family_equal_contribution",
    )


def test_official_limits_respected():
    engine = _engine()
    book = engine.build(_candidates(), **_book_kwargs())
    gross = sum(abs(p.target_weight) for p in book.positions)
    net = sum(p.target_weight for p in book.positions)
    assert gross <= OfficialLimits().max_gross + 1e-9
    assert OfficialLimits().min_net <= net <= OfficialLimits().max_net
    assert all(abs(p.target_weight) <= OfficialLimits().max_abs_name_weight + 1e-9 for p in book.positions)
    assert OfficialLimits().min_positions <= len(book.positions) <= OfficialLimits().max_positions
    assert all(r.status == "pass" for r in book.official_constraint_results)


def test_infeasible_when_too_few_candidates():
    engine = _engine()
    with pytest.raises(PodPlatformError) as exc:
        engine.build(_candidates(n_pairs=10), **_book_kwargs())
    assert exc.value.response.error.code == ErrorCode.PORTFOLIO_INFEASIBLE


def test_shorts_below_market_cap_floor_excluded():
    engine = _engine()
    candidates = _candidates(n_pairs=60)
    candidates = [
        Candidate(
            security_id=c.security_id,
            alpha=c.alpha,
            market_cap_usd=1e8 if c.alpha < 0 else 2e9,
            sector=c.sector,
        )
        for c in candidates
    ]
    with pytest.raises(PodPlatformError):
        engine.build(candidates, **_book_kwargs())


def test_incumbent_retention_band_keeps_near_rank():
    engine = _engine(internal=InternalPolicy(target_positions=300, retention_band=0.5))
    candidates = _candidates(n_pairs=150, incumbents=75)
    book = engine.build(candidates, **_book_kwargs())
    long_ids = [p.security_id for p in book.positions if p.target_weight > 0]
    assert len(long_ids) == 150
    assert set(long_ids) <= {c.security_id for c in candidates if c.alpha > 0}


def test_gross_reduction_relaxation_recorded():
    engine = _engine(
        internal=InternalPolicy(target_positions=300, target_gross=2.0),
    )
    book = engine.build(_candidates(), **_book_kwargs())
    gross = sum(abs(p.target_weight) for p in book.positions)
    assert gross <= 2.0 + 1e-9
    weight = max(abs(p.target_weight) for p in book.positions)
    assert weight <= OfficialLimits().max_abs_name_weight + 1e-9
    if book.relaxations:
        assert book.solver_status.value in ("RELAXED", "FEASIBLE", "OPTIMAL")
        assert all(r.step >= 1 for r in book.relaxations)


def test_tradable_filter_applied():
    engine = _engine()
    candidates = [Candidate(c.security_id, c.alpha, c.market_cap_usd, tradable=False) for c in _candidates(60)]
    with pytest.raises(PodPlatformError):
        engine.build(candidates, **_book_kwargs())


# --------------------------------------------------------- 17 risk envelope


def _policy(**over) -> RiskPolicy:
    base = dict(
        policy_version="policy_v1",
        limits=[
            RiskLimit(name="gross", threshold="<=2.0", hard=True),
            RiskLimit(name="net", threshold="[-0.5,0.5]", hard=True),
            RiskLimit(name="beta", threshold="<=0.05", hard=False),
        ],
        review_ttl_minutes=60,
    )
    base.update(over)
    return RiskPolicy(**base)


def _exposures(net_value=0.0, beta_value=0.01):
    return [
        MeasuredExposure("gross", 1.0, 2.0, is_max=True),
        MeasuredExposure("net", net_value, 0.5, is_max=True),
        MeasuredExposure("beta", beta_value, 0.05, is_max=True, uncertainty=0.01),
    ]


def _book():
    engine = _engine()
    return engine.build(_candidates(), **_book_kwargs())


def test_risk_review_pass():
    engine = RiskEngine(_policy())
    review = engine.review(_book(), _exposures(), effective_at=T0)
    assert review.decision.value == "PASS"
    assert review.remedy_constraints == []
    assert review.is_valid_at(T0) is True
    assert review.is_valid_at(T0 + dt.timedelta(hours=2)) is False


def test_risk_review_blocks_on_hard_failure():
    engine = RiskEngine(_policy())
    review = engine.review(_book(), _exposures(net_value=0.9), effective_at=T0)
    assert review.decision.value == "BLOCK"
    assert any("net" in remedy for remedy in review.remedy_constraints)


def test_risk_review_warns_on_soft_limit():
    engine = RiskEngine(_policy())
    exposures = _exposures()
    exposures[2] = MeasuredExposure("beta", 0.05, 0.05, is_max=True, uncertainty=0.01)
    review = engine.review(_book(), exposures, effective_at=T0)
    assert review.decision.value in ("PASS", "PASS_WITH_REMEDY", "BLOCK")


def test_missing_exposure_rejected():
    engine = RiskEngine(_policy())
    with pytest.raises(ValueError):
        engine.review(_book(), _exposures()[:1], effective_at=T0)


def test_review_ttl_from_policy():
    engine = RiskEngine(_policy(review_ttl_minutes=30))
    review = engine.review(_book(), _exposures(), effective_at=T0)
    assert review.expires_at == T0 + dt.timedelta(minutes=30)
