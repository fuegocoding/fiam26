"""Part 5 acceptance: charter coverage, mechanism IDs, contract-valid forecasts (spec 4, 15.3)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from pod_platform.strategies import (
    ALL_CANDIDATES,
    Belief,
    Charter,
    RankSpreadBridge,
    SecurityFeatures,
)

UTC = dt.timezone.utc
T0 = dt.datetime(2021, 1, 4, 21, 0, tzinfo=UTC)
RESOLUTION = T0 + dt.timedelta(days=45)
EXPIRY = T0 + dt.timedelta(days=25)
D = Decimal


def _features(**values) -> SecurityFeatures:
    return SecurityFeatures(security_id="sec_1", as_of=T0, values=values)


FEATURES = {
    "strat1_earnings_innovation_v1": {
        "earnings_surprise": 0.8,
        "consensus_revision_3m": 0.4,
    },
    "strat2_capital_allocation_v1": {
        "net_buyback_intensity": 0.6,
        "equity_issuance_intensity": 0.1,
    },
    "strat3_valuation_gap_v1": {
        "valuation_gap": 0.3,
        "earnings_quality_score": 0.5,
    },
    "strat4_liquidity_pressure_v1": {
        "reversal_signal": -0.7,
        "volume_shock": 0.2,
    },
}


def test_every_charter_has_exactly_one_candidate():
    charters = [cls.charter for cls in ALL_CANDIDATES]
    assert sorted(c.value for c in charters) == sorted(c.value for c in Charter)
    assert len(set(charters)) == len(charters)


def test_all_candidates_are_non_pretrained_controls():
    assert all(cls.uses_pretrained_model is False for cls in ALL_CANDIDATES)


def test_mechanism_and_source_family_ids_present():
    for cls in ALL_CANDIDATES:
        instance = cls()
        assert instance.mechanism_id
        assert len(instance.source_family_ids) >= 1
        assert instance.target_id
        assert instance.reference_type


def test_sign_conventions():
    long_surplus = ALL_CANDIDATES[0]().score(
        _features(earnings_surprise=1.0, consensus_revision_3m=1.0), RESOLUTION
    )
    short_surplus = ALL_CANDIDATES[0]().score(
        _features(earnings_surprise=-1.0, consensus_revision_3m=-1.0), RESOLUTION
    )
    assert long_surplus.prediction > 0
    assert short_surplus.prediction < 0

    buyback = ALL_CANDIDATES[1]().score(
        _features(net_buyback_intensity=1.0, equity_issuance_intensity=0.0), RESOLUTION
    )
    issuance = ALL_CANDIDATES[1]().score(
        _features(net_buyback_intensity=0.0, equity_issuance_intensity=1.0), RESOLUTION
    )
    assert buyback.prediction > 0
    assert issuance.prediction < 0


def test_scaled_score_is_bounded():
    for cls in ALL_CANDIDATES:
        instance = cls()
        belief = instance.score(_features(**FEATURES[cls.candidate_id]), RESOLUTION)
        scaled = instance.scaled_score(belief)
        assert D("-1") <= scaled <= D("1")


def test_forecast_contract_valid_for_every_candidate():
    for cls in ALL_CANDIDATES:
        instance = cls()
        forecast = instance.forecast(
            _features(**FEATURES[cls.candidate_id]),
            model_id="model_v1",
            admission_certificate_id="cert_v1",
            expected_resolution_at=RESOLUTION,
            expires_at=EXPIRY,
        )
        assert forecast is not None
        assert forecast.sealed().verify_seal()
        assert forecast.mechanism_ids == [instance.mechanism_id]
        assert forecast.return_forecast.unit == "decimal_return"
        assert forecast.return_forecast.horizon_trading_days == 20
        assert forecast.return_forecast.bridge_id == "bridge_rank_spread_v1"
        assert forecast.is_expired(T0) is False


def test_bridge_sign_follows_belief_sign():
    bridge = RankSpreadBridge(
        bridge_id="b1", horizon_trading_days=20, slope_bp_per_unit=D("12")
    )
    belief = Belief(
        security_id="sec_1",
        target_id="t1",
        prediction=D("0.5"),
        unit="decimal_return",
        reference_prediction=D("0"),
        reference_type="baseline",
        information_cutoff=T0,
        expected_resolution_at=RESOLUTION,
    )
    positive = bridge.expected_excess_return(belief, D("1"))
    negative = bridge.expected_excess_return(belief, D("-1"))
    assert positive > 0 > negative
    assert positive == D("0.001200")


def test_bridge_rejects_unbounded_slope_units():
    with pytest.raises(Exception):
        RankSpreadBridge(
            bridge_id="b1", horizon_trading_days=20, slope_bp_per_unit=D("12"), unit="percent"
        )


def test_belief_rejects_resolution_before_cutoff():
    with pytest.raises(ValueError):
        Belief(
            security_id="sec_1",
            target_id="t1",
            prediction=D("0.5"),
            unit="decimal_return",
            reference_prediction=D("0"),
            reference_type="baseline",
            information_cutoff=T0,
            expected_resolution_at=T0,
        )


def test_missing_feature_declared():
    with pytest.raises(KeyError):
        ALL_CANDIDATES[0]().score(_features(earnings_surprise=0.5), RESOLUTION)
