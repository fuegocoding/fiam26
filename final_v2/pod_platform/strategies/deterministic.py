"""Deterministic strategy candidates, one per charter (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 4.1-4.4).

Every candidate here is conventionally fitted or rule-based. The spec's 4.4
control requires at least one deterministic implementation without a
pretrained language model in the loop; all four of these qualify.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Optional

from .base import Belief, Charter, RankSpreadBridge, SecurityFeatures, StrategyCandidate

D = Decimal

BRIDGE = RankSpreadBridge(
    bridge_id="bridge_rank_spread_v1",
    horizon_trading_days=20,
    slope_bp_per_unit=D("12"),
)


def _scaled(score: float) -> Decimal:
    bounded = max(-1.0, min(1.0, score))
    return D(str(round(bounded, 6)))


def _belief(
    candidate: StrategyCandidate,
    features: SecurityFeatures,
    prediction: float,
    expected_resolution_at: dt.datetime,
) -> Belief:
    return Belief(
        security_id=features.security_id,
        target_id=candidate.target_id,
        prediction=D(str(round(prediction, 6))),
        unit=candidate.unit,
        reference_prediction=D("0"),
        reference_type=candidate.reference_type,
        information_cutoff=features.as_of,
        expected_resolution_at=expected_resolution_at,
    )


class EarningsInnovationCandidate(StrategyCandidate):
    charter = Charter.FUNDAMENTALS_EXPECTATIONS
    candidate_id = "strat1_earnings_innovation_v1"
    mechanism_id = "earnings_innovation_underreaction"
    source_family_ids = ("fundamental_disclosure", "expectations_state")
    target_id = "next_reported_earnings_innovation"
    unit = "decimal_surprise"
    reference_type = "seasonal_random_walk"
    uses_pretrained_model = False
    bridge = BRIDGE

    def score(
        self, features: SecurityFeatures, expected_resolution_at: dt.datetime
    ) -> Optional[Belief]:
        surprise = features.get("earnings_surprise")
        revision = features.get("consensus_revision_3m")
        return _belief(
            self, features, 0.5 * surprise + 0.5 * revision, expected_resolution_at
        )

    def scaled_score(self, belief: Belief) -> Decimal:
        return _scaled(float(belief.prediction))


class CapitalAllocationEventCandidate(StrategyCandidate):
    charter = Charter.CORPORATE_EVENTS
    candidate_id = "strat2_capital_allocation_v1"
    mechanism_id = "capital_allocation_misread"
    source_family_ids = ("corporate_filing_events",)
    target_id = "residual_return_after_event"
    unit = "decimal_return"
    reference_type = "event_indicator_only"
    uses_pretrained_model = False
    bridge = BRIDGE

    def score(
        self, features: SecurityFeatures, expected_resolution_at: dt.datetime
    ) -> Optional[Belief]:
        net_buyback = features.get("net_buyback_intensity")
        issuance = features.get("equity_issuance_intensity")
        return _belief(
            self, features, net_buyback - issuance, expected_resolution_at
        )

    def scaled_score(self, belief: Belief) -> Decimal:
        return _scaled(float(belief.prediction))


class ValuationGapCandidate(StrategyCandidate):
    charter = Charter.VALUATION_RELATIVE_VALUE
    candidate_id = "strat3_valuation_gap_v1"
    mechanism_id = "valuation_vs_business_economics"
    source_family_ids = ("fundamental_disclosure", "market_prices")
    target_id = "residual_medium_horizon_return"
    unit = "decimal_return"
    reference_type = "sector_relative_value"
    uses_pretrained_model = False
    bridge = BRIDGE

    def score(
        self, features: SecurityFeatures, expected_resolution_at: dt.datetime
    ) -> Optional[Belief]:
        gap = features.get("valuation_gap")
        quality = features.get("earnings_quality_score")
        return _belief(self, features, gap + 0.25 * quality, expected_resolution_at)

    def scaled_score(self, belief: Belief) -> Decimal:
        return _scaled(float(belief.prediction))


class LiquidityPressureCandidate(StrategyCandidate):
    charter = Charter.MARKET_DYNAMICS
    candidate_id = "strat4_liquidity_pressure_v1"
    mechanism_id = "temporary_price_pressure_reversal"
    source_family_ids = ("market_prices", "market_liquidity")
    target_id = "twenty_day_residual_return"
    unit = "decimal_return"
    reference_type = "lagged_return_reversal"
    uses_pretrained_model = False
    bridge = BRIDGE

    def score(
        self, features: SecurityFeatures, expected_resolution_at: dt.datetime
    ) -> Optional[Belief]:
        reversal = features.get("reversal_signal")
        volume_shock = features.get("volume_shock")
        return _belief(self, features, reversal + 0.5 * volume_shock, expected_resolution_at)

    def scaled_score(self, belief: Belief) -> Decimal:
        return _scaled(float(belief.prediction))


ALL_CANDIDATES: tuple[type[StrategyCandidate], ...] = (
    EarningsInnovationCandidate,
    CapitalAllocationEventCandidate,
    ValuationGapCandidate,
    LiquidityPressureCandidate,
)
