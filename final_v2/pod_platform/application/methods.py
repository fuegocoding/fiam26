"""Bounded reusable-methods registry (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 10.6, 6.2 ``methods.search/describe``).

Only methods that exist in this code base are registered; each card names its
reference implementation as ``module:attribute`` and tests verify that the path
imports and resolves. Cards describe inputs, assumptions, baselines and failure
modes. Registering a method does not authorize its use, and nothing here executes
agent-supplied code (a method id selects a reviewed implementation; it never
selects a file or a function from user input).
"""

from __future__ import annotations

import importlib
from typing import Optional, Sequence

from pydantic import Field

from ..contracts.base import StrictModel
from ..contracts.errors import ErrorCode
from .common import fail


class MethodCard(StrictModel):
    method_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    version: str = Field(pattern=r"^[0-9A-Za-z.\-]{1,32}$")
    summary: str = Field(min_length=10)
    supported_targets: list[str] = Field(min_length=1)
    data_shapes: list[str] = Field(min_length=1)
    min_history: str = Field(min_length=1)
    assumptions: list[str] = Field(min_length=1)
    interfaces: list[str] = Field(min_length=1)
    hyperparameters: dict[str, str] = Field(default_factory=dict)
    compute: str = Field(min_length=1)
    reference_implementation: str = Field(
        pattern=r"^[A-Za-z0-9_.]+:[A-Za-z0-9_]+$"
    )
    tests: list[str] = Field(min_length=1)
    baselines: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(min_length=1)
    spec_sections: list[str] = Field(default_factory=list)
    uses_pretrained_model: bool = False
    optional_dependencies: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.method_id}@{self.version}"


class MethodsRegistry:
    def __init__(self, cards: Sequence[MethodCard] = ()):
        self._cards: dict[str, MethodCard] = {}
        for card in cards:
            self.register(card)

    def register(self, card: MethodCard) -> str:
        if card.key in self._cards:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"method card {card.key} already registered",
                "Method cards are immutable; publish a new version.",
            )
        self._cards[card.key] = card
        return card.key

    def list(self) -> list[MethodCard]:
        return [self._cards[k] for k in sorted(self._cards)]

    def describe(
        self, method_id: str, version: Optional[str] = None
    ) -> MethodCard:
        matches = [
            c
            for c in self._cards.values()
            if c.method_id == method_id
            and (version is None or c.version == version)
        ]
        if not matches:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"unknown method {method_id!r}",
                "Only registered methods can be described.",
            )
        return sorted(matches, key=lambda c: c.version)[-1]

    def search(
        self,
        *,
        target_type: Optional[str] = None,
        data_shape: Optional[str] = None,
        text: Optional[str] = None,
    ) -> list[MethodCard]:
        needle = (text or "").lower().strip()
        out: list[MethodCard] = []
        for card in self.list():
            if target_type and target_type not in card.supported_targets:
                continue
            if data_shape and data_shape not in card.data_shapes:
                continue
            if (
                needle
                and needle
                not in (
                    card.method_id
                    + " "
                    + card.summary
                    + " "
                    + " ".join(card.assumptions)
                ).lower()
            ):
                continue
            out.append(card)
        return out

    @staticmethod
    def verify_reference(card: MethodCard) -> bool:
        """Import the reference implementation (optional dependencies may make this False)."""
        module_name, attribute = card.reference_implementation.split(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return False
        return hasattr(module, attribute)


def _card(**kwargs) -> MethodCard:
    return MethodCard(**kwargs)


def default_cards() -> list[MethodCard]:
    return [
        _card(
            method_id="deterministic_charter_candidates",
            version="1",
            summary="Four rule-based strategy candidates, one per charter, scoring named features without a pretrained model.",
            supported_targets=["decimal_surprise", "decimal_return"],
            data_shapes=["security_features_by_date"],
            min_history="none: scores use only features known at the cutoff; bridge slope is fitted separately on design data",
            assumptions=[
                "features carry availability metadata upstream",
                "a score is a belief, not a position",
            ],
            interfaces=[
                "score(features, expected_resolution_at)",
                "forecast(features, model_id, certificate_id, ...)",
            ],
            compute="negligible",
            reference_implementation="pod_platform.strategies.deterministic:ALL_CANDIDATES",
            tests=["tests/test_strategies.py"],
            baselines=[
                "naive persistence",
                "sector-relative value",
                "lagged return reversal",
            ],
            failure_modes=[
                "already priced",
                "generic factor exposure",
                "feature timing leakage",
            ],
            spec_sections=["4.1", "4.2", "4.3", "4.4", "15.1"],
        ),
        _card(
            method_id="rank_spread_bridge",
            version="1",
            summary="Fixed linear map from a bounded belief score to a decimal_return forecast at a declared horizon.",
            supported_targets=["decimal_return"],
            data_shapes=["scaled_scores"],
            min_history="enough matured design observations to fit the slope (fitted outside the model)",
            assumptions=[
                "monotone relation between score and horizon excess return",
                "slope fitted on design data only",
            ],
            interfaces=["expected_excess_return(belief, scaled_score)"],
            hyperparameters={
                "slope_bp_per_unit": "basis points of expected excess return per unit scaled score"
            },
            compute="negligible",
            reference_implementation="pod_platform.strategies.base:RankSpreadBridge",
            tests=["tests/test_strategies.py"],
            baselines=["zero-alpha"],
            failure_modes=[
                "unstable slope",
                "score scale drift",
                "sign reversal across partitions",
            ],
            spec_sections=["15.3"],
        ),
        _card(
            method_id="family_equal_weight_integration",
            version="1",
            summary="Family-level mean, equal family weights, neutral prior with inflated uncertainty for a missing family.",
            supported_targets=["decimal_return"],
            data_shapes=["forecasts_by_security_and_family"],
            min_history="none",
            assumptions=[
                "families are not independent evidence",
                "missing family is not an observed zero",
            ],
            interfaces=["integrate(estimates)"],
            compute="negligible",
            reference_implementation="pod_platform.allocation.integration:FamilyIntegrator",
            tests=["tests/test_allocation_risk.py"],
            baselines=["single-family forecast"],
            failure_modes=[
                "duplicated information counted twice",
                "coverage-conditioned renormalization (not done)",
            ],
            spec_sections=["15.6", "16.3"],
        ),
        _card(
            method_id="deterministic_selection_retention_band",
            version="1",
            summary="Rank, select with an incumbent retention band, then size under official limits with a typed infeasibility report.",
            supported_targets=["target_weights"],
            data_shapes=["integrated_alphas_with_universe_flags"],
            min_history="none",
            assumptions=[
                "official limits are never relaxed",
                "dust positions are not added for breadth",
            ],
            interfaces=["build(candidates, ...)"],
            hyperparameters={
                "target_positions": "internal breadth",
                "retention_band": "incumbent keep fraction",
            },
            compute="negligible",
            reference_implementation="pod_platform.allocation.construction:ConstructionEngine",
            tests=["tests/test_allocation_risk.py"],
            baselines=["equal-weight top/bottom"],
            failure_modes=[
                "too few shortable names",
                "cap-binding gross reduction",
            ],
            spec_sections=["16.5", "16.7", "16.11"],
        ),
        _card(
            method_id="registered_rank_ic_comparison",
            version="1",
            summary="Persistent, budgeted comparison of supplied forecasts versus a baseline by date-level rank IC.",
            supported_targets=[
                "decimal_return",
                "decimal_surprise",
                "probability",
            ],
            data_shapes=["forecast_outcome_panel"],
            min_history="at least min_dates formation dates with matured labels",
            assumptions=[
                "labels mature before the evaluation clock",
                "one IC per formation date",
            ],
            interfaces=[
                "register(plan)",
                "evaluate(experiment, variant, observations, ...)",
            ],
            compute="light",
            reference_implementation="pod_platform.research.evaluation:RegisteredEvaluator",
            tests=["tests/test_registered_evaluation.py"],
            baselines=["naive baseline prediction"],
            failure_modes=[
                "descriptive SE only",
                "overlapping horizons reduce effective sample",
            ],
            spec_sections=["13.1", "13.6", "13.7"],
        ),
        _card(
            method_id="timing_shift_placebo",
            version="1",
            summary="Cyclic one-date prediction shift per security; a genuine signal should vanish under it.",
            supported_targets=["decimal_return"],
            data_shapes=["forecast_outcome_panel"],
            min_history="at least two formation dates per security",
            assumptions=[
                "preserves cross-sectional structure and serial dependence"
            ],
            interfaces=["timing_shift_placebo(observations)"],
            compute="negligible",
            reference_implementation="pod_platform.application.integrity:timing_shift_placebo",
            tests=["tests/test_reference_workflow.py"],
            baselines=["candidate IC"],
            failure_modes=[
                "slowly-moving predictions leave residual placebo IC"
            ],
            spec_sections=["13.9"],
        ),
        _card(
            method_id="causal_feature_operators",
            version="1",
            summary="Reviewed as-of operators (latest, lag, difference, growth, rolling mean) with revision filtering.",
            supported_targets=["feature"],
            data_shapes=["bitemporal_observations"],
            min_history="operator dependent",
            assumptions=[
                "every input carries available_at and version_known_at"
            ],
            interfaces=[
                "materialize(recipe, observations, formation_at, security_ids)"
            ],
            compute="light",
            reference_implementation="pod_platform.research.features:materialize",
            tests=["tests/test_feature_language.py"],
            baselines=[],
            failure_modes=[
                "missing availability metadata",
                "survivor-only coverage",
            ],
            spec_sections=["11.6"],
        ),
        _card(
            method_id="fixed_rule_execution_planner",
            version="1",
            summary="No-trade band, entry-edge test, participation-capped daily splitting and order expiry.",
            supported_targets=["orders"],
            data_shapes=["desired_trades_with_liquidity_snapshots"],
            min_history="trailing ADV and volatility known before the decision",
            assumptions=[
                "close-to-close convention",
                "costs are modeled, not observed",
            ],
            interfaces=[
                "plan(desires, liquidity, decision_at, target_book_id)"
            ],
            hyperparameters={
                "max_daily_participation": "fraction of ADV",
                "no_trade_band": "weight band",
            },
            compute="negligible",
            reference_implementation="pod_platform.execution.planner:ExecutionPlanner",
            tests=["tests/test_execution.py"],
            baselines=["fill everything at once (rejected: hides impact)"],
            failure_modes=["stale ADV", "foregone alpha while waiting"],
            spec_sections=["18.2", "18.3", "18.7"],
        ),
        _card(
            method_id="sqrt_impact_cost_model",
            version="1",
            summary="Half-spread tiers plus square-root participation impact plus explicit fees; sell and cover pay too.",
            supported_targets=["trading_cost"],
            data_shapes=["liquidity_snapshot"],
            min_history="trailing volatility and ADV",
            assumptions=[
                "coefficients are configured assumptions, not observed execution"
            ],
            interfaces=[
                "cost_rate(liquidity, order_dollars)",
                "expected_cost(liquidity, order_dollars)",
            ],
            hyperparameters={
                "eta": "impact coefficient",
                "half_spread_by_size_bp": "tier schedule",
            },
            compute="negligible",
            reference_implementation="pod_platform.execution.planner:CostModel",
            tests=["tests/test_execution.py"],
            baselines=["flat bp per side"],
            failure_modes=["understated impact in stress"],
            spec_sections=["18.3", "29.3"],
        ),
        _card(
            method_id="monthly_accounting_replay",
            version="1",
            summary="Monthly ledger replay with drifted holdings, borrow, cash, costs and NAV reconciliation.",
            supported_targets=["nav_path"],
            data_shapes=["monthly_excess_returns"],
            min_history="two or more contiguous periods for ratio statistics",
            assumptions=[
                "return-index marks, not quotes",
                "no margin or daily execution realism",
            ],
            interfaces=["rebalance_to_targets(...)", "advance(...)"],
            compute="light",
            reference_implementation="pod_platform.orchestration.replay:MonthlyAccountingReplay",
            tests=["tests/test_replay.py"],
            baselines=[],
            failure_modes=[
                "missing outcomes fail closed",
                "terminal returns need liquidation evidence",
            ],
            spec_sections=["19.1", "19.2", "19.4"],
        ),
        _card(
            method_id="daily_ledger_reconciliation",
            version="1",
            summary="Daily NAV/cost/benchmark reconciliation from supplied dated prices or returns, fills and cash entries.",
            supported_targets=["nav_path", "daily_beta", "worst_day"],
            data_shapes=[
                "daily_returns_or_prices",
                "dated_fills",
                "dated_cash_entries",
            ],
            min_history="two trading days (beta needs more)",
            assumptions=[
                "only supplied observations are used",
                "no interpolation",
                "explicit trading calendar",
            ],
            interfaces=["reconcile_daily(inputs, policy)"],
            compute="light",
            reference_implementation="pod_platform.reporting.daily:reconcile_daily",
            tests=["tests/test_daily_attribution.py"],
            baselines=["monthly replay NAV"],
            failure_modes=[
                "absent held outcome fails closed",
                "suspense marks flagged as not exact",
            ],
            spec_sections=["9.9", "19.2", "19.7", "19.8"],
        ),
        _card(
            method_id="factor_idiosyncratic_attribution",
            version="1",
            summary="Holdings-based and regression-based split of return into diagnostic-factor and idiosyncratic parts.",
            supported_targets=["attribution"],
            data_shapes=[
                "weights_returns_exposures",
                "portfolio_and_factor_returns",
            ],
            min_history="regression needs more observations than factors",
            assumptions=[
                "factors are internal diagnostics, not standard external definitions"
            ],
            interfaces=[
                "holdings_factor_attribution(...)",
                "regression_factor_attribution(...)",
            ],
            compute="light",
            reference_implementation="pod_platform.reporting.attribution:holdings_factor_attribution",
            tests=["tests/test_daily_attribution.py"],
            baselines=[],
            failure_modes=["collinear factors", "unstable loadings"],
            spec_sections=["19.8", "19.12"],
        ),
        _card(
            method_id="selection_sizing_control",
            version="1",
            summary="Selection versus sizing split with a seeded random-sizing control; stated baseline and interaction.",
            supported_targets=["attribution"],
            data_shapes=["position_weights_and_period_returns"],
            min_history="several periods",
            assumptions=[
                "accounting split is order dependent, not uniquely causal"
            ],
            interfaces=["selection_vs_sizing(...)"],
            compute="light",
            reference_implementation="pod_platform.reporting.attribution:selection_vs_sizing",
            tests=["tests/test_daily_attribution.py"],
            baselines=["equal-weight within side"],
            failure_modes=["few names make the control noisy"],
            spec_sections=["19.11", "19.12"],
        ),
        _card(
            method_id="alpha_decay_curve",
            version="1",
            summary="Dated post-entry payoff by trading-day window from supplied daily returns; refuses without prerequisites.",
            supported_targets=["payoff_curve"],
            data_shapes=["forecast_cohorts", "daily_returns"],
            min_history="at least min_cohorts formation cohorts with complete daily returns",
            assumptions=[
                "first earned return is the day after executable entry"
            ],
            interfaces=["alpha_decay_curve(...)"],
            compute="light",
            reference_implementation="pod_platform.reporting.attribution:alpha_decay_curve",
            tests=["tests/test_daily_attribution.py"],
            baselines=[],
            failure_modes=["too few cohorts", "incomplete daily returns"],
            spec_sections=["18.6", "19.11"],
        ),
        _card(
            method_id="failure_taxonomy_classifier",
            version="1",
            summary="Spec 12.2 failure taxonomy on measured inputs only; unclassified when inputs are missing.",
            supported_targets=["diagnosis"],
            data_shapes=["measured_metrics"],
            min_history="none",
            assumptions=["diagnoses are hypotheses, not conclusions"],
            interfaces=["classify_failure(metrics)"],
            compute="negligible",
            reference_implementation="pod_platform.application.diagnostics:classify_failure",
            tests=["tests/test_reference_workflow.py"],
            baselines=[],
            failure_modes=["silent on unmeasured causes"],
            spec_sections=["12.2", "12.5"],
        ),
    ]


def default_registry() -> MethodsRegistry:
    return MethodsRegistry(default_cards())
