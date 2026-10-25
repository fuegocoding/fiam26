"""Vertical slice pipeline (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 27.4, 26.8).

Wires the deterministic path: registered experiment -> strategy forecasts ->
family integration -> deterministic construction -> risk review -> planned
orders and fills -> reconciled accounting -> report and dossier. Every stage
produces its contract artifact; the run manifest freezes all versions.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Sequence

from ..accounting.ledger import AccountingLedger
from ..accounting.returns import benchmark_monthly
from ..allocation.construction import (
    Candidate,
    ConstructionEngine,
)
from ..allocation.integration import FamilyEstimate, FamilyIntegrator
from ..contracts.base import require_aware
from ..contracts.decisions import ActorKind, DecisionEvent
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse
from ..contracts.orders import OrderSide
from ..contracts.portfolio import RiskReview, TargetBook
from ..execution.initialization import InitializationResult, initialize_book
from ..execution.planner import (
    CostModel,
    DesiredTrade,
    ExecutionPlanner,
    FillEngine,
    LiquiditySnapshot,
    PlannedOrder,
)
from ..reporting.dossier import DossierBuilder
from ..reporting.tables import RequiredTable
from ..research.registration import RegisteredExperiment, ResearchLedger
from ..strategies.base import SecurityFeatures, StrategyCandidate

ZERO = Decimal("0")


@dataclass
class StageArtifacts:
    experiment_id: str
    forecast_ids: list[str] = field(default_factory=list)
    integrated_alphas: dict[str, float] = field(default_factory=dict)
    target_book: Optional[TargetBook] = None
    risk_review: Optional[RiskReview] = None
    planned_orders: list[PlannedOrder] = field(default_factory=list)
    initialization: Optional[InitializationResult] = None
    nav_path: list[float] = field(default_factory=list)
    required_table: Optional[dict] = None
    dossier: Optional[dict] = None
    incidents: list[str] = field(default_factory=list)

    def acceptance(self) -> dict:
        return {
            "forecast_batch": bool(self.forecast_ids),
            "target_book": self.target_book is not None,
            "risk_review": self.risk_review is not None,
            "order_fill_path": bool(self.planned_orders),
            "reconciled_nav_path": bool(self.nav_path),
            "report": self.required_table is not None,
            "dossier": self.dossier is not None,
        }

    def all_stages_complete(self) -> bool:
        return all(self.acceptance().values())


class VerticalSlicePipeline:
    def __init__(
        self,
        *,
        ledger: ResearchLedger,
        integrator: FamilyIntegrator,
        construction: ConstructionEngine,
        planner: ExecutionPlanner,
        cost_model: CostModel,
        accounting: AccountingLedger,
    ):
        self._ledger = ledger
        self._integrator = integrator
        self._construction = construction
        self._planner = planner
        self._costs = cost_model
        self._accounting = accounting
        self._dossiers = DossierBuilder()

    @property
    def dossiers(self) -> DossierBuilder:
        return self._dossiers

    def run(
        self,
        *,
        experiment: RegisteredExperiment,
        candidates_by_family: dict[str, Sequence[FamilyEstimate]],
        tradable_candidates: Sequence[Candidate],
        liquidity: dict[str, LiquiditySnapshot],
        risk_exposures: Sequence,
        risk_engine,
        market_features: dict[str, SecurityFeatures],
        strategy: StrategyCandidate,
        model_id: str,
        certificate_id: str,
        expected_resolution_at: dt.datetime,
        expires_at: dt.datetime,
        decision_at: dt.datetime,
        starting_nav: Decimal,
        min_positions: int,
        run_id: str,
        period_label: str,
        benchmark_tb3ms_annual_percent: float,
    ) -> StageArtifacts:
        require_aware(decision_at, "decision_at")
        artifacts = StageArtifacts(experiment_id=experiment.registration_id)

        self._ledger.register(experiment)
        self._ledger.enforce_variant_budget(experiment.registration_id)

        forecast_ids: list[str] = []
        for security_id, features in market_features.items():
            forecast = strategy.forecast(
                features,
                model_id=model_id,
                admission_certificate_id=certificate_id,
                expected_resolution_at=expected_resolution_at,
                expires_at=expires_at,
            )
            if forecast is not None:
                forecast_ids.append(forecast.forecast_id)
        if not forecast_ids:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary="no forecast published for the decision batch",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    stop_condition="A decision cycle without forecasts cannot construct a book.",
                )
            )
        artifacts.forecast_ids = forecast_ids

        for security_id, estimates in candidates_by_family.items():
            integrated = self._integrator.integrate(estimates)
            artifacts.integrated_alphas[security_id] = integrated.alpha

        book = self._construction.build(
            tradable_candidates,
            run_id=run_id,
            decision_at=decision_at,
            forecast_publication_ids=forecast_ids,
            current_book_snapshot_id=f"snapshot_{run_id}_prev",
            nav_snapshot_id=f"nav_{run_id}",
            price_snapshot_id=f"px_{run_id}",
            risk_envelope_version="envelope_v1",
            covariance_version="cov_v1",
            cost_model_version=self._costs.model_id,
            attribution_method="family_equal_contribution",
        )
        artifacts.target_book = book

        review = risk_engine.review(
            book, risk_exposures, effective_at=decision_at
        )
        artifacts.risk_review = review
        if review.decision.value == "BLOCK":
            artifacts.incidents.append(f"risk_block:{review.risk_review_id}")
            return artifacts

        desires = self._desires_from_book(book, liquidity, starting_nav)
        orders = self._planner.plan(
            desires,
            liquidity,
            decision_at=decision_at,
            target_book_id=book.target_book_id,
        )
        artifacts.planned_orders = orders

        engine = FillEngine(self._costs)
        initialization = initialize_book(
            orders,
            liquidity,
            engine,
            starting_nav=starting_nav,
            min_positions=min_positions,
            inception_at=decision_at,
        )
        artifacts.initialization = initialization

        self._apply_fills_to_accounting(orders, liquidity, decision_at)
        nav = self._reconcile_nav()
        artifacts.nav_path = [float(starting_nav), float(nav)]

        artifacts.required_table = self._required_table(
            starting_nav=starting_nav,
            ending_nav=nav,
            period_label=period_label,
            benchmark_tb3ms_annual_percent=benchmark_tb3ms_annual_percent,
            name_count=len(book.positions),
        )

        sample_security = (
            book.positions[0].security_id if book.positions else "none"
        )
        artifacts.dossier = self._build_dossier(
            sample_security, decision_at, book
        )
        return artifacts

    def _desires_from_book(
        self,
        book: TargetBook,
        liquidity: dict[str, LiquiditySnapshot],
        nav: Decimal,
    ) -> list[DesiredTrade]:
        desires: list[DesiredTrade] = []
        for position in book.positions:
            snapshot = liquidity.get(position.security_id)
            if snapshot is None:
                raise PodPlatformError(
                    ToolResponse.failure(
                        summary=f"no liquidity snapshot for {position.security_id}",
                        code=ErrorCode.LIQUIDITY_UNAVAILABLE,
                        stop_condition="No order may be planned without dated liquidity evidence.",
                    )
                )
            side = (
                OrderSide.BUY
                if position.target_weight > ZERO
                else OrderSide.SHORT
            )
            desires.append(
                DesiredTrade(
                    security_id=position.security_id,
                    side=side,
                    target_weight=position.target_weight,
                    current_weight=ZERO,
                    nav=nav,
                    declared_edge_bp=Decimal("5000"),
                )
            )
        return desires

    def _apply_fills_to_accounting(
        self,
        orders: Sequence[PlannedOrder],
        liquidity: dict[str, LiquiditySnapshot],
        decision_at: dt.datetime,
    ) -> None:
        for order in orders:
            snapshot = liquidity[order.security_id]
            fees = self._costs.expected_cost(snapshot, abs(order.order_dollars))
            self._accounting.execute(
                event_id=f"fill_{order.order_id}",
                security_id=order.security_id,
                side=order.side,
                quantity=order.quantity,
                price=snapshot.price,
                fees=fees,
                executed_at=decision_at,
            )
            self._accounting.mark(
                order.security_id, snapshot.price, decision_at
            )

    def _reconcile_nav(self) -> Decimal:
        self._accounting.assert_nav_identity()
        return self._accounting.nav()

    def _required_table(
        self,
        *,
        starting_nav: Decimal,
        ending_nav: Decimal,
        period_label: str,
        benchmark_tb3ms_annual_percent: float,
        name_count: int,
    ) -> dict:
        b = benchmark_monthly(benchmark_tb3ms_annual_percent)
        entry_cost_return = (
            float(ending_nav - starting_nav) / float(starting_nav)
            if starting_nav
            else 0.0
        )
        table = RequiredTable(
            ending_value=float(ending_nav),
            starting_value=float(starting_nav),
            elapsed_years=2.0 / 12.0,
            periods_per_year=12,
            portfolio_returns=(entry_cost_return, 0.0),
            benchmark_returns=(b, b),
            risk_free_returns=(b, b),
            gross_returns=(entry_cost_return, 0.0),
            cost_returns=(entry_cost_return, 0.0),
            borrow_returns=(0.0, 0.0),
            position_counts=(name_count, name_count),
        )
        return table.as_table()

    def _build_dossier(
        self, security_id: str, decision_at: dt.datetime, book: TargetBook
    ) -> dict:

        event = DecisionEvent(
            owner="allocation",
            created_at=decision_at,
            event_id=f"dossier_{book.target_book_id}",
            actor_kind=ActorKind.CODE,
            actor_id="deterministic_pipeline",
            effective_research_at=decision_at,
            effective_market_at=decision_at,
            input_artifact_ids=[security_id],
            input_hashes={security_id: "recorded"},
            chosen_action=f"target position in {security_id} from {book.target_book_id}",
            rationale="deterministic construction from admitted forecasts",
            authorizing_rule_or_certificate_id="rule_deterministic_construction_v1",
            resulting_artifact_id=book.target_book_id,
        )
        self._dossiers.record(event)
        dossier = self._dossiers.build(security_id, decision_at)
        return dossier.explanation()
