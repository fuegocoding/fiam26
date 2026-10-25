"""Replay control layer (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 16.5, 19.1-19.4, 22.4, 26.5, 26.8).

Reusable, fail-closed controls for historical replays:

* source eligibility gate (formation vs explicitly labeled research-only);
* outcome resolution with no zero-fill (dated liquidation evidence only);
* a monthly accounting replay on the real ``AccountingLedger`` with drifted
  holdings, turnover vs drifted holdings, cash/borrow/cost and NAV reconciliation;
* all-period, all-stage compliance on actual weights (pre and post recorded);
* fresh-output-directory artifact writing with sha256 per artifact;
* a freeze assessment that only calls ``freeze_run`` when measured
  prerequisites hold;
* ``ControlledReplay``: approved forecasts -> certificates -> source gate ->
  integration -> construction -> independent risk -> planner/fills -> ledger ->
  compliance -> manifest/report/exports/dossier.

Limitations (monthly inputs): see ``MONTHLY_LIMITATIONS``. Nothing here calls an
LLM, trains a model, or certifies alpha.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from pydantic import Field, model_validator

from ..accounting.ledger import AccountingLedger, BorrowSchedule
from ..accounting.returns import benchmark_monthly
from ..allocation.construction import Candidate, ConstructionEngine, InternalPolicy, OfficialLimits
from ..allocation.integration import FamilyEstimate, FamilyIntegrator
from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..contracts.decisions import ActorKind, DecisionEvent
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse
from ..contracts.forecasts import Forecast
from ..contracts.orders import OrderSide
from ..contracts.portfolio import RiskDecision, TargetPosition
from ..contracts.sources import CardStatus, SourceCard
from ..execution.planner import (
    CostModel,
    CostModelConfig,
    DesiredTrade,
    ExecutionPlanner,
    ExecutionPolicy,
    FillEngine,
    LiquiditySnapshot,
)
from ..reporting.compliance import ActualHolding, ComplianceEngine, ComplianceReport
from ..reporting.dossier import DossierBuilder, HoldingsRow, holdings_csv, returns_csv
from ..reporting.tables import RequiredTable
from ..research.certificates import (
    CertificateCheck,
    CertificateEnvironment,
    CertificatePolicy,
    RevocationLedger,
    check_scope_limits,
    resolve_policy,
    validate_certificate,
)
from ..research.promotion import AdmissionCertificate
from ..risk.envelope import MeasuredExposure, RiskEngine, RiskLimit, RiskPolicy
from .freeze import FreezeResult, freeze_run
from .run_manifest import RunManifest

ZERO = Decimal("0")
ONE = Decimal("1")
PRICE_FLOOR = Decimal("1e-12")

MONTHLY_LIMITATIONS = (
    "Inputs are monthly per-name excess returns: no intramonth path, no daily marks, no daily liquidity realism.",
    "Positions are marked on a return-index price (base 1 at entry), not on observed quotes.",
    "Fills occur at the formation-date reference; costs are modeled (flat bp or planner cost model), not observed.",
    "Short proceeds earn the risk-free rate; no margin, rebate, or funding spread is modeled.",
    "Names absent from the formation universe are exited at the carried mark; no forced-hold logic.",
)


def fail(code: ErrorCode, summary: str, stop: str) -> None:
    raise PodPlatformError(ToolResponse.failure(summary=summary, code=code, stop_condition=stop))


def utc_close(day: dt.date) -> dt.datetime:
    """Timezone-aware (UTC) market-close reference for a calendar date."""
    return dt.datetime(day.year, day.month, day.day, 21, 0, tzinfo=dt.timezone.utc)


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def _dec(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --------------------------------------------------------------- source gate


@dataclass(frozen=True)
class SourceUse:
    name: str  # consuming family / component
    source_id: str
    role: str = "external"  # "core" or "external"
    card: Optional[SourceCard] = None


@dataclass
class SourceGateResult:
    research_only: bool
    uses: list[dict] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    @property
    def formation_inputs_approved(self) -> bool:
        return not self.blocked and bool(self.uses)

    @property
    def freeze_claim_allowed(self) -> bool:
        return self.formation_inputs_approved and not self.research_only

    @property
    def run_label(self) -> str:
        if self.research_only:
            return "RESEARCH_ONLY_NOT_FREEZABLE"
        return "FORMATION_ELIGIBLE"

    def as_dict(self) -> dict:
        return {
            "run_label": self.run_label,
            "research_only": self.research_only,
            "formation_inputs_approved": self.formation_inputs_approved,
            "freeze_claim_allowed": self.freeze_claim_allowed,
            "blocked_sources": list(self.blocked),
            "uses": list(self.uses),
        }


def source_cards_from_records(records: Iterable[Mapping[str, Any]]) -> dict[str, SourceCard]:
    cards: dict[str, SourceCard] = {}
    for record in records:
        card = SourceCard.model_validate(record)
        if card.source_id in cards:
            fail(ErrorCode.SCHEMA_MISMATCH, f"duplicate source card {card.source_id}", "One card per source id.")
        cards[card.source_id] = card
    return cards


def load_source_cards(path: Path) -> dict[str, SourceCard]:
    return source_cards_from_records(json.loads(Path(path).read_text(encoding="utf-8")))


def _card_eligible(use: SourceUse) -> bool:
    card = use.card
    return (
        card is not None
        and card.source_id == use.source_id
        and card.status == CardStatus.APPROVED
        and card.formation_eligible
    )


class SourceEligibilityGate:
    """Formation requires an APPROVED, formation_eligible card for every source.

    ``research_only`` permits ineligible sources for a labeled exploratory run,
    which can never claim freeze or admission.
    """

    def __init__(self, research_only: bool = False):
        self.research_only = research_only

    def resolve(
        self, uses: Iterable[tuple[str, str, str]], cards: Mapping[str, SourceCard]
    ) -> list[SourceUse]:
        return [SourceUse(name, source_id, role, cards.get(source_id)) for name, source_id, role in uses]

    def evaluate(self, uses: Sequence[SourceUse]) -> SourceGateResult:
        result = SourceGateResult(research_only=self.research_only)
        for use in uses:
            eligible = _card_eligible(use)
            result.uses.append(
                {
                    "consumer": use.name,
                    "source_id": use.source_id,
                    "role": use.role,
                    "card_snapshot_id": use.card.snapshot_id if use.card else None,
                    "card_status": use.card.status.value if use.card else "NO_CARD",
                    "formation_eligible": eligible,
                }
            )
            if not eligible and use.source_id not in result.blocked:
                result.blocked.append(use.source_id)
        if result.blocked and not self.research_only:
            fail(
                ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE,
                f"sources not formation-eligible: {', '.join(sorted(result.blocked))}",
                "Formation needs APPROVED formation-eligible source cards; use an explicit research-only run for exploration.",
            )
        return result


def file_fingerprint(path: Path, *, hash_content: bool = True) -> dict:
    path = Path(path)
    if not path.is_file():
        fail(ErrorCode.SCHEMA_MISMATCH, f"lineage file missing: {path}", "Core source lineage must reference existing files.")
    stat = path.stat()
    digest = None
    if hash_content:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "sha256": digest,
    }


@dataclass
class CoreSourceLineage:
    entries: dict[str, dict] = field(default_factory=dict)

    def add(self, source_id: str, path: Path, *, role: str, hash_content: bool = True) -> None:
        self.entries[source_id] = {"role": role, **file_fingerprint(path, hash_content=hash_content)}

    @property
    def fully_hashed(self) -> bool:
        return bool(self.entries) and all(e.get("sha256") for e in self.entries.values())

    def as_dict(self) -> dict:
        return {k: self.entries[k] for k in sorted(self.entries)}


# ------------------------------------------------------------------- outcomes


class LiquidationEvidence(StrictModel):
    """Explicit dated evidence that a held name was liquidated inside a period."""

    security_id: str = Field(min_length=1)
    liquidated_at: dt.datetime
    realized_excess_return: float
    evidence_id: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "LiquidationEvidence":
        require_aware(self.liquidated_at, "liquidated_at")
        if not math.isfinite(self.realized_excess_return):
            raise ValueError("realized_excess_return must be finite")
        return self


@dataclass
class ResolvedOutcomes:
    returns: dict[str, float] = field(default_factory=dict)
    liquidated: dict[str, LiquidationEvidence] = field(default_factory=dict)
    basis: dict[str, str] = field(default_factory=dict)


def resolve_outcomes(
    held: Iterable[str],
    outcomes: Mapping[str, Any],
    liquidations: Optional[Mapping[str, LiquidationEvidence]] = None,
    *,
    formation_at: dt.datetime,
    period_end_at: dt.datetime,
) -> ResolvedOutcomes:
    """Resolve a realized return for every held name or fail closed.

    Missing/non-finite outcomes are never zero-filled. The only substitute is
    ``LiquidationEvidence`` dated inside (formation, period_end]; later quotes
    are rejected.
    """
    formation_at = require_aware(formation_at, "formation_at")
    period_end_at = require_aware(period_end_at, "period_end_at")
    liquidations = liquidations or {}
    resolved = ResolvedOutcomes()
    missing: list[str] = []
    for sid in sorted(set(held)):
        evidence = liquidations.get(sid)
        if evidence is not None:
            when = require_aware(evidence.liquidated_at, "liquidated_at")
            if not (formation_at < when <= period_end_at):
                fail(
                    ErrorCode.ASOF_VIOLATION,
                    f"liquidation evidence for {sid} dated {when.isoformat()} is outside ({formation_at.isoformat()}, {period_end_at.isoformat()}]",
                    "Liquidation evidence must be dated inside the holding period; later quotes are future information.",
                )
            resolved.returns[sid] = float(evidence.realized_excess_return)
            resolved.liquidated[sid] = evidence
            resolved.basis[sid] = f"liquidation:{evidence.evidence_id}"
            continue
        value = _finite(outcomes.get(sid))
        if value is None:
            missing.append(sid)
            continue
        resolved.returns[sid] = value
        resolved.basis[sid] = "outcome"
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        fail(
            ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
            f"missing outcome for {len(missing)} held name(s) in period ending {period_end_at.date()}: {shown}",
            "Missing realized returns are never zero-filled; supply dated liquidation evidence or stop.",
        )
    return resolved


def validate_contiguous_periods(periods: Sequence[tuple[dt.date, dt.date]]) -> None:
    """Each period is (formation_date, period_end); consecutive periods must abut."""
    previous_end: Optional[dt.date] = None
    for formation, end in periods:
        if end <= formation:
            fail(ErrorCode.ASOF_VIOLATION, f"period {formation}->{end} is not forward in time", "Periods must move forward.")
        if previous_end is not None and formation != previous_end:
            fail(
                ErrorCode.ASOF_VIOLATION,
                f"period gap: previous period ended {previous_end} but next formation is {formation}",
                "Months may not be silently skipped; every period must abut the previous one.",
            )
        previous_end = end


# ------------------------------------------------------------------ accounting


@dataclass(frozen=True)
class FillInstruction:
    security_id: str
    signed_dollars: Decimal = ZERO
    fee: Decimal = ZERO
    close_out: bool = False
    entry_price: Optional[Decimal] = None
    order_id: str = ""


@dataclass
class TradeResult:
    nav_open: Decimal
    nav_post_trade: Decimal
    fees: Decimal
    traded_dollars: Decimal
    turnover_one_way: float
    pre_weights: dict[str, Decimal]
    post_weights: dict[str, Decimal]
    trades: list[dict]


@dataclass
class PeriodResult:
    nav_open: Decimal
    nav_post_trade: Decimal
    nav_end: Decimal
    net_return: float
    gross_return: float
    cost_return: float
    borrow_return: float
    cash_income: Decimal
    active_return: float
    rf_monthly: float
    names: int
    gross_exposure: float
    net_exposure: float
    drifted_weights: dict[str, Decimal]
    liquidated: list[str]
    reconciliation: dict

    def as_dict(self) -> dict:
        return {
            "nav_open": str(self.nav_open),
            "nav_post_trade": str(self.nav_post_trade),
            "nav_end": str(self.nav_end),
            "net_return": self.net_return,
            "gross_return": self.gross_return,
            "cost_return": self.cost_return,
            "borrow_return": self.borrow_return,
            "cash_income": str(self.cash_income),
            "active_return": self.active_return,
            "rf_monthly": self.rf_monthly,
            "names": self.names,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "liquidated": list(self.liquidated),
            "reconciliation": self.reconciliation,
        }


@dataclass
class _OpenPeriod:
    nav_open: Decimal
    nav_post: Decimal
    fees: Decimal


class MonthlyAccountingReplay:
    """Monthly replay on the real ``AccountingLedger``.

    Prices are a return index (base ``entry_price`` at entry), advanced by
    ``rf + excess`` each period. Cash earns ``rf``; borrow accrues on short
    notional at the formation mark; trading fees come out of cash.
    """

    def __init__(
        self,
        starting_nav: Decimal,
        *,
        borrow_schedule: BorrowSchedule,
        entry_price: Decimal = ONE,
        periods_per_year: int = 12,
        trading_days_per_period: int = 21,
    ):
        self.ledger = AccountingLedger(_dec(starting_nav))
        self.schedule = borrow_schedule
        self.entry_price = _dec(entry_price)
        self.periods_per_year = periods_per_year
        self.trading_days = trading_days_per_period
        self._open: Optional[_OpenPeriod] = None
        self._cum_pnl = ZERO

    def nav(self) -> Decimal:
        return self.ledger.nav()

    def held(self) -> set[str]:
        return {p.security_id for p in self.ledger.marked_positions()}

    def actual_weights(self) -> dict[str, Decimal]:
        positions = self.ledger.marked_positions()
        if not positions:
            return {}
        nav = self.ledger.nav()
        if nav <= ZERO:
            fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, "NAV is not positive", "Accounting cannot continue on non-positive NAV.")
        return {p.security_id: p.market_value() / nav for p in positions}

    def _tolerance(self, nav: Decimal) -> Decimal:
        return max(Decimal("1e-6"), abs(nav) * Decimal("1e-15"))

    def _execute_signed(self, sid: str, qty: Decimal, price: Decimal, fee: Decimal, at: dt.datetime, tag: str) -> None:
        position = self.ledger.position(sid)
        current = position.quantity
        parts: list[tuple[OrderSide, Decimal]] = []
        if qty > ZERO:
            remaining = qty
            if current < ZERO:
                cover = min(remaining, -current)
                parts.append((OrderSide.COVER, cover))
                remaining -= cover
            if remaining > ZERO:
                parts.append((OrderSide.BUY, remaining))
        else:
            remaining = -qty
            if current > ZERO:
                sell = min(remaining, current)
                parts.append((OrderSide.SELL, sell))
                remaining -= sell
            if remaining > ZERO:
                parts.append((OrderSide.SHORT, remaining))
        total = sum((q for _, q in parts), ZERO)
        assigned = ZERO
        for index, (side, part_qty) in enumerate(parts):
            part_fee = fee - assigned if index == len(parts) - 1 else fee * part_qty / total
            assigned += part_fee
            self.ledger.execute(f"{tag}_{sid}_{index}", sid, side, part_qty, price, part_fee, at)
        self.ledger.mark(sid, price, at)

    def apply_fills(self, fills: Sequence[FillInstruction], *, at: dt.datetime, tag: str = "fill") -> TradeResult:
        at = require_aware(at, "at")
        if self._open is not None:
            fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, "previous period was not advanced", "Each trade must be followed by period accounting.")
        nav_open = self.ledger.nav()
        pre_weights = self.actual_weights()
        traded = ZERO
        fees_total = ZERO
        trades: list[dict] = []
        for fill in fills:
            position = self.ledger.positions.get(fill.security_id)
            held = position is not None and position.quantity != ZERO
            price = position.last_mark if held else (fill.entry_price or self.entry_price)
            if price is None or price <= ZERO:
                fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, f"no positive price for {fill.security_id}", "Fills need a positive reference price.")
            if fill.close_out:
                if not held:
                    continue
                qty = -position.quantity
            else:
                qty = fill.signed_dollars / price
            if qty == ZERO:
                continue
            self._execute_signed(fill.security_id, qty, price, fill.fee, at, f"{tag}{fill.order_id}")
            traded += abs(qty) * price
            fees_total += fill.fee
            trades.append(
                {
                    "security_id": fill.security_id,
                    "order_id": fill.order_id,
                    "quantity": str(qty),
                    "price": str(price),
                    "dollars": str(qty * price),
                    "fee": str(fill.fee),
                }
            )
        nav_post = self.ledger.nav()
        self._open = _OpenPeriod(nav_open=nav_open, nav_post=nav_post, fees=fees_total)
        turnover = float(traded / (2 * nav_open)) if nav_open > ZERO else 0.0
        return TradeResult(
            nav_open=nav_open,
            nav_post_trade=nav_post,
            fees=fees_total,
            traded_dollars=traded,
            turnover_one_way=turnover,
            pre_weights=pre_weights,
            post_weights=self.actual_weights(),
            trades=trades,
        )

    def rebalance_to_targets(
        self,
        targets: Mapping[str, Any],
        *,
        at: dt.datetime,
        cost_bp_per_side: Decimal,
        entry_prices: Optional[Mapping[str, Decimal]] = None,
    ) -> TradeResult:
        """Trade from drifted actual holdings to target weights over the held-name union."""
        nav = self.ledger.nav()
        weights = self.actual_weights()
        fills: list[FillInstruction] = []
        for sid in sorted(set(targets) | self.held()):
            target = _dec(targets.get(sid, 0))
            if target == ZERO and sid in weights:
                position = self.ledger.positions[sid]
                dollars = -position.market_value()
                close = True
            else:
                dollars = target * nav - weights.get(sid, ZERO) * nav
                close = False
            if dollars == ZERO:
                continue
            fills.append(
                FillInstruction(
                    security_id=sid,
                    signed_dollars=dollars,
                    fee=abs(dollars) * _dec(cost_bp_per_side) / Decimal(10000),
                    close_out=close,
                    entry_price=(entry_prices or {}).get(sid),
                    order_id=f"_{at.date().isoformat()}",
                )
            )
        return self.apply_fills(fills, at=at, tag="rebal")

    def advance(
        self,
        *,
        period_end_at: dt.datetime,
        rf_monthly: float,
        resolved: ResolvedOutcomes,
        market_caps: Optional[Mapping[str, Decimal]] = None,
    ) -> PeriodResult:
        period_end_at = require_aware(period_end_at, "period_end_at")
        if self._open is None:
            fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, "no trade precedes this period", "Trade before advancing.")
        opened = self._open
        ledger = self.ledger
        held = [(p.security_id, p.quantity, p.last_mark) for p in sorted(ledger.marked_positions(), key=lambda p: p.security_id)]
        missing = [sid for sid, _, _ in held if sid not in resolved.returns]
        if missing:
            fail(
                ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                f"no resolved outcome for held names: {', '.join(missing[:10])}",
                "Every held name needs an outcome or dated liquidation evidence.",
            )
        rf = _dec(rf_monthly)
        # Validate every implied total price before any ledger mutation: a price at or
        # below zero is invalid data, never silently floored. Exactly -100% is only
        # accepted with dated liquidation evidence (ledger needs a positive mark, so
        # the technical PRICE_FLOOR is used and recorded as the liquidation price).
        marks: dict[str, Decimal] = {}
        for sid, _qty, p0 in held:
            raw = p0 * (ONE + rf + _dec(resolved.returns[sid]))
            if raw < ZERO or (raw == ZERO and sid not in resolved.liquidated):
                fail(
                    ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                    f"implied total return for {sid} is at or below -100% (price {raw}) on {period_end_at.date()}",
                    "Impossible prices are never floored; supply dated liquidation evidence for an exact -100% or fix the input.",
                )
            marks[sid] = raw if raw > ZERO else PRICE_FLOOR
        weights_during = self.actual_weights()
        gross_exposure = float(sum((abs(w) for w in weights_during.values()), ZERO))
        net_exposure = float(sum(weights_during.values(), ZERO))
        nav_post = ledger.nav()

        income_before = ledger.cash_income
        ledger.accrue_cash(rf * self.periods_per_year, self.periods_per_year, period_end_at)
        income = ledger.cash_income - income_before
        borrow = ledger.accrue_borrow(self.schedule, dict(market_caps or {}), self.trading_days, period_end_at)

        pnl = ZERO
        active = ZERO
        liquidated: list[str] = []
        for sid, qty, p0 in held:
            x = _dec(resolved.returns[sid])
            p1 = marks[sid]
            pnl += qty * (p1 - p0)
            active += qty * p0 * x
            ledger.mark(sid, p1, period_end_at)
            if sid in resolved.liquidated:
                side = OrderSide.SELL if qty > ZERO else OrderSide.COVER
                ledger.execute(f"liquidate_{sid}_{period_end_at.date().isoformat()}", sid, side, abs(qty), p1, ZERO, period_end_at)
                liquidated.append(sid)
        self._cum_pnl += pnl
        nav_end = ledger.nav()

        tol = self._tolerance(nav_end)
        expected_period = nav_post + income + pnl - borrow
        recomputed = ledger.cash + sum((p.market_value() for p in ledger.marked_positions()), ZERO)
        expected_cum = (
            ledger.starting_cash
            + self._cum_pnl
            + ledger.cash_income
            - ledger.trading_costs
            - ledger.borrow_costs
            - ledger.funding_costs
        )
        residuals = {
            "period_nav_residual": str(nav_end - expected_period),
            "cumulative_conservation_residual": str(nav_end - expected_cum),
            "identity_residual": str(nav_end - recomputed),
        }
        for name, value in residuals.items():
            if abs(Decimal(value)) > tol:
                fail(
                    ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                    f"NAV reconciliation failed on {period_end_at.date()}: {name}={value}",
                    "Accounting must conserve cash, positions, costs and borrow before any report.",
                )
        net_return = float(nav_end / opened.nav_open - ONE)
        cost_return = float(opened.fees / opened.nav_open)
        borrow_return = float(borrow / opened.nav_open)
        self._open = None
        return PeriodResult(
            nav_open=opened.nav_open,
            nav_post_trade=nav_post,
            nav_end=nav_end,
            net_return=net_return,
            gross_return=net_return + cost_return + borrow_return,
            cost_return=cost_return,
            borrow_return=borrow_return,
            cash_income=income,
            active_return=float(active / opened.nav_open),
            rf_monthly=float(rf),
            names=len(held),
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            drifted_weights=self.actual_weights(),
            liquidated=liquidated,
            reconciliation=residuals,
        )


# ------------------------------------------------------------------ compliance


class PeriodComplianceLog:
    """Compliance on actual weights for every period and stage, pre and post recorded."""

    def __init__(self, limits: OfficialLimits, *, fail_closed: bool = True):
        self._engine = ComplianceEngine(limits)
        self.fail_closed = fail_closed
        self.records: list[dict] = []

    def record(
        self,
        period: str,
        stage: str,
        pre_weights: Mapping[str, Any],
        post_weights: Mapping[str, Any],
        market_caps: Mapping[str, Any],
        formation_caps: Optional[Mapping[str, Any]] = None,
    ) -> ComplianceReport:
        pre = {k: _dec(v) for k, v in pre_weights.items() if _dec(v) != ZERO}
        post = {k: _dec(v) for k, v in post_weights.items() if _dec(v) != ZERO}
        holdings: list[ActualHolding] = []
        for sid, weight in sorted(post.items()):
            cap = market_caps.get(sid)
            if cap is None:
                fail(ErrorCode.SCHEMA_MISMATCH, f"no market-cap evidence for held {sid} ({period}/{stage})", "Compliance needs a cap for every held name.")
            formation_cap = (formation_caps or {}).get(sid, cap)
            holdings.append(ActualHolding(sid, weight, _dec(cap), _dec(formation_cap)))
        report = self._engine.check(holdings)
        failing = [c.name for c in report.checks if c.status == "fail" and c.scope == "official"]
        self.records.append(
            {
                "period": period,
                "stage": stage,
                "pre_weights": {k: str(v) for k, v in sorted(pre.items())},
                "post_weights": {k: str(v) for k, v in sorted(post.items())},
                "pre_summary": _weight_summary(pre),
                "post_summary": _weight_summary(post),
                "official_pass": report.official_pass,
                "failing_official": failing,
                "internal_warnings": [c.name for c in report.checks if c.status == "warn"],
                "report": report.as_dict(),
            }
        )
        if failing and self.fail_closed:
            fail(
                ErrorCode.PORTFOLIO_INFEASIBLE,
                f"official limit breach at {period}/{stage}: {', '.join(failing)}",
                "Official limits are checked on actual weights in every period and never relaxed.",
            )
        return report

    @property
    def all_official_pass(self) -> bool:
        return bool(self.records) and all(r["official_pass"] for r in self.records)

    def internal_warnings(self) -> list[dict]:
        return [
            {"period": r["period"], "stage": r["stage"], "warnings": r["internal_warnings"]}
            for r in self.records
            if r["internal_warnings"]
        ]

    def periods(self) -> list[str]:
        return sorted({r["period"] for r in self.records})

    def as_dict(self) -> dict:
        return {
            "all_official_pass": self.all_official_pass,
            "records": self.records,
            "internal_warnings": self.internal_warnings(),
            "periods_checked": self.periods(),
        }


def _weight_summary(weights: Mapping[str, Decimal]) -> dict:
    return {
        "names": len(weights),
        "gross": str(sum((abs(w) for w in weights.values()), ZERO)),
        "net": str(sum(weights.values(), ZERO)),
    }


@dataclass
class AccountedPeriod:
    trade: TradeResult
    result: PeriodResult
    resolved: ResolvedOutcomes


def account_period(
    replay: MonthlyAccountingReplay,
    compliance: PeriodComplianceLog,
    *,
    period: str,
    formation_at: dt.datetime,
    end_at: dt.datetime,
    construction_weights: Mapping[str, Any],
    post_risk_weights: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    liquidations: Optional[Mapping[str, LiquidationEvidence]],
    rf_monthly: float,
    cost_bp_per_side: Any,
    market_caps: Mapping[str, Any],
) -> AccountedPeriod:
    """One hardened period: all-stage compliance, drift-based trade, no-zero-fill outcomes.

    ``market_caps`` must cover every name held before or after the trade.
    Compliance records (pre and post weights) are appended before any breach
    raises, so a failed run still carries its evidence.
    """
    drifted = replay.actual_weights()
    compliance.record(period, "formation_target", drifted, construction_weights, market_caps)
    compliance.record(period, "post_risk", construction_weights, post_risk_weights, market_caps)
    trade = replay.rebalance_to_targets(
        post_risk_weights, at=formation_at, cost_bp_per_side=_dec(cost_bp_per_side)
    )
    compliance.record(period, "post_trade_actual", trade.pre_weights, trade.post_weights, market_caps)
    resolved = resolve_outcomes(
        replay.held(), outcomes, liquidations, formation_at=formation_at, period_end_at=end_at
    )
    caps = {k: _dec(v) for k, v in market_caps.items()}
    result = replay.advance(period_end_at=end_at, rf_monthly=rf_monthly, resolved=resolved, market_caps=caps)
    compliance.record(period, "period_end_drifted", trade.post_weights, result.drifted_weights, market_caps)
    return AccountedPeriod(trade, result, resolved)


# Sources consumed by each legacy family / component. A family whose sources lack
# an APPROVED formation-eligible card is blocked for formation.
LEGACY_FAMILY_SOURCES: dict[str, tuple[str, ...]] = {
    "inventory_retail": ("core_chars_panel",),
    "oplev_software": ("core_chars_panel",),
    "runway_micro": ("core_chars_panel",),
    "e7_surprise_breadth": ("core_chars_panel", "external_parquet"),
    "conf_revision_gated": ("core_chars_panel", "external_parquet"),
    "xgb_rank": ("core_chars_panel", "core_factor_list"),
    "filing_events": ("core_chars_panel", "external_parquet", "filings_8k"),
    "gap_micro": ("core_chars_panel", "external_parquet"),
    "demand": ("demand_card_foot_traffic",),
    "traded": ("traded_demand",),
    "gate_e7_demand": ("demand_card_foot_traffic",),
    "graph_tilt": ("graph_forecasts",),
    "short_interest": ("core_chars_panel", "external_parquet"),
    "risk_free_benchmark": ("external_parquet",),
}
LEGACY_NONCORE_FAMILIES = ("demand", "traded", "gate_e7_demand")


def legacy_source_uses(families: Iterable[str], cards: Mapping[str, SourceCard]) -> list[SourceUse]:
    uses: list[SourceUse] = []
    for family in families:
        if family not in LEGACY_FAMILY_SOURCES:
            fail(ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE, f"family {family!r} has no declared source lineage", "Every family declares the sources it consumes.")
        for source_id in LEGACY_FAMILY_SOURCES[family]:
            role = "external" if family in LEGACY_NONCORE_FAMILIES or source_id not in ("core_chars_panel", "core_factor_list") else "core"
            uses.append(SourceUse(family, source_id, role, cards.get(source_id)))
    return uses


# ---------------------------------------------------------------- output files


def prepare_output_dir(path: Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            fail(ErrorCode.SCHEMA_MISMATCH, f"output path is not a directory: {path}", "Choose a fresh output directory.")
        if any(path.iterdir()) and not overwrite:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"output directory is not empty: {path}",
                "Existing artifacts are preserved; use a fresh directory or pass an explicit overwrite.",
            )
    else:
        path.mkdir(parents=True)
    return path


class ArtifactWriter:
    def __init__(self, out_dir: Path, *, overwrite: bool = False):
        self.out_dir = prepare_output_dir(out_dir, overwrite=overwrite)
        self.overwrite = overwrite
        self.hashes: dict[str, str] = {}
        self.payloads: dict[str, bytes] = {}

    def write_bytes(self, name: str, payload: bytes) -> str:
        if not name or "/" in name or "\\" in name or name.startswith("."):
            fail(ErrorCode.SCHEMA_MISMATCH, f"invalid artifact name {name!r}", "Artifact names are plain file names.")
        target = self.out_dir / name
        if self.overwrite:
            fd, tmp = tempfile.mkstemp(dir=self.out_dir, prefix=".tmp_")
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            os.replace(tmp, target)
        else:
            with target.open("xb") as handle:
                handle.write(payload)
        digest = sha256_hex(payload)
        self.hashes[name] = digest
        self.payloads[name] = payload
        return digest

    def write_text(self, name: str, text: str) -> str:
        return self.write_bytes(name, text.encode("utf-8"))

    def write_json(self, name: str, obj: Any) -> str:
        return self.write_text(name, dumps(obj))


# --------------------------------------------------------------------- freeze


@dataclass
class FreezeAssessment:
    status: str
    reasons: list[str] = field(default_factory=list)
    result: Optional[FreezeResult] = None

    @property
    def frozen(self) -> bool:
        return self.status == "FROZEN"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "freeze_result": self.result.as_dict() if self.result else None,
        }


def assess_freeze(
    *,
    manifest: Optional[RunManifest],
    outputs: Mapping[str, bytes],
    gate: SourceGateResult,
    certificate_count: int,
    deployed_forecast_count: int,
    accounting_reconciled: bool,
    compliance: PeriodComplianceLog,
    trial_count: int,
    evaluation_access_count: int,
    frozen_at: dt.datetime,
) -> FreezeAssessment:
    """Return FROZEN only when measured prerequisites hold and ``freeze_run`` accepts."""
    reasons: list[str] = []
    if gate.research_only:
        reasons.append("research-only run cannot claim freeze or admission")
    if not gate.formation_inputs_approved:
        reasons.append(f"formation inputs not approved: {', '.join(sorted(gate.blocked)) or 'no sources declared'}")
    if manifest is None:
        reasons.append("run manifest incomplete: required lineage/versions were not declared")
    if deployed_forecast_count <= 0 or certificate_count < deployed_forecast_count:
        reasons.append(f"certificates {certificate_count} do not cover {deployed_forecast_count} deployed forecasts")
    if not accounting_reconciled:
        reasons.append("accounting did not reconcile")
    if not compliance.all_official_pass:
        reasons.append("official compliance not passed in every checked period")
    if trial_count <= 0 or evaluation_access_count < 0:
        reasons.append("trial count / evaluation exposure not declared")
    if reasons:
        return FreezeAssessment("NOT_FROZEN", reasons)
    try:
        result = freeze_run(
            manifest,
            dict(outputs),
            frozen_at=frozen_at,
            formation_inputs_approved=True,
            certificate_count=certificate_count,
            deployed_forecast_count=deployed_forecast_count,
            accounting_reconciled=True,
            actual_holdings_compliant=True,
            trial_count=trial_count,
            evaluation_access_count=evaluation_access_count,
        )
    except PodPlatformError as error:
        return FreezeAssessment("NOT_FROZEN", [error.response.summary])
    return FreezeAssessment("FROZEN", [], result)


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def environment_lock_hash() -> str:
    parts = [f"python={sys.version.split()[0]}"]
    for name in ("numpy", "pandas", "pydantic"):
        try:
            module = __import__(name)
            parts.append(f"{name}={getattr(module, '__version__', 'unknown')}")
        except Exception:  # pragma: no cover - optional modules
            parts.append(f"{name}=absent")
    return sha256_hex("|".join(parts).encode("utf-8"))


# ------------------------------------------------------------ controlled replay


@dataclass(frozen=True)
class MarketRow:
    security_id: str
    ticker: str
    name: str
    label_source: str
    label_reference_date: str
    market_cap_usd: Decimal
    beta: float
    sector: str
    tradable: bool
    price: Decimal
    adv_dollars: Decimal
    daily_volatility: float
    size_tier: str


@dataclass
class FixtureData:
    config: dict
    cards: dict[str, SourceCard]
    certificates: dict[str, AdmissionCertificate]
    forecasts: list[Forecast]
    market: dict[dt.date, dict[str, MarketRow]]
    outcomes: dict[dt.date, dict[str, Optional[float]]]
    periods: list[tuple[dt.date, dt.date, float]]
    liquidations: dict[dt.date, dict[str, LiquidationEvidence]]
    lineage: dict
    file_hashes: dict[str, str]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_fixture(directory: Path) -> FixtureData:
    directory = Path(directory)

    def required(name: str) -> Path:
        path = directory / name
        if not path.is_file():
            fail(ErrorCode.SCHEMA_MISMATCH, f"fixture file missing: {name}", "Controlled replay needs explicit fixtures.")
        return path

    file_hashes: dict[str, str] = {}

    def read(name: str, optional: bool = False) -> Optional[bytes]:
        path = directory / name
        if optional and not path.is_file():
            return None
        data = required(name).read_bytes()
        file_hashes[name] = sha256_hex(data)
        return data

    config = json.loads(read("config.json").decode("utf-8"))
    cards = source_cards_from_records(json.loads(read("source_cards.json").decode("utf-8")))
    certificates: dict[str, AdmissionCertificate] = {}
    for record in json.loads(read("certificates.json").decode("utf-8")):
        certificate = AdmissionCertificate.model_validate(record)
        certificates[certificate.certificate_id] = certificate
    forecasts = [Forecast.model_validate(r) for r in json.loads(read("forecasts.json").decode("utf-8"))]
    market: dict[dt.date, dict[str, MarketRow]] = {}
    outcomes: dict[dt.date, dict[str, Optional[float]]] = {}
    read("market.csv")
    for row in _read_csv(directory / "market.csv"):
        day = dt.date.fromisoformat(row["formation_date"])
        sid = row["security_id"]
        if sid in market.setdefault(day, {}):
            fail(ErrorCode.SCHEMA_MISMATCH, f"duplicate market row {day}/{sid}", "One row per security and date.")
        market[day][sid] = MarketRow(
            security_id=sid,
            ticker=row["ticker"],
            name=row["name"],
            label_source=row["label_source"],
            label_reference_date=row["label_reference_date"],
            market_cap_usd=Decimal(row["market_cap_usd"]),
            beta=float(row["beta"]),
            sector=row["sector"],
            tradable=row["tradable"].strip().lower() in ("1", "true", "yes"),
            price=Decimal(row["price"]),
            adv_dollars=Decimal(row["adv_dollars"]),
            daily_volatility=float(row["daily_volatility"]),
            size_tier=row["size_tier"],
        )
        outcome = row.get("outcome_excess_return", "")
        outcomes.setdefault(day, {})[sid] = _finite(outcome) if str(outcome).strip() else None
    periods: list[tuple[dt.date, dt.date, float]] = []
    read("periods.csv")
    for row in _read_csv(directory / "periods.csv"):
        tb3ms = _finite(row.get("tb3ms_annual_percent"))
        if tb3ms is None:
            fail(ErrorCode.SCHEMA_MISMATCH, f"missing TB3MS for {row.get('formation_date')}", "The risk-free input is never zero-filled.")
        periods.append((dt.date.fromisoformat(row["formation_date"]), dt.date.fromisoformat(row["period_end"]), tb3ms))
    liquidations: dict[dt.date, dict[str, LiquidationEvidence]] = {}
    raw_liq = read("liquidations.json", optional=True)
    if raw_liq:
        for record in json.loads(raw_liq.decode("utf-8")):
            record = dict(record)
            day = dt.date.fromisoformat(record.pop("period_formation_date"))
            evidence = LiquidationEvidence.model_validate(record)
            liquidations.setdefault(day, {})[evidence.security_id] = evidence
    raw_lineage = read("lineage.json", optional=True)
    lineage = json.loads(raw_lineage.decode("utf-8")) if raw_lineage else {}
    return FixtureData(config, cards, certificates, forecasts, market, outcomes, periods, liquidations, lineage, file_hashes)


LINEAGE_REQUIRED = (
    "security_master_version",
    "economic_graph_version",
    "calendar_policy",
    "timezone_policy",
    "data_access_policy_version",
    "evaluation_policy_version",
    "trial_count",
    "evaluation_access_count",
)

RiskOverlay = Callable[[Mapping[str, Decimal], dt.date], Mapping[str, Decimal]]


class ControlledReplay:
    """Small complete controlled path from approved forecasts to dossier hashes."""

    def __init__(
        self,
        fixture: FixtureData,
        out_dir: Path,
        *,
        research_only: bool = False,
        overwrite: bool = False,
        risk_overlay: Optional[RiskOverlay] = None,
        certificate_policy: "None | str | CertificatePolicy" = None,
        revocations: Optional[RevocationLedger] = None,
    ):
        self.fx = fixture
        self.cfg = fixture.config
        self.research_only = research_only
        # Explicit certificate policy: argument > config["certificate_policy"] > "fixture_compat".
        # fixture_compat accepts legacy free-text certificates for controlled-replay fixtures only
        # and labels them legacy_unverified; "production" rejects them (see research/certificates.py).
        self.cert_policy = resolve_policy(certificate_policy or self.cfg.get("certificate_policy"))
        self.revocations = revocations
        self.certificate_checks: dict[str, CertificateCheck] = {}
        self.writer = ArtifactWriter(out_dir, overwrite=overwrite)
        self.overlay = risk_overlay
        self.gate = SourceEligibilityGate(research_only)
        self.compliance = PeriodComplianceLog(OfficialLimits(**self.cfg.get("official_limits", {})))
        self.periods_log: list[dict] = []
        self.review_log: list[dict] = []
        self.actual_review_log: list[dict] = []

    # ------------------------------------------------------------- helpers
    def _risk_engine(self) -> tuple[RiskEngine, dict]:
        r = self.cfg["risk"]
        limits = [
            RiskLimit(name="gross", threshold=f"<= {r['max_gross']}", hard=True),
            RiskLimit(name="abs_net", threshold=f"<= {r['max_abs_net']}", hard=True),
            RiskLimit(name="max_name_weight", threshold=f"<= {r['max_name_weight']}", hard=True),
            RiskLimit(name="position_count", threshold=f">= {r['min_names']}", hard=True),
        ]
        if "max_abs_beta" in r:
            limits.append(RiskLimit(name="abs_beta", threshold=f"<= {r['max_abs_beta']}", hard=False))
        return RiskEngine(RiskPolicy(policy_version=r["policy_version"], limits=limits)), r

    def _measure(self, weights: Mapping[str, Decimal], rows: Mapping[str, MarketRow], r: dict) -> list[MeasuredExposure]:
        gross = float(sum((abs(w) for w in weights.values()), ZERO))
        net = float(sum(weights.values(), ZERO))
        max_name = float(max((abs(w) for w in weights.values()), default=ZERO))
        exposures = [
            MeasuredExposure("gross", gross, float(r["max_gross"]), True),
            MeasuredExposure("abs_net", abs(net), float(r["max_abs_net"]), True),
            MeasuredExposure("max_name_weight", max_name, float(r["max_name_weight"]), True),
            MeasuredExposure("position_count", float(len(weights)), float(r["min_names"]), False),
        ]
        if "max_abs_beta" in r:
            beta = 0.0
            for sid, w in weights.items():
                if sid not in rows:
                    fail(ErrorCode.SCHEMA_MISMATCH, f"no beta evidence for {sid}", "Risk needs measured exposures for every name.")
                beta += float(w) * rows[sid].beta
            exposures.append(MeasuredExposure("abs_beta", abs(beta), float(r["max_abs_beta"]), True))
        return exposures

    def _review_actual_fills(
        self,
        risk_engine: RiskEngine,
        risk_cfg: dict,
        reviewed: Any,
        post_weights: Mapping[str, Decimal],
        rows: Mapping[str, MarketRow],
        period: str,
        decision_at: dt.datetime,
    ) -> Any:
        """Re-apply the independent risk policy to actual post-fill weights; fail closed.

        Weights are the ledger's post-trade weights over the held-name union, so partial or
        uneven fills are measured as they are. Each attempt is logged (including failures)
        in ``actual_review_log`` before any exception propagates.
        """
        weights = {k: _dec(v) for k, v in post_weights.items() if _dec(v) != ZERO}
        entry: dict = {"period": period, "stage": "actual_fill", "reviewed_kind": "actual", "post_weights": {k: str(v) for k, v in sorted(weights.items())}}
        self.actual_review_log.append(entry)
        try:
            book = reviewed.model_copy(
                update={
                    "target_book_id": f"{reviewed.target_book_id}_actualfill",
                    "positions": [TargetPosition(security_id=k, target_weight=v) for k, v in sorted(weights.items())],
                }
            )
            review = risk_engine.review(
                book, self._measure(weights, rows, risk_cfg), effective_at=decision_at, reviewed_kind="actual"
            )
        except Exception as error:
            entry.update({"decision": "INVALID", "error": f"{type(error).__name__}: {error}"})
            fail(ErrorCode.RISK_REVIEW_REQUIRED, f"actual-fill risk review could not be produced for {period}: {type(error).__name__}", "Execution results need a valid independent review; the period is not accepted.")
        entry.update(review.model_dump(mode="json"))
        if review.decision in (RiskDecision.BLOCK, RiskDecision.INFEASIBLE):
            fail(ErrorCode.PORTFOLIO_INFEASIBLE, f"independent risk review blocked actual fills {period}: {'; '.join(review.remedy_constraints)}", "Actual post-fill weights must satisfy the same independent risk policy; the period is not accepted.")
        if not review.is_valid_at(decision_at):
            fail(ErrorCode.RISK_REVIEW_REQUIRED, f"actual-fill risk review for {period} is not valid at decision time", "Execution results need a valid review.")
        return review

    def _active_forecasts(self, decision_at: dt.datetime) -> list[Forecast]:
        best: dict[tuple[str, str], Forecast] = {}
        for f in self.fx.forecasts:
            # created_at is the publication time: a forecast published after the decision
            # is future information even if its information_cutoff is old.
            if (
                f.created_at > decision_at
                or f.information_cutoff > decision_at
                or f.earliest_formation_at > decision_at
                or f.is_expired(decision_at)
            ):
                continue
            key = (f.security_id, f.strategy_id)
            rank = (f.information_cutoff, f.created_at, f.forecast_id)
            if key not in best or rank > (best[key].information_cutoff, best[key].created_at, best[key].forecast_id):
                best[key] = f
        return [best[k] for k in sorted(best)]

    def _verify_certificate(self, forecast: Forecast, decision_at: dt.datetime) -> AdmissionCertificate:
        cert = self.fx.certificates.get(forecast.admission_certificate_id)
        snapshots = {
            source_id: self.fx.cards[source_id].snapshot_id
            for source_id in forecast.source_family_ids
            if source_id in self.fx.cards
        }
        # Only sealed certificates outside research-only runs; a research-only run needs no
        # formation-grade certificate (and can never freeze), so it accepts any environment.
        policy = self.cert_policy.replace(require_seal=self.cert_policy.require_seal and not self.research_only)
        check = validate_certificate(
            forecast,
            cert,
            decision_at=decision_at,
            source_snapshots=snapshots,
            universe=self.cfg.get("universe_id"),
            namespace=self.cfg.get("namespace"),
            policy=policy,
            environment=CertificateEnvironment.TEST if self.research_only else None,
            revocations=self.revocations,
        )
        self.certificate_checks[check.certificate_id] = check
        return cert

    def _check_certificate_limits(self, active: Sequence[Forecast], post_risk: Mapping[str, Decimal], nav: Decimal) -> None:
        """Book-level scope limits (declared numeric limits compare against the whole book)."""
        weights = [_dec(w) for w in post_risk.values() if _dec(w) != ZERO]
        exposures = {
            "gross": sum((abs(w) for w in weights), ZERO),
            "net": sum(weights, ZERO),
            "max_name_weight": max((abs(w) for w in weights), default=ZERO),
            "nav_usd": nav,
            "position_count": len(weights),
        }
        for certificate_id in sorted({f.admission_certificate_id for f in active}):
            check_scope_limits(self.fx.certificates[certificate_id], exposures)

    # ----------------------------------------------------------------- run
    def run(self) -> dict:
        try:
            return self._run()
        except PodPlatformError as error:
            self._write_failure(str(error.response.error.code.value), error.response.summary)
            raise
        except Exception as error:
            self._write_failure("UNTYPED_EXCEPTION", f"{type(error).__name__}: {error}")
            raise

    def _write_failure(self, code: str, summary: str) -> None:
        body = {
            "status": "FAILED_CLOSED",
            "error_code": code,
            "summary": summary,
            "periods_completed": [p["period"] for p in self.periods_log],
            "frozen": False,
            "run_label": "RESEARCH_ONLY_NOT_FREEZABLE" if self.research_only else "FORMATION_ELIGIBLE_ATTEMPT",
        }
        self.writer.write_json("failure.json", body)
        self.writer.write_json("compliance_report.partial.json", self.compliance.as_dict())
        self.writer.write_json("periods.partial.json", self.periods_log)
        self.writer.write_json("risk_reviews.partial.json", self.review_log)
        self.writer.write_json("risk_reviews_actual_fill.partial.json", self.actual_review_log)

    def _run(self) -> dict:
        cfg = self.cfg
        fx = self.fx
        periods = fx.periods
        validate_contiguous_periods([(a, b) for a, b, _ in periods])
        if cfg.get("run_kind") not in ("replay", "trial"):
            fail(ErrorCode.SCHEMA_MISMATCH, "config.run_kind must be 'replay' or 'trial'", "Runs carry an explicit trial/replay label.")

        official = OfficialLimits(**cfg.get("official_limits", {}))
        engine = ConstructionEngine(official, InternalPolicy(**cfg.get("internal_policy", {})))
        integrator = FamilyIntegrator(cfg["admitted_families"], neutral_uncertainty=cfg.get("neutral_uncertainty", 0.1))
        cost_model = CostModel(CostModelConfig(**cfg["cost_model"]))
        planner = ExecutionPlanner(ExecutionPolicy(**cfg["execution_policy"]), cost_model)
        fills_engine = FillEngine(cost_model)
        risk_engine, risk_cfg = self._risk_engine()
        window_days = int(cfg.get("fill_window_days", 5))
        borrow = cfg.get("borrow", {})
        schedule = BorrowSchedule(
            schedule_id=borrow.get("schedule_id", "controlled_flat"),
            large_cap_threshold_usd=Decimal(str(borrow.get("large_cap_threshold_usd", "1e10"))),
            large_cap_annual_rate=Decimal(str(borrow.get("large_cap_annual_rate", "0.01"))),
            other_annual_rate=Decimal(str(borrow.get("other_annual_rate", "0.05"))),
        )
        replay = MonthlyAccountingReplay(Decimal(str(cfg["starting_nav"])), borrow_schedule=schedule)
        core = [("core", sid, "core") for sid in cfg.get("core_sources", [])]
        gate_uses: dict[tuple[str, str], SourceUse] = {}
        dossier = DossierBuilder()

        caps: dict[str, Decimal] = {}
        monthly: list[dict] = []
        holdings_rows: list[list] = []
        final_rows: list[HoldingsRow] = []
        ever_held: dict[str, dt.datetime] = {}
        certificates_used: dict[str, AdmissionCertificate] = {}
        forecasts_used: dict[str, Forecast] = {}

        for formation, period_end, tb3ms in periods:
            period = formation.isoformat()
            decision_at = utc_close(formation)
            end_at = utc_close(period_end)
            rows = fx.market.get(formation)
            if not rows:
                fail(ErrorCode.SCHEMA_MISMATCH, f"no market data for formation {period}", "Periods are never skipped.")
            for sid, row in rows.items():
                caps[sid] = row.market_cap_usd
            active = self._active_forecasts(decision_at)
            if not active:
                fail(ErrorCode.SCHEMA_MISMATCH, f"no active approved forecasts at {period}", "A decision cycle without forecasts cannot construct a book.")
            uses = list(self.gate.resolve(core, fx.cards))
            for f in active:
                if f.strategy_id not in integrator.families:
                    fail(ErrorCode.ADMISSION_CERTIFICATE_INVALID, f"strategy {f.strategy_id} is not an admitted family", "Only admitted families are integrated.")
                self._verify_certificate(f, decision_at)
                uses += self.gate.resolve([(f.strategy_id, s, "external") for s in f.source_family_ids], fx.cards)
            gate_result = self.gate.evaluate(uses)
            for use in uses:
                gate_uses[(use.name, use.source_id)] = use
            for f in active:
                certificates_used[f.admission_certificate_id] = fx.certificates[f.admission_certificate_id]
                forecasts_used[f.forecast_id] = f

            by_security: dict[str, list[Forecast]] = {}
            for f in active:
                if f.security_id in rows:
                    by_security.setdefault(f.security_id, []).append(f)
            held_before = replay.held()
            candidates: list[Candidate] = []
            alphas: dict[str, float] = {}
            for sid, fcs in sorted(by_security.items()):
                integrated = integrator.integrate(
                    [FamilyEstimate(f.strategy_id, sid, float(f.return_forecast.expected_excess_return), cfg.get("neutral_uncertainty", 0.1)) for f in fcs]
                )
                alphas[sid] = integrated.alpha
                row = rows[sid]
                candidates.append(
                    Candidate(
                        security_id=sid,
                        alpha=integrated.alpha,
                        market_cap_usd=float(row.market_cap_usd),
                        tradable=row.tradable,
                        incumbent=sid in held_before,
                        sector=row.sector,
                        beta=row.beta,
                    )
                )
            forecast_ids = sorted({f.forecast_id for fcs in by_security.values() for f in fcs})
            book = engine.build(
                candidates,
                run_id=f"{cfg['run_id']}_{period}",
                decision_at=decision_at,
                forecast_publication_ids=forecast_ids,
                current_book_snapshot_id=f"book_snapshot_{period}",
                nav_snapshot_id=f"nav_{period}",
                price_snapshot_id=f"px_{period}",
                risk_envelope_version=risk_cfg["policy_version"],
                covariance_version=cfg.get("covariance_version", "none_declared"),
                cost_model_version=cost_model.model_id,
                attribution_method="family_equal_contribution",
            )
            target = {p.security_id: p.target_weight for p in book.positions}
            drifted = replay.actual_weights()
            self.compliance.record(period, "formation_target", drifted, target, caps)

            post_risk = dict(target)
            if self.overlay is not None:
                post_risk = {k: _dec(v) for k, v in self.overlay(target, formation).items()}
            reviewed = book
            if post_risk != target:
                reviewed = book.model_copy(
                    update={
                        "target_book_id": f"{book.target_book_id}_postrisk",
                        "positions": [TargetPosition(security_id=k, target_weight=v) for k, v in sorted(post_risk.items())],
                    }
                )
            review = risk_engine.review(reviewed, self._measure(post_risk, rows, risk_cfg), effective_at=decision_at)
            self.review_log.append({"period": period, **review.model_dump(mode="json")})
            self.compliance.record(period, "post_risk", target, post_risk, caps)
            self._check_certificate_limits(active, post_risk, replay.nav())
            if review.decision in (RiskDecision.BLOCK, RiskDecision.INFEASIBLE):
                fail(ErrorCode.PORTFOLIO_INFEASIBLE, f"independent risk review blocked {period}: {'; '.join(review.remedy_constraints)}", "A blocked book is never traded.")
            if not review.is_valid_at(decision_at):
                fail(ErrorCode.RISK_REVIEW_REQUIRED, f"risk review for {period} is not valid at decision time", "Execution needs a valid review.")

            # ---- planner -> fills -> ledger (held-name union kept)
            nav_open = replay.nav()
            union = sorted(set(post_risk) | held_before)
            liquidity: dict[str, LiquiditySnapshot] = {}
            desires: list[DesiredTrade] = []
            for sid in union:
                row = rows.get(sid)
                if row is None:
                    fail(ErrorCode.LIQUIDITY_UNAVAILABLE, f"no liquidity evidence at {period} for {sid}", "Held names need dated liquidity evidence to be traded.")
                liquidity[sid] = LiquiditySnapshot(sid, row.adv_dollars, row.daily_volatility, row.size_tier, row.price)
                target_w = post_risk.get(sid, ZERO)
                current_w = drifted.get(sid, ZERO)
                if target_w == current_w:
                    continue
                desires.append(
                    DesiredTrade(
                        security_id=sid,
                        side=OrderSide.BUY if target_w > current_w else OrderSide.SELL,
                        target_weight=target_w,
                        current_weight=current_w,
                        nav=nav_open,
                        declared_edge_bp=Decimal(str(abs(alphas.get(sid, 0.0)) * 10000.0)),
                    )
                )
            orders = planner.plan(desires, liquidity, decision_at=decision_at, target_book_id=reviewed.target_book_id)
            instructions: list[FillInstruction] = []
            order_log: list[dict] = []
            for order in orders:
                state = fills_engine.open(order)
                for day in range(window_days):
                    if not state.pending:
                        break
                    last_day = day == order.days_to_complete - 1
                    fills_engine.fill_day(
                        state,
                        liquidity[order.security_id],
                        fill_dollars=None if last_day else order.daily_fill_dollars,
                    )
                if state.pending:
                    fills_engine.expire(state, decision_at)
                fills_engine.assert_no_assumed_holdings([state])
                sign = ONE if order.order_dollars > ZERO else -ONE
                complete = state.remaining_dollars == ZERO
                order_log.append(
                    {
                        "order_id": order.order_id,
                        "security_id": order.security_id,
                        "order_dollars": str(order.order_dollars),
                        "filled_dollars": str(state.filled_dollars),
                        "unfilled_dollars": format(state.remaining_dollars.normalize(), "f"),
                        "expired": state.expired,
                        "cost": str(state.costs_paid),
                    }
                )
                if state.filled_dollars == ZERO:
                    continue
                instructions.append(
                    FillInstruction(
                        security_id=order.security_id,
                        signed_dollars=sign * state.filled_dollars,
                        fee=state.costs_paid,
                        close_out=complete and post_risk.get(order.security_id, ZERO) == ZERO,
                        entry_price=rows[order.security_id].price,
                        order_id=f"_{order.order_id}",
                    )
                )
            trade = replay.apply_fills(instructions, at=decision_at, tag="fill")
            self.compliance.record(period, "post_trade_actual", trade.pre_weights, trade.post_weights, caps)
            # same independent policy, applied to what was actually filled (held-name union incl. unfilled legs)
            actual_review = self._review_actual_fills(
                risk_engine, risk_cfg, reviewed, trade.post_weights, rows, period, decision_at
            )

            # ---- outcome resolution happens only after the trade
            outcomes = fx.outcomes.get(formation, {})
            resolved = resolve_outcomes(
                replay.held(), outcomes, fx.liquidations.get(formation, {}), formation_at=decision_at, period_end_at=end_at
            )
            result = replay.advance(period_end_at=end_at, rf_monthly=tb3ms / 1200.0, resolved=resolved, market_caps=caps)
            self.compliance.record(period, "period_end_drifted", trade.post_weights, result.drifted_weights, caps)

            for sid, weight in sorted(trade.post_weights.items()):
                holdings_rows.append([period, sid, str(weight), "long" if weight > ZERO else "short"])
                ever_held[sid] = decision_at
            monthly.append(
                {
                    "period": period,
                    "period_end": period_end.isoformat(),
                    **result.as_dict(),
                    "turnover_one_way": trade.turnover_one_way,
                    "trading_fees": str(trade.fees),
                    "traded_dollars": str(trade.traded_dollars),
                    "benchmark_monthly": benchmark_monthly(tb3ms),
                }
            )
            self.periods_log.append(
                {
                    "period": period,
                    "forecast_ids": forecast_ids,
                    "certificate_ids": sorted({f.admission_certificate_id for f in active}),
                    "target_book_id": book.target_book_id,
                    "target_book_digest": book.digest(),
                    "risk_review_id": review.risk_review_id,
                    "risk_review_digest": review.digest(),
                    "risk_decision": review.decision.value,
                    "actual_fill_risk_review_id": actual_review.risk_review_id,
                    "actual_fill_risk_review_digest": actual_review.digest(),
                    "actual_fill_risk_decision": actual_review.decision.value,
                    "orders": order_log,
                    "trades": trade.trades,
                    "outcome_basis": resolved.basis,
                    "source_gate": gate_result.as_dict(),
                }
            )
            for sid in sorted(trade.post_weights):
                for f in by_security.get(sid, []):
                    dossier.record(
                        _event(f"forecast_{period}_{sid}_{f.strategy_id}", decision_at, f"admit forecast {f.forecast_id} for {sid}", f.forecast_id, {f.forecast_id: f.digest(), f.admission_certificate_id: certificates_used[f.admission_certificate_id].digest()}, f.admission_certificate_id)
                    )
                dossier.record(_event(f"target_{period}_{sid}", decision_at, f"target position in {sid} from {reviewed.target_book_id}", reviewed.target_book_id, {reviewed.target_book_id: book.digest()}, "rule_deterministic_construction_v1"))
                dossier.record(_event(f"review_{period}_{sid}", decision_at, f"risk review {review.risk_review_id} for {sid}", review.risk_review_id, {review.risk_review_id: review.digest()}, risk_cfg["policy_version"]))
            for o in order_log:
                if o["security_id"] in trade.post_weights:
                    dossier.record(_event(f"fill_{period}_{o['order_id']}", decision_at, f"fill order for {o['security_id']}", f"fill_{o['order_id']}", {o["order_id"]: sha256_hex(dumps(o).encode())}, cost_model.model_id))

        last_formation = periods[-1][0]
        last_rows = fx.market[last_formation]
        for sid, weight in sorted(replay.actual_weights().items()):
            row = last_rows.get(sid)
            if row is not None:
                position = replay.ledger.positions[sid]
                final_rows.append(
                    HoldingsRow(sid, row.ticker, row.name, row.label_source, row.label_reference_date, float(position.quantity), float(weight), "long" if weight > ZERO else "short")
                )
        return self._finalize(
            replay, monthly, holdings_rows, final_rows, dossier, ever_held, gate_uses, certificates_used, forecasts_used, periods, last_rows
        )

    # ------------------------------------------------------------ outputs
    def _finalize(self, replay, monthly, holdings_rows, final_rows, dossier, ever_held, gate_uses, certificates_used, forecasts_used, periods, last_rows) -> dict:
        cfg = self.cfg
        w = self.writer
        gate = self.gate.evaluate(list(gate_uses.values()))
        n = len(monthly)
        table: Optional[dict] = None
        table_note = None
        if n >= 2:
            table = RequiredTable(
                ending_value=float(replay.nav()),
                starting_value=float(replay.ledger.starting_cash),
                elapsed_years=n / 12.0,
                periods_per_year=12,
                portfolio_returns=tuple(m["net_return"] for m in monthly),
                benchmark_returns=tuple(m["benchmark_monthly"] for m in monthly),
                risk_free_returns=tuple(m["rf_monthly"] for m in monthly),
                gross_returns=tuple(m["gross_return"] for m in monthly),
                cost_returns=tuple(m["cost_return"] for m in monthly),
                borrow_returns=tuple(m["borrow_return"] for m in monthly),
                position_counts=tuple(m["names"] for m in monthly),
                gross_exposures=tuple(m["gross_exposure"] for m in monthly),
                net_exposures=tuple(m["net_exposure"] for m in monthly),
                turnover_per_rebalance=tuple(m["turnover_one_way"] for m in monthly),
            ).as_table()
        else:
            table_note = "fewer than two periods: ratio statistics are undefined"
        labels = {
            "run_kind": cfg["run_kind"],
            "evidence_class": "controlled_fixture_replay",
            "frequency": "monthly",
            "return_basis": "net of modeled trading cost and borrow on ledger NAV",
        }
        w.write_json("monthly.json", monthly)
        w.write_json("performance_report.json", {"labels": labels, "table": table, "note": table_note, "limitations": list(MONTHLY_LIMITATIONS)})
        w.write_json("compliance_report.json", self.compliance.as_dict())
        w.write_json("risk_reviews.json", self.review_log)
        w.write_json("risk_reviews_actual_fill.json", self.actual_review_log)
        w.write_json("periods.json", self.periods_log)
        w.write_text("returns_export.csv", returns_csv([(m["period"], m["net_return"]) for m in monthly]))
        w.write_text("holdings_export.csv", holdings_csv(final_rows))
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["period", "security_id", "post_trade_weight", "side"])
        writer.writerows(holdings_rows)
        w.write_text("holdings_by_period.csv", buffer.getvalue())
        explanations = {sid: dossier.build(sid, at).explanation() for sid, at in sorted(ever_held.items())}
        w.write_json("decision_dossier.json", explanations)

        lineage = self.fx.lineage
        missing_lineage = [k for k in LINEAGE_REQUIRED if lineage.get(k) in (None, "")]
        config_hashes = {
            "config": canonical_digest(cfg),
            **{f"fixture:{name}": digest for name, digest in sorted(self.fx.file_hashes.items())},
        }
        manifest: Optional[RunManifest] = None
        if not self.research_only and not missing_lineage and gate.formation_inputs_approved:
            manifest = RunManifest(
                run_id=cfg["run_id"],
                code_commit_or_tree_hash=tree_hash(Path(__file__).resolve().parent.parent),
                environment_lock_hash=environment_lock_hash(),
                config_hashes=config_hashes,
                source_snapshot_ids={u["source_id"]: u["card_snapshot_id"] for u in gate.uses},
                security_master_version=lineage["security_master_version"],
                economic_graph_version=lineage["economic_graph_version"],
                model_versions={f.model_id: c.model_hash for f in forecasts_used.values() for c in [certificates_used[f.admission_certificate_id]]},
                calendar_policy=lineage["calendar_policy"],
                timezone_policy=lineage["timezone_policy"],
                data_access_policy_version=lineage["data_access_policy_version"],
                evaluation_policy_version=lineage["evaluation_policy_version"],
                initial_accounting_state={"starting_nav": str(replay.ledger.starting_cash), "currency": "USD"},
                expected_outputs={name: name for name in sorted(w.hashes)},
            )
            w.write_json("run_manifest.json", manifest.model_dump(mode="json"))

        certificate_count = sum(1 for f in forecasts_used.values() if certificates_used[f.admission_certificate_id].verify_seal())
        frozen_at = cfg.get("frozen_at")
        frozen_at = dt.datetime.fromisoformat(frozen_at) if frozen_at else utc_close(periods[-1][1])
        freeze_outputs = {
            "holdings_export": w.payloads["holdings_export.csv"],
            "returns_export": w.payloads["returns_export.csv"],
            "performance_report": w.payloads["performance_report.json"],
            "compliance_report": w.payloads["compliance_report.json"],
            "decision_dossier": w.payloads["decision_dossier.json"],
        }
        assessment = assess_freeze(
            manifest=manifest,
            outputs=freeze_outputs,
            gate=gate,
            certificate_count=certificate_count,
            deployed_forecast_count=len(forecasts_used),
            accounting_reconciled=True,  # advance() raises on any residual, so reaching here is measured
            compliance=self.compliance,
            trial_count=int(lineage.get("trial_count", 0) or 0),
            evaluation_access_count=int(lineage["evaluation_access_count"]) if lineage.get("evaluation_access_count") is not None else -1,
            frozen_at=frozen_at,
        )
        w.write_json("freeze_assessment.json", assessment.as_dict())
        summary = {
            "run_id": cfg["run_id"],
            "labels": labels,
            "source_policy": gate.as_dict(),
            "environment": {"environment_lock_hash": environment_lock_hash(), "python": sys.version.split()[0]},
            "periods": n,
            "ending_nav": str(replay.nav()),
            "freeze_status": assessment.status,
            "freeze_reasons": assessment.reasons,
            "missing_lineage": missing_lineage,
            "internal_warnings": self.compliance.internal_warnings(),
            "limitations": list(MONTHLY_LIMITATIONS),
            "llm_called": False,
            "certificate_policy": self.cert_policy.name,
            "certificate_checks": [self.certificate_checks[k].as_dict() for k in sorted(self.certificate_checks)],
            # a controlled fixture replay never authorizes a real competition book, whatever the certificates say
            "certificates_authorize_competition_book": False,
        }
        w.write_json("run_summary.json", summary)
        w.write_json("artifact_hashes.json", dict(sorted(w.hashes.items())))
        return {"summary": summary, "assessment": assessment, "hashes": dict(w.hashes)}


def _event(event_id: str, at: dt.datetime, action: str, result_id: str, inputs: dict[str, str], authority: str) -> DecisionEvent:
    return DecisionEvent(
        owner="orchestration",
        created_at=at,
        event_id=event_id,
        actor_kind=ActorKind.CODE,
        actor_id="controlled_replay",
        effective_research_at=at,
        effective_market_at=at,
        input_artifact_ids=sorted(inputs),
        input_hashes=dict(inputs),
        chosen_action=action,
        rationale="deterministic controlled replay step",
        authorizing_rule_or_certificate_id=authority,
        resulting_artifact_id=result_id,
    )
