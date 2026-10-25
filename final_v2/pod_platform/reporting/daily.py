"""Daily reporting adapter (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 9.9, 19.2, 19.5-19.8, 26.4, 26.7).

Consumes *actual dated observations only*: closing prices or total returns per
security, dated fills with explicit fees, dated cash/borrow entries, the dated
TB3MS series and (optionally) dated S&P total returns. Nothing is inferred from
monthly data and nothing is interpolated:

* the trading calendar is explicit; observations on a non-trading day are
  rejected (or ignored *and recorded* under an explicit policy) and fills on a
  non-trading day are always rejected;
* a held security with no supplied outcome on a trading day fails closed, unless
  dated ``SuspenseEvidence`` documents a halt - then the last mark is carried
  (bounded by ``max_suspense_days``), the report says ``exact_nav = False`` and
  the position stays visible;
* every day's NAV is reconciled by two routes (state recomputation versus the
  flow identity ``NAV_t = NAV_{t-1} + position P&L + cash income - costs``);
* the benchmark is recomputed from TB3MS with the brief's daily convention
  ``TB3MS/25200 + 0.04/252`` and compared with any reported series.

Returns are *total* returns (no dividend is added on top). Borrow, funding and
other costs are the dated cash entries the caller supplies; the adapter does not
invent a borrow schedule.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Mapping, Optional, Sequence

from pydantic import Field, field_validator, model_validator

from ..accounting.returns import (
    annualized_arithmetic,
    annualized_volatility,
    benchmark_daily,
    cagr,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
)
from ..contracts.base import StrictModel
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse

ZERO = Decimal("0")
ONE = Decimal("1")
CASH_KINDS = ("borrow", "funding", "other_fee", "dividend_credit", "other")


def _fail(code: ErrorCode, summary: str, stop: str) -> None:
    raise PodPlatformError(ToolResponse.failure(summary=summary, code=code, stop_condition=stop))


def _dec(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


# ------------------------------------------------------------------- records


class DailyReturn(StrictModel):
    security_id: str = Field(min_length=1)
    on: dt.date
    total_return: Decimal  # decimal total return for the day (not percent, not bp)

    @model_validator(mode="after")
    def _check(self) -> "DailyReturn":
        if not self.total_return.is_finite() or self.total_return <= Decimal("-1"):
            raise ValueError("a daily total return must be finite and above -100%")
        return self


class DailyPrice(StrictModel):
    security_id: str = Field(min_length=1)
    on: dt.date
    close: Decimal  # total-return-adjusted close

    @model_validator(mode="after")
    def _check(self) -> "DailyPrice":
        if not self.close.is_finite() or self.close <= ZERO:
            raise ValueError("a close price must be finite and positive")
        return self


class DailyFill(StrictModel):
    """A fill at the close of ``on``. ``quantity`` is signed: buy/cover > 0, sell/short < 0."""

    security_id: str = Field(min_length=1)
    on: dt.date
    quantity: Decimal
    price: Decimal
    fee_usd: Decimal = ZERO  # explicit modeled cost (spread, impact, fees) in USD, >= 0

    @model_validator(mode="after")
    def _check(self) -> "DailyFill":
        if not (self.quantity.is_finite() and self.price.is_finite() and self.fee_usd.is_finite()):
            raise ValueError("fills must be finite")
        if self.quantity == ZERO or self.price <= ZERO or self.fee_usd < ZERO:
            raise ValueError("fill quantity must be non-zero, price positive and fee non-negative")
        return self


class DailyCashEntry(StrictModel):
    """Dated cash movement with explicit direction; amounts are USD and never negative."""

    on: dt.date
    kind: Literal["borrow", "funding", "other_fee", "dividend_credit", "other"]
    debit_usd: Decimal = ZERO
    credit_usd: Decimal = ZERO
    evidence_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "DailyCashEntry":
        if not (self.debit_usd.is_finite() and self.credit_usd.is_finite()):
            raise ValueError("cash entries must be finite")
        if self.debit_usd < ZERO or self.credit_usd < ZERO:
            raise ValueError("debit and credit amounts are non-negative; direction is the field name")
        if self.debit_usd == ZERO and self.credit_usd == ZERO:
            raise ValueError("a cash entry needs a debit or a credit")
        return self


class DailyRate(StrictModel):
    on: dt.date
    tb3ms_annual_percent: float  # annualized PERCENT, e.g. 5.04 for 5.04%

    @field_validator("tb3ms_annual_percent")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("TB3MS must be finite")
        return value


class DailyMarket(StrictModel):
    on: dt.date
    total_return: float

    @field_validator("total_return")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value) or value <= -1:
            raise ValueError("market return must be finite and above -100%")
        return value


class SuspenseEvidence(StrictModel):
    security_id: str = Field(min_length=1)
    on: dt.date
    evidence_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class DailyPolicy(StrictModel):
    trading_days: list[dt.date] = Field(min_length=2)
    non_trading_observations: Literal["reject", "ignore_recorded"] = "reject"
    max_suspense_days: int = Field(default=5, ge=0)
    tolerance: float = Field(default=1e-6, gt=0)
    periods_per_year: int = Field(default=252, ge=1)
    beta_min_observations: int = Field(default=20, ge=3)
    rolling_window: int = Field(default=252, ge=3)

    @model_validator(mode="after")
    def _check(self) -> "DailyPolicy":
        days = self.trading_days
        if any(b <= a for a, b in zip(days, days[1:])):
            raise ValueError("trading_days must be strictly increasing")
        return self


@dataclass
class DailyInputs:
    start_cash_usd: Decimal
    initial_positions: Mapping[str, tuple[Decimal, Decimal]] = field(default_factory=dict)  # id -> (quantity, price)
    returns: Sequence[DailyReturn] = ()
    prices: Sequence[DailyPrice] = ()
    fills: Sequence[DailyFill] = ()
    cash_entries: Sequence[DailyCashEntry] = ()
    risk_free: Sequence[DailyRate] = ()
    market_returns: Sequence[DailyMarket] = ()
    suspense: Sequence[SuspenseEvidence] = ()
    reported_benchmark: Optional[Mapping[dt.date, float]] = None


# --------------------------------------------------------------------- stats


def ols_beta(excess_portfolio: Sequence[float], excess_market: Sequence[float]) -> dict[str, Optional[float]]:
    """OLS of portfolio excess return on market excess return with classical standard errors."""
    n = len(excess_portfolio)
    if n != len(excess_market):
        raise ValueError("series length mismatch")
    if n < 3:
        raise ValueError("beta needs at least three observations")
    mx = sum(excess_market) / n
    my = sum(excess_portfolio) / n
    sxx = sum((x - mx) ** 2 for x in excess_market)
    if sxx <= 0:
        raise ValueError("market excess return has no dispersion; beta is undefined")
    sxy = sum((x - mx) * (y - my) for x, y in zip(excess_market, excess_portfolio))
    beta = sxy / sxx
    alpha = my - beta * mx
    residuals = [y - alpha - beta * x for x, y in zip(excess_market, excess_portfolio)]
    sse = sum(r * r for r in residuals)
    sigma2 = sse / (n - 2)
    se_beta = math.sqrt(sigma2 / sxx)
    se_alpha = math.sqrt(sigma2 * (1.0 / n + mx * mx / sxx))
    syy = sum((y - my) ** 2 for y in excess_portfolio)
    r2 = 1.0 - sse / syy if syy > 0 else None
    return {"beta": beta, "alpha": alpha, "se_beta": se_beta, "se_alpha": se_alpha, "r2": r2, "n": float(n)}


def daily_beta(
    portfolio_returns: Sequence[float],
    market_returns: Sequence[float],
    risk_free_daily: Sequence[float],
    *,
    periods_per_year: int = 252,
) -> dict[str, Optional[float]]:
    """Daily beta/alpha of portfolio versus market, both in excess of the daily risk-free rate."""
    if not (len(portfolio_returns) == len(market_returns) == len(risk_free_daily)):
        raise ValueError("series length mismatch")
    y = [p - r for p, r in zip(portfolio_returns, risk_free_daily)]
    x = [m - r for m, r in zip(market_returns, risk_free_daily)]
    stats = ols_beta(y, x)
    stats["annualized_alpha"] = stats["alpha"] * periods_per_year  # type: ignore[operator]
    stats["annualized_alpha_se"] = stats["se_alpha"] * periods_per_year  # type: ignore[operator]
    return stats


def rolling_beta(
    dates: Sequence[dt.date],
    portfolio_returns: Sequence[float],
    market_returns: Sequence[float],
    risk_free_daily: Sequence[float],
    window: int,
) -> list[dict[str, Any]]:
    """Rolling daily beta over a trailing window of ``window`` observations (brief: 252)."""
    if not (len(dates) == len(portfolio_returns) == len(market_returns) == len(risk_free_daily)):
        raise ValueError("series length mismatch")
    out: list[dict[str, Any]] = []
    for end in range(window, len(dates) + 1):
        sl = slice(end - window, end)
        try:
            stats = daily_beta(portfolio_returns[sl], market_returns[sl], risk_free_daily[sl])
        except ValueError:
            continue
        out.append({"date": dates[end - 1].isoformat(), "beta": stats["beta"], "se_beta": stats["se_beta"], "window": window})
    return out


def compound_periods(
    dates: Sequence[dt.date],
    returns: Sequence[float],
    periods: Sequence[tuple[str, dt.date, dt.date]],
) -> list[dict[str, Any]]:
    """Compound daily returns into (label, start_exclusive, end_inclusive) periods.

    A formation at the close of ``start`` earns the returns of days ``start < d <= end``.
    A period containing no daily observation is reported with ``days = 0`` and no return.
    """
    out: list[dict[str, Any]] = []
    for label, start, end in periods:
        growth = 1.0
        days = 0
        for d, r in zip(dates, returns):
            if start < d <= end:
                growth *= 1.0 + r
                days += 1
        out.append({"period": label, "start": start.isoformat(), "end": end.isoformat(), "days": days,
                    "compounded_return": growth - 1.0 if days else None})
    return out


def reconcile_to_periods(
    daily: "DailyReport",
    reported: Sequence[tuple[str, dt.date, dt.date, float]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Compare compounded daily returns with separately reported period returns (spec 26.7).

    The comparison is a *check*; it never replaces the daily series. Differences within
    ``tolerance`` are reported as reconciled, otherwise as mismatches (not silently absorbed).
    """
    compounded = compound_periods(
        [row["date_obj"] for row in daily.rows], [row["return"] for row in daily.rows], [(p, s, e) for p, s, e, _ in reported]
    )
    rows = []
    for (label, _s, _e, value), c in zip(reported, compounded):
        diff = None if c["compounded_return"] is None else c["compounded_return"] - value
        rows.append({"period": label, "reported": value, "compounded_daily": c["compounded_return"], "days": c["days"],
                     "difference": diff, "reconciled": diff is not None and abs(diff) <= tolerance})
    return {"periods": rows, "all_reconciled": bool(rows) and all(r["reconciled"] for r in rows), "tolerance": tolerance}


# -------------------------------------------------------------------- report


@dataclass
class DailyReport:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def nav_path(self) -> list[float]:
        return [float(r["nav"]) for r in self.rows]

    @property
    def returns(self) -> list[float]:
        return [r["return"] for r in self.rows]

    def as_dict(self) -> dict[str, Any]:
        clean_rows = [{k: v for k, v in r.items() if k != "date_obj"} for r in self.rows]
        return {"summary": self.summary, "rows": clean_rows}


def reconcile_daily(inputs: DailyInputs, policy: DailyPolicy) -> DailyReport:
    """Run the daily ledger over the explicit calendar and reconcile it. Fails closed."""
    days = list(policy.trading_days)
    day_set = set(days)
    tol = Decimal(str(policy.tolerance))
    ignored: list[dict[str, str]] = []

    def filter_dated(items: Sequence[Any], label: str, allow_ignore: bool = True) -> list[Any]:
        kept = []
        for item in items:
            if item.on in day_set:
                kept.append(item)
            elif allow_ignore and policy.non_trading_observations == "ignore_recorded":
                ignored.append({"kind": label, "on": item.on.isoformat()})
            else:
                _fail(
                    ErrorCode.ASOF_VIOLATION,
                    f"{label} dated {item.on.isoformat()} is not on the explicit trading calendar",
                    "Observations on non-trading days are rejected (or ignored and recorded under an explicit policy).",
                )
        return kept

    returns = filter_dated(inputs.returns, "return")
    prices = filter_dated(inputs.prices, "price")
    fills = filter_dated(inputs.fills, "fill", allow_ignore=False)
    entries = filter_dated(inputs.cash_entries, "cash entry")
    rates = filter_dated(inputs.risk_free, "risk-free rate")
    markets = filter_dated(inputs.market_returns, "market return")
    suspense = filter_dated(inputs.suspense, "suspense evidence")

    ret_map: dict[tuple[str, dt.date], Decimal] = {}
    for r in returns:
        key = (r.security_id, r.on)
        if key in ret_map:
            _fail(ErrorCode.SCHEMA_MISMATCH, f"duplicate return for {key[0]} on {key[1]}", "One observation per security and day.")
        ret_map[key] = r.total_return
    px_map: dict[tuple[str, dt.date], Decimal] = {}
    for p in prices:
        key = (p.security_id, p.on)
        if key in px_map:
            _fail(ErrorCode.SCHEMA_MISMATCH, f"duplicate price for {key[0]} on {key[1]}", "One observation per security and day.")
        px_map[key] = p.close
    rf_map: dict[dt.date, Decimal] = {}
    for rate in rates:
        if rate.on in rf_map:
            _fail(ErrorCode.SCHEMA_MISMATCH, f"duplicate TB3MS observation on {rate.on}", "One rate per day.")
        rf_map[rate.on] = Decimal(str(rate.tb3ms_annual_percent))
    if len({m.on for m in markets}) != len(markets):
        _fail(ErrorCode.SCHEMA_MISMATCH, "duplicate market return date", "One market observation per day.")
    if len({(m.security_id, m.on) for m in suspense}) != len(suspense):
        _fail(ErrorCode.SCHEMA_MISMATCH, "duplicate suspense evidence", "One suspense observation per security/day.")
    mkt_map = {m.on: m.total_return for m in markets}
    fills_by_day: dict[dt.date, list[DailyFill]] = {}
    for f in fills:
        fills_by_day.setdefault(f.on, []).append(f)
    entries_by_day: dict[dt.date, list[DailyCashEntry]] = {}
    for e in entries:
        entries_by_day.setdefault(e.on, []).append(e)
    suspense_by_key = {(s.security_id, s.on): s for s in suspense}
    missing_rates = [d.isoformat() for d in days if d not in rf_map]
    if missing_rates:
        _fail(ErrorCode.SCHEMA_MISMATCH, f"no TB3MS observation for {len(missing_rates)} trading day(s), first {missing_rates[0]}",
              "The risk-free rate is never zero-filled or carried across missing days.")
    if inputs.start_cash_usd < ZERO:
        _fail(ErrorCode.SCHEMA_MISMATCH, "starting cash cannot be negative", "Start from a non-negative cash balance.")

    cash = _dec(inputs.start_cash_usd)
    quantity: dict[str, Decimal] = {}
    mark: dict[str, Decimal] = {}
    stale: dict[str, int] = {}
    for sid, (q, p) in inputs.initial_positions.items():
        q, p = _dec(q), _dec(p)
        if p <= ZERO:
            _fail(ErrorCode.SCHEMA_MISMATCH, f"initial price for {sid} must be positive", "Initial positions need a positive mark.")
        if q != ZERO:
            quantity[sid], mark[sid] = q, p

    def nav_state() -> Decimal:
        return cash + sum((quantity[s] * mark[s] for s in quantity if quantity[s] != ZERO), ZERO)

    nav_prev = nav_state()
    if nav_prev <= ZERO:
        _fail(ErrorCode.SCHEMA_MISMATCH, "starting NAV must be positive", "A daily series needs positive starting NAV.")
    start_nav = nav_prev
    rows: list[dict[str, Any]] = []
    totals = {"trading": ZERO, "borrow": ZERO, "funding": ZERO, "other_fee": ZERO, "credits": ZERO}
    suspense_marks: list[dict[str, Any]] = []
    fill_costs_input = sum((f.fee_usd for f in fills), ZERO)
    max_residual = ZERO

    for day in days:
        held_at_open = [s for s in sorted(quantity) if quantity[s] != ZERO]
        missing: list[str] = []
        new_marks: dict[str, Decimal] = {}
        for sid in held_at_open:
            previous = mark[sid]
            key = (sid, day)
            if key in px_map and key in ret_map:
                implied = previous * (ONE + ret_map[key])
                if abs(implied - px_map[key]) > tol * max(ONE, abs(px_map[key])):
                    _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                          f"price and return disagree for {sid} on {day}: implied {implied} vs close {px_map[key]}",
                          "Conflicting dated observations are never reconciled by choosing one.")
                new_marks[sid] = px_map[key]
                stale[sid] = 0
            elif key in px_map:
                new_marks[sid] = px_map[key]
                stale[sid] = 0
            elif key in ret_map:
                new_marks[sid] = previous * (ONE + ret_map[key])
                stale[sid] = 0
            elif key in suspense_by_key:
                stale[sid] = stale.get(sid, 0) + 1
                if stale[sid] > policy.max_suspense_days:
                    _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                          f"{sid} has been in suspense for {stale[sid]} trading days (limit {policy.max_suspense_days})",
                          "Unresolved terminal outcomes need dated evidence; suspense is bounded.")
                new_marks[sid] = previous
                ev = suspense_by_key[key]
                suspense_marks.append({"security_id": sid, "on": day.isoformat(), "evidence_id": ev.evidence_id, "reason": ev.reason,
                                       "stale_days": stale[sid]})
            else:
                missing.append(sid)
        if missing:
            shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
            _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                  f"no supplied outcome for {len(missing)} held name(s) on {day}: {shown}",
                  "Absent held outcomes are never zero-filled or interpolated; supply them or dated suspense evidence.")
        position_pnl = sum((quantity[s] * (new_marks[s] - mark[s]) for s in held_at_open), ZERO)
        for sid, price in new_marks.items():
            mark[sid] = price
        rf_daily = rf_map[day] / Decimal(25200)
        income = cash * rf_daily
        cash += income
        debits = {"borrow": ZERO, "funding": ZERO, "other_fee": ZERO, "other": ZERO}
        credits = ZERO
        for entry in entries_by_day.get(day, []):
            cash += entry.credit_usd - entry.debit_usd
            credits += entry.credit_usd
            if entry.kind in ("borrow", "funding", "other_fee", "other"):
                debits[entry.kind] += entry.debit_usd
        trading_fee = ZERO
        slippage = ZERO
        for fill in fills_by_day.get(day, []):
            sid = fill.security_id
            cash -= fill.quantity * fill.price + fill.fee_usd
            trading_fee += fill.fee_usd
            # the day's close mark: today's observed close/return mark, else an earlier same-day mark, else the fill itself
            if sid in new_marks:
                day_mark = new_marks[sid]
            elif (sid, day) in px_map:
                day_mark = px_map[(sid, day)]
            elif sid in mark:
                day_mark = mark[sid]
            else:
                day_mark = fill.price
            # a fill away from the close shows up as P&L, never hidden
            slippage += fill.quantity * (day_mark - fill.price)
            quantity[sid] = quantity.get(sid, ZERO) + fill.quantity
            mark[sid] = day_mark
            stale.setdefault(sid, 0)
            if quantity[sid] == ZERO:
                quantity.pop(sid)
                mark.pop(sid, None)
        nav_now = nav_state()
        other_debits = debits["other"] + debits["other_fee"]
        expected = nav_prev + position_pnl + income + credits - debits["borrow"] - debits["funding"] - other_debits - trading_fee + slippage
        residual = nav_now - expected
        max_residual = max(max_residual, abs(residual))
        if abs(residual) > Decimal("1e-10") * max(ONE, abs(nav_now)):
            _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                  f"NAV reconciliation failed on {day}: residual {residual}",
                  "Daily accounting must conserve cash, positions, costs and income before any report.")
        if nav_now <= ZERO:
            _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, f"NAV is not positive on {day}", "Accounting cannot continue on non-positive NAV.")
        gross = sum((abs(quantity[s] * mark[s]) for s in quantity), ZERO)
        net = sum((quantity[s] * mark[s] for s in quantity), ZERO)
        daily_return = float(nav_now / nav_prev - ONE)
        totals["trading"] += trading_fee
        totals["borrow"] += debits["borrow"]
        totals["funding"] += debits["funding"]
        totals["other_fee"] += other_debits
        totals["credits"] += credits
        rows.append(
            {
                "date": day.isoformat(),
                "date_obj": day,
                "nav": str(nav_now),
                "return": daily_return,
                "risk_free_daily": float(rf_daily),
                "benchmark_daily": benchmark_daily(float(rf_map[day])),
                "position_pnl": str(position_pnl),
                "cash_income": str(income),
                "trading_cost": str(trading_fee),
                "borrow_cost": str(debits["borrow"]),
                "funding_cost": str(debits["funding"]),
                "other_cost": str(other_debits),
                "fill_vs_close_pnl": str(slippage),
                "gross_exposure": float(gross / nav_now),
                "net_exposure": float(net / nav_now),
                "names": len([s for s in quantity if quantity[s] != ZERO]),
                "suspense_names": sorted(m["security_id"] for m in suspense_marks if m["on"] == day.isoformat()),
                "nav_residual": str(residual),
            }
        )
        nav_prev = nav_now

    if totals["trading"] != fill_costs_input:
        _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, "recorded trading costs differ from the supplied fill fees",
              "Costs must reconcile to the fee ledger (spec 26.7).")

    benchmark_check: dict[str, Any] = {"supplied": inputs.reported_benchmark is not None}
    if inputs.reported_benchmark is not None:
        diffs = []
        for row in rows:
            reported = inputs.reported_benchmark.get(row["date_obj"])
            if reported is None:
                _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, f"reported benchmark has no value for {row['date']}",
                      "A supplied benchmark must cover every trading day.")
            diffs.append(abs(float(reported) - row["benchmark_daily"]))
        benchmark_check.update({"max_abs_difference": max(diffs), "reconciled": max(diffs) <= policy.tolerance})
        if not benchmark_check["reconciled"]:
            _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                  f"reported benchmark differs from TB3MS/25200 + 0.04/252 by up to {max(diffs):.3e}",
                  "The benchmark must reconcile before performance statistics are published.")
    else:
        benchmark_check["reconciled"] = None

    returns_list = [r["return"] for r in rows]
    bench_list = [r["benchmark_daily"] for r in rows]
    rf_list = [r["risk_free_daily"] for r in rows]
    end_nav = nav_prev
    compounded = 1.0
    for r in returns_list:
        compounded *= 1.0 + r
    nav_check = abs(float(start_nav) * compounded - float(end_nav)) / float(start_nav)
    if nav_check > float(policy.tolerance) * 10:
        _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, "ending NAV disagrees with compounded daily returns",
              "Ending NAV must agree with the cumulative return (spec 26.7).")
    n = len(rows)
    worst = min(rows, key=lambda r: r["return"])
    best = max(rows, key=lambda r: r["return"])
    nav_path = [float(start_nav)] + [float(r["nav"]) for r in rows]
    active = [p - b for p, b in zip(returns_list, bench_list)]
    stats: dict[str, Any] = {}
    stats["annualized_arithmetic_return"] = annualized_arithmetic(returns_list, policy.periods_per_year)
    if n >= 2:
        stats["annualized_volatility"] = annualized_volatility(returns_list, policy.periods_per_year)
        try:
            stats["information_ratio_vs_benchmark"] = information_ratio(active, policy.periods_per_year)
        except ValueError:
            stats["information_ratio_vs_benchmark"] = None
        try:
            stats["sharpe_ratio"] = sharpe_ratio([p - r for p, r in zip(returns_list, rf_list)], policy.periods_per_year)
        except ValueError:
            stats["sharpe_ratio"] = None
        stats["tracking_error"] = annualized_volatility(active, policy.periods_per_year)
    beta_block: dict[str, Any] = {"status": "not_computed", "reason": "no market returns supplied"}
    rolling: list[dict[str, Any]] = []
    if mkt_map:
        absent = [r["date"] for r in rows if r["date_obj"] not in mkt_map]
        if absent:
            _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, f"no market return for {len(absent)} trading day(s), first {absent[0]}",
                  "Daily beta needs a market return on every trading day; none is interpolated.")
        market_list = [mkt_map[r["date_obj"]] for r in rows]
        if n >= policy.beta_min_observations:
            try:
                beta_block = {"status": "computed", **daily_beta(returns_list, market_list, rf_list, periods_per_year=policy.periods_per_year),
                              "basis": "portfolio and market excess of TB3MS/25200 daily"}
            except ValueError as exc:
                beta_block = {"status": "undefined", "reason": str(exc)}
            if n >= policy.rolling_window:
                rolling = rolling_beta([r["date_obj"] for r in rows], returns_list, market_list, rf_list, policy.rolling_window)
        else:
            beta_block = {"status": "insufficient_observations", "n": n, "required": policy.beta_min_observations}
    summary = {
        "frequency": "daily",
        "basis": "supplied dated observations only; no interpolation; no inference from monthly data",
        "days": n,
        "first_day": rows[0]["date"],
        "last_day": rows[-1]["date"],
        "start_nav": str(start_nav),
        "end_nav": str(end_nav),
        "cumulative_return": compounded - 1.0,
        "worst_day": {"date": worst["date"], "return": worst["return"]},
        "best_day": {"date": best["date"], "return": best["return"]},
        "max_drawdown": max_drawdown(nav_path),
        "costs_usd": {k: str(v) for k, v in totals.items()},
        "benchmark_reconciliation": benchmark_check,
        "daily_beta": beta_block,
        "rolling_beta": rolling,
        "statistics": stats,
        "max_abs_nav_residual": str(max_residual),
        "exact_nav": not suspense_marks,
        "suspense_marks": suspense_marks,
        "ignored_non_trading_observations": ignored,
        "limitations": [
            "Returns are treated as total returns; no dividend is added.",
            "Borrow/funding/other costs are the supplied dated entries; none is invented.",
            "A suspense mark carries the last price; the NAV is then not exact and must be reported with sensitivity.",
        ],
    }
    return DailyReport(rows=rows, summary=summary)


def cagr_from_report(report: DailyReport, periods_per_year: int = 252) -> float:
    """CAGR on the explicit elapsed-trading-day convention (days / periods_per_year years)."""
    years = report.summary["days"] / periods_per_year
    return cagr(float(report.summary["start_nav"]), float(report.summary["end_nav"]), years)
