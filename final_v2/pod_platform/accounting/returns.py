"""Return identity, turnover, benchmark and performance statistics
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 19.2, 19.5-19.7)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ReturnInputs:
    risk_free: float
    weights: tuple[tuple[str, float], ...]
    returns: tuple[tuple[str, float], ...]
    trading_cost: float = 0.0
    borrow_cost: float = 0.0
    funding_cost: float = 0.0


def portfolio_return(inputs: ReturnInputs) -> float:
    ret_map = dict(inputs.returns)
    total = inputs.risk_free
    for security_id, weight in inputs.weights:
        if security_id not in ret_map:
            raise ValueError(f"missing return for {security_id}")
        total += weight * (ret_map[security_id] - inputs.risk_free)
    return (
        total - inputs.trading_cost - inputs.borrow_cost - inputs.funding_cost
    )


def one_way_turnover(
    pre_weights: Sequence[tuple[str, float]],
    post_weights: Sequence[tuple[str, float]],
) -> float:
    pre = dict(pre_weights)
    post = dict(post_weights)
    keys = set(pre) | set(post)
    return 0.5 * sum(abs(post.get(k, 0.0) - pre.get(k, 0.0)) for k in keys)


def annualized_turnover(
    per_rebalance: float, rebalances_per_year: float
) -> float:
    if rebalances_per_year <= 0:
        raise ValueError("rebalances_per_year must be positive")
    return per_rebalance * rebalances_per_year


def irregular_turnover(turnover_sum: float, elapsed_years: float) -> float:
    if elapsed_years <= 0:
        raise ValueError("elapsed_years must be positive")
    return turnover_sum / elapsed_years


def benchmark_monthly(tb3ms_annual_percent: float) -> float:
    return tb3ms_annual_percent / 1200.0 + 0.04 / 12.0


def benchmark_daily(tb3ms_annual_percent: float) -> float:
    return tb3ms_annual_percent / 25200.0 + 0.04 / 252.0


def active_returns(
    portfolio: Sequence[float], benchmark: Sequence[float]
) -> list[float]:
    if len(portfolio) != len(benchmark):
        raise ValueError(
            "portfolio and benchmark series must have equal length"
        )
    return [p - b for p, b in zip(portfolio, benchmark)]


def information_ratio(active: Sequence[float], periods_per_year: int) -> float:
    if len(active) < 2:
        raise ValueError("information ratio needs at least two observations")
    mean = sum(active) / len(active)
    var = sum((x - mean) ** 2 for x in active) / (len(active) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        raise ValueError(
            "zero active-return dispersion; information ratio undefined"
        )
    return math.sqrt(periods_per_year) * mean / sd


def sharpe_ratio(excess: Sequence[float], periods_per_year: int) -> float:
    if len(excess) < 2:
        raise ValueError("Sharpe ratio needs at least two observations")
    mean = sum(excess) / len(excess)
    var = sum((x - mean) ** 2 for x in excess) / (len(excess) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        raise ValueError(
            "zero excess-return dispersion; Sharpe ratio undefined"
        )
    return math.sqrt(periods_per_year) * mean / sd


def annualized_arithmetic(
    returns: Sequence[float], periods_per_year: int
) -> float:
    return periods_per_year * (sum(returns) / len(returns))


def annualized_volatility(
    returns: Sequence[float], periods_per_year: int
) -> float:
    if len(returns) < 2:
        raise ValueError("volatility needs at least two observations")
    mean = sum(returns) / len(returns)
    var = sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)
    return math.sqrt(periods_per_year) * math.sqrt(var)


def cagr(starting_nav: float, ending_nav: float, elapsed_years: float) -> float:
    if starting_nav <= 0 or ending_nav <= 0:
        raise ValueError("NAV values must be positive")
    if elapsed_years <= 0:
        raise ValueError("elapsed_years must be positive")
    return (ending_nav / starting_nav) ** (1.0 / elapsed_years) - 1.0


def max_drawdown(nav_path: Sequence[float]) -> float:
    peak = float("-inf")
    worst = 0.0
    for value in nav_path:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


@dataclass(frozen=True)
class PeriodStats:
    periods_per_year: int
    arithmetic_annual_return: float
    annual_volatility: float
    information_ratio: Optional[float]
    sharpe: Optional[float]


def summarize_periods(
    portfolio: Sequence[float],
    benchmark: Sequence[float],
    risk_free: Sequence[float],
    periods_per_year: int,
) -> PeriodStats:
    if not (len(portfolio) == len(benchmark) == len(risk_free)):
        raise ValueError("all series must have equal length")
    excess = [p - r for p, r in zip(portfolio, risk_free)]
    active = active_returns(portfolio, benchmark)
    ir = (
        information_ratio(active, periods_per_year)
        if len(active) >= 2
        else None
    )
    sh = sharpe_ratio(excess, periods_per_year) if len(excess) >= 2 else None
    return PeriodStats(
        periods_per_year=periods_per_year,
        arithmetic_annual_return=annualized_arithmetic(
            portfolio, periods_per_year
        ),
        annual_volatility=annualized_volatility(portfolio, periods_per_year),
        information_ratio=ir,
        sharpe=sh,
    )
