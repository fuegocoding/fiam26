"""Required statistics table and exhibit series (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 19.9-19.10).

Every reported number carries its convention label (CAGR vs arithmetic
annualized, daily vs monthly frequency, gross vs net of modeled costs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..accounting.returns import (
    annualized_arithmetic,
    annualized_volatility,
    cagr,
    max_drawdown,
)


@dataclass(frozen=True)
class SeriesPoint:
    date: str
    value: float


@dataclass
class RequiredTable:
    ending_value: float
    starting_value: float
    elapsed_years: float
    periods_per_year: int
    portfolio_returns: tuple[float, ...]
    benchmark_returns: tuple[float, ...]
    risk_free_returns: tuple[float, ...]
    gross_returns: tuple[float, ...]
    cost_returns: tuple[float, ...]
    borrow_returns: tuple[float, ...]
    position_counts: tuple[int, ...] = field(default_factory=tuple)
    gross_exposures: tuple[float, ...] = field(default_factory=tuple)
    net_exposures: tuple[float, ...] = field(default_factory=tuple)
    turnover_per_rebalance: tuple[float, ...] = field(default_factory=tuple)

    def _check_lengths(self) -> None:
        n = len(self.portfolio_returns)
        for name, series in (
            ("benchmark_returns", self.benchmark_returns),
            ("risk_free_returns", self.risk_free_returns),
            ("gross_returns", self.gross_returns),
        ):
            if len(series) != n:
                raise ValueError(f"{name} length {len(series)} != portfolio length {n}")

    def cumulative_return(self) -> float:
        self._check_lengths()
        compounded = 1.0
        for r in self.portfolio_returns:
            compounded *= 1.0 + r
        return compounded - 1.0

    def annualized_arithmetic_return(self) -> float:
        self._check_lengths()
        return annualized_arithmetic(self.portfolio_returns, self.periods_per_year)

    def cagr(self) -> float:
        return cagr(self.starting_value, self.ending_value, self.elapsed_years)

    def annualized_volatility(self) -> float:
        self._check_lengths()
        return annualized_volatility(self.portfolio_returns, self.periods_per_year)

    def max_drawdown(self) -> float:
        self._check_lengths()
        path = [self.starting_value]
        value = self.starting_value
        for r in self.portfolio_returns:
            value *= 1.0 + r
            path.append(value)
        return max_drawdown(path)

    def max_period_loss(self) -> float:
        self._check_lengths()
        return min(self.portfolio_returns) if self.portfolio_returns else 0.0

    def tracking_error(self) -> float:
        self._check_lengths()
        active = [p - b for p, b in zip(self.portfolio_returns, self.benchmark_returns)]
        return annualized_volatility(active, self.periods_per_year)

    def information_ratio(self) -> float:
        self._check_lengths()
        active = [p - b for p, b in zip(self.portfolio_returns, self.benchmark_returns)]
        mean = sum(active) / len(active)
        var = sum((x - mean) ** 2 for x in active) / (len(active) - 1)
        if var == 0:
            raise ValueError("zero active dispersion; information ratio undefined")
        return math.sqrt(self.periods_per_year) * mean / math.sqrt(var)

    def sharpe_ratio(self) -> float:
        self._check_lengths()
        excess = [p - r for p, r in zip(self.portfolio_returns, self.risk_free_returns)]
        mean = sum(excess) / len(excess)
        var = sum((x - mean) ** 2 for x in excess) / (len(excess) - 1)
        if var == 0:
            raise ValueError("zero excess dispersion; Sharpe undefined")
        return math.sqrt(self.periods_per_year) * mean / math.sqrt(var)

    def turnover_per_rebalance_avg(self) -> float:
        if not self.turnover_per_rebalance:
            return 0.0
        return sum(self.turnover_per_rebalance) / len(self.turnover_per_rebalance)

    def turnover_annualized(self) -> float:
        if not self.turnover_per_rebalance:
            return 0.0
        if self.elapsed_years <= 0:
            raise ValueError("elapsed years must be positive")
        return sum(self.turnover_per_rebalance) / self.elapsed_years

    def position_count_stats(self) -> dict[str, float]:
        if not self.position_counts:
            return {"average": 0.0, "minimum": 0.0, "maximum": 0.0}
        return {
            "average": sum(self.position_counts) / len(self.position_counts),
            "minimum": float(min(self.position_counts)),
            "maximum": float(max(self.position_counts)),
        }

    def gross_exposure_stats(self) -> dict[str, float]:
        if not self.gross_exposures:
            return {"average": 0.0, "maximum": 0.0}
        return {
            "average": sum(self.gross_exposures) / len(self.gross_exposures),
            "maximum": max(self.gross_exposures),
        }

    def net_exposure_stats(self) -> dict[str, float]:
        if not self.net_exposures:
            return {"average": 0.0, "minimum": 0.0, "maximum": 0.0}
        return {
            "average": sum(self.net_exposures) / len(self.net_exposures),
            "minimum": min(self.net_exposures),
            "maximum": max(self.net_exposures),
        }

    def annual_cost_pct_nav(self) -> float:
        self._check_lengths()
        if self.elapsed_years <= 0:
            raise ValueError("elapsed years must be positive")
        total = sum(self.cost_returns) + sum(self.borrow_returns)
        return total / self.elapsed_years

    def as_table(self) -> dict[str, object]:
        return {
            "ending_value": self.ending_value,
            "cumulative_return": self.cumulative_return(),
            "annualized_arithmetic_return": self.annualized_arithmetic_return(),
            "cagr": self.cagr(),
            "cagr_convention": "compounded annual growth rate, distinct from arithmetic annualized",
            "annualized_volatility": self.annualized_volatility(),
            "max_drawdown": self.max_drawdown(),
            "max_period_loss": self.max_period_loss(),
            "tracking_error": self.tracking_error(),
            "information_ratio": self.information_ratio(),
            "information_ratio_cost_basis": "net of modeled trading and borrow costs",
            "sharpe_ratio": self.sharpe_ratio(),
            "turnover_per_rebalance_avg": self.turnover_per_rebalance_avg(),
            "turnover_annualized": self.turnover_annualized(),
            "position_counts": self.position_count_stats(),
            "gross_exposure": self.gross_exposure_stats(),
            "net_exposure": self.net_exposure_stats(),
            "annual_cost_pct_nav": self.annual_cost_pct_nav(),
            "periods_per_year": self.periods_per_year,
        }


@dataclass
class ExhibitSeries:
    convention: str = "compounded_active_increments"

    def cumulative_active(
        self,
        portfolio_returns: Sequence[float],
        benchmark_returns: Sequence[float],
        starting_value: float = 1.0,
    ) -> list[SeriesPoint]:
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("series length mismatch")
        points: list[SeriesPoint] = []
        value = starting_value
        for i, (p, b) in enumerate(zip(portfolio_returns, benchmark_returns)):
            value *= 1.0 + (p - b)
            points.append(SeriesPoint(date=str(i), value=value))
        return points

    def relative_wealth(
        self,
        portfolio_returns: Sequence[float],
        benchmark_returns: Sequence[float],
        starting_value: float = 1.0,
    ) -> list[SeriesPoint]:
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("series length mismatch")
        points: list[SeriesPoint] = []
        p_value = starting_value
        b_value = starting_value
        for i, (p, b) in enumerate(zip(portfolio_returns, benchmark_returns)):
            p_value *= 1.0 + p
            b_value *= 1.0 + b
            points.append(SeriesPoint(date=str(i), value=p_value / b_value))
        return points
