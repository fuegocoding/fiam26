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
    spx_returns: tuple[float, ...] = field(default_factory=tuple)
    max_one_day_loss: Optional[float] = None
    max_one_day_loss_date: Optional[str] = None
    beta_daily: Optional[float] = None
    beta_daily_se: Optional[float] = None
    beta_daily_t_hac: Optional[float] = None
    avg_holding_period_days: Optional[float] = None
    avg_position_size_musd: Optional[float] = None

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

    def max_drawdown_duration_periods(self, returns: Optional[Sequence[float]] = None) -> int:
        series = self.portfolio_returns if returns is None else returns
        peak = self.starting_value
        val = self.starting_value
        cur_dur = 0
        max_dur = 0
        for r in series:
            val *= 1.0 + r
            if val >= peak:
                peak = val
                cur_dur = 0
            else:
                cur_dur += 1
                if cur_dur > max_dur:
                    max_dur = cur_dur
        return max_dur

    def hit_rate_vs_benchmark(self) -> float:
        self._check_lengths()
        if not self.portfolio_returns:
            return 0.0
        wins = sum(1 for p, b in zip(self.portfolio_returns, self.benchmark_returns) if p > b)
        return wins / len(self.portfolio_returns)

    def sp500_regression(self) -> dict[str, Optional[float]]:
        if not self.spx_returns or len(self.spx_returns) != len(self.portfolio_returns):
            return {
                "beta_monthly": None,
                "beta_monthly_se": None,
                "beta_monthly_t_hac": None,
                "alpha_monthly_ann": None,
                "alpha_monthly_se_ann": None,
                "alpha_monthly_t_hac": None,
            }
        import numpy as np
        from ..accounting.evaluation import ols, ols_hac

        y = np.array([p - rf for p, rf in zip(self.portfolio_returns, self.risk_free_returns)], dtype=float)
        x = np.array([s - rf for s, rf in zip(self.spx_returns, self.risk_free_returns)], dtype=float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() <= 6:
            return {
                "beta_monthly": None,
                "beta_monthly_se": None,
                "beta_monthly_t_hac": None,
                "alpha_monthly_ann": None,
                "alpha_monthly_se_ann": None,
                "alpha_monthly_t_hac": None,
            }
        b, se = ols(y[mask], x[mask][:, None])
        _, th, _ = ols_hac(y[mask], x[mask][:, None], lags=3)
        return {
            "beta_monthly": float(b[1]),
            "beta_monthly_se": float(se[1]),
            "beta_monthly_t_hac": float(th[1]),
            "alpha_monthly_ann": float(b[0] * self.periods_per_year),
            "alpha_monthly_se_ann": float(se[0] * self.periods_per_year),
            "alpha_monthly_t_hac": float(th[0]),
        }

    def benchmark_stats(self) -> dict[str, object]:
        self._check_lengths()
        b_comp = 1.0
        for r in self.benchmark_returns:
            b_comp *= 1.0 + r
        b_end = self.starting_value * b_comp
        b_cum = b_comp - 1.0
        b_arith = annualized_arithmetic(self.benchmark_returns, self.periods_per_year)
        b_cagr = cagr(self.starting_value, b_end, self.elapsed_years)
        b_vol = annualized_volatility(self.benchmark_returns, self.periods_per_year)
        b_path = [self.starting_value]
        v = self.starting_value
        for r in self.benchmark_returns:
            v *= 1.0 + r
            b_path.append(v)
        b_mdd = max_drawdown(b_path)
        b_mdd_dur = self.max_drawdown_duration_periods(self.benchmark_returns)
        b_min = min(self.benchmark_returns) if self.benchmark_returns else 0.0
        return {
            "ending_value": b_end,
            "cumulative_return": b_cum,
            "annualized_arithmetic_return": b_arith,
            "cagr": b_cagr,
            "annualized_volatility": b_vol,
            "max_drawdown": b_mdd,
            "max_drawdown_duration_periods": b_mdd_dur,
            "max_period_loss": b_min,
        }

    def as_table(self) -> dict[str, object]:
        reg = self.sp500_regression()
        bench = self.benchmark_stats()
        pos_stats = self.position_count_stats()
        gross_stats = self.gross_exposure_stats()
        net_stats = self.net_exposure_stats()
        t_avg = self.turnover_per_rebalance_avg()
        t_ann = self.turnover_annualized()
        hold_days = (
            self.avg_holding_period_days
            if self.avg_holding_period_days is not None
            else (gross_stats["average"] / max(t_ann, 1e-9) * 252.0 if t_ann > 0 else 0.0)
        )
        avg_pos_pct = (
            gross_stats["average"] / max(pos_stats["average"], 1.0)
            if pos_stats["average"] > 0
            else 0.0
        )
        avg_pos_musd = (
            self.avg_position_size_musd
            if self.avg_position_size_musd is not None
            else (avg_pos_pct * (self.starting_value / 1e6))
        )
        section_9_1 = {
            "01_ending_value_usd": {"strategy": self.ending_value, "benchmark": bench["ending_value"]},
            "02_cumulative_return_net": {"strategy": self.cumulative_return(), "benchmark": bench["cumulative_return"]},
            "03_annualized_arithmetic_return": {"strategy": self.annualized_arithmetic_return(), "benchmark": bench["annualized_arithmetic_return"]},
            "04_cagr_compounded": {"strategy": self.cagr(), "benchmark": bench["cagr"]},
            "05_annualized_volatility": {"strategy": self.annualized_volatility(), "benchmark": bench["annualized_volatility"]},
            "06_tracking_error": {"strategy": self.tracking_error(), "benchmark": 0.0},
            "07_information_ratio": {"strategy": self.information_ratio(), "benchmark": 0.0},
            "08_sharpe_ratio_vs_tb3ms": {"strategy": self.sharpe_ratio(), "benchmark": None},
            "09_beta_vs_sp500": {
                "beta_monthly": reg["beta_monthly"],
                "beta_monthly_se": reg["beta_monthly_se"],
                "beta_monthly_t_hac": reg["beta_monthly_t_hac"],
                "beta_daily": self.beta_daily,
                "beta_daily_se": self.beta_daily_se,
                "beta_daily_t_hac": self.beta_daily_t_hac,
            },
            "10_alpha_vs_sp500_annualized": {
                "alpha_monthly_ann": reg["alpha_monthly_ann"],
                "alpha_monthly_se_ann": reg["alpha_monthly_se_ann"],
                "alpha_monthly_t_hac": reg["alpha_monthly_t_hac"],
            },
            "11_max_drawdown": {"strategy": self.max_drawdown(), "benchmark": bench["max_drawdown"]},
            "12_max_drawdown_duration_months": {
                "strategy": self.max_drawdown_duration_periods(),
                "benchmark": bench["max_drawdown_duration_periods"],
            },
            "13_max_1m_loss_and_max_1d_loss": {
                "max_1m_loss_strategy": self.max_period_loss(),
                "max_1m_loss_benchmark": bench["max_period_loss"],
                "max_1d_loss_strategy": self.max_one_day_loss,
                "max_1d_loss_date": self.max_one_day_loss_date,
            },
            "14_hit_rate_vs_benchmark": self.hit_rate_vs_benchmark(),
            "15_number_of_positions": pos_stats,
            "16_turnover_per_rebalance_avg": t_avg,
            "17_turnover_annualized": t_ann,
            "18_average_holding_period_trading_days": hold_days,
            "19_average_position_size": {
                "avg_position_size_musd": avg_pos_musd,
                "avg_position_size_pct_nav": avg_pos_pct,
            },
            "20_gross_exposure_pct_nav": gross_stats,
            "21_net_exposure_pct_nav": net_stats,
            "22_annual_total_cost_pct_nav": {
                "total_cost_pct_nav": self.annual_cost_pct_nav(),
                "trading_cost_pct_nav": sum(self.cost_returns) / self.elapsed_years if self.elapsed_years > 0 else 0.0,
                "borrow_cost_pct_nav": sum(self.borrow_returns) / self.elapsed_years if self.elapsed_years > 0 else 0.0,
            },
        }
        return {
            "ending_value": self.ending_value,
            "cumulative_return": self.cumulative_return(),
            "annualized_arithmetic_return": self.annualized_arithmetic_return(),
            "cagr": self.cagr(),
            "cagr_convention": "compounded annual growth rate, distinct from arithmetic annualized",
            "annualized_volatility": self.annualized_volatility(),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_duration_months": self.max_drawdown_duration_periods(),
            "max_period_loss": self.max_period_loss(),
            "max_one_day_loss": self.max_one_day_loss,
            "max_one_day_loss_date": self.max_one_day_loss_date,
            "tracking_error": self.tracking_error(),
            "information_ratio": self.information_ratio(),
            "information_ratio_cost_basis": "net of modeled trading and borrow costs",
            "sharpe_ratio": self.sharpe_ratio(),
            "hit_rate_vs_benchmark": self.hit_rate_vs_benchmark(),
            "sp500_regression": reg,
            "beta_daily": self.beta_daily,
            "beta_daily_se": self.beta_daily_se,
            "beta_daily_t_hac": self.beta_daily_t_hac,
            "turnover_per_rebalance_avg": t_avg,
            "turnover_annualized": t_ann,
            "average_holding_period_trading_days": hold_days,
            "average_position_size_musd": avg_pos_musd,
            "position_counts": pos_stats,
            "gross_exposure": gross_stats,
            "net_exposure": net_stats,
            "annual_cost_pct_nav": self.annual_cost_pct_nav(),
            "benchmark_summary": bench,
            "section_9_1_21_rows": section_9_1,
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
