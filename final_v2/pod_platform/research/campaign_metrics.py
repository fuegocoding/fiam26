"""Competition reporting conventions over an actual daily net execution book."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    valid = np.isfinite(y) & np.isfinite(x)
    y, x = y[valid], x[valid]
    if len(y) < 3 or np.std(x) == 0:
        return {
            "alpha": None,
            "beta": None,
            "beta_se": None,
            "regression_observations": len(y),
        }
    design = np.column_stack([np.ones(len(x)), x])
    inverse = np.linalg.pinv(design.T @ design)
    coefficient = inverse @ design.T @ y
    residual = y - design @ coefficient
    # Heteroskedasticity robust standard error; rolling uses the same definition.
    covariance = (
        inverse
        @ ((design * residual[:, None]).T @ (design * residual[:, None]))
        @ inverse
        * len(x)
        / (len(x) - 2)
    )
    return {
        "alpha": float(coefficient[0]),
        "beta": float(coefficient[1]),
        "beta_se": float(np.sqrt(max(0, covariance[1, 1]))),
        "regression_observations": len(y),
    }


def frequency_metrics(returns, cash, market, annual):
    returns, cash, market = (
        np.asarray(x, float) for x in (returns, cash, market)
    )
    excess, active = returns - cash, returns - cash - 0.04 / annual
    vol = np.std(excess, ddof=1)
    tracking = np.std(active, ddof=1)
    reg = regression(excess, market - cash)
    return {
        "observations": len(returns),
        "periods_per_year": annual,
        "annualized_arithmetic_return": float(np.mean(returns) * annual),
        "cagr": float(
            np.expm1(np.log1p(returns).sum() * annual / len(returns))
        ),
        "annualized_volatility": float(
            np.std(returns, ddof=1) * np.sqrt(annual)
        ),
        "sharpe": float(np.mean(excess) / vol * np.sqrt(annual))
        if vol > 1e-12
        else None,
        "net_information_ratio": float(
            np.mean(active) / tracking * np.sqrt(annual)
        )
        if tracking > 1e-12
        else None,
        "annualized_alpha": reg.pop("alpha") * annual
        if reg["alpha"] is not None
        else None,
        **reg,
    }


def competition_report(
    dates, net, benchmark, *, turnover=None, returns_include_cash=False
):
    """Benchmark requires official rate yield and S&P returns on every session.

    Rates are lagged publication observations; they are never alpha inputs.
    The daily rate uses the challenge's /252 convention, not calendar accrual.
    """
    dates = pd.DatetimeIndex(dates)
    book = pd.DataFrame({"net": net}, index=dates)
    aligned = benchmark.reindex(dates)
    if aligned["tb3ms"].isna().any():
        raise ValueError("Official benchmark coverage is incomplete")
    book["cash"] = aligned.tb3ms / 25200
    book["market"] = aligned.sp_return
    # Net trading book is excess P&L on the NAV; the full NAV earns cash.
    book["total"] = book.net if returns_include_cash else book.net + book.cash
    monthly = (1 + book[["total", "market"]]).resample("ME").prod() - 1
    # Challenge monthly convention: annual rate in percent divided by 1200.
    monthly["cash"] = aligned.tb3ms.resample("ME").last() / 1200
    incomplete = book.market.isna().resample("ME").max()
    monthly.loc[incomplete, "market"] = np.nan
    wealth = (1 + book.total).cumprod()
    high = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    rolling = []
    for end in range(252, len(book) + 1):
        sample = book.iloc[end - 252 : end]
        rolling.append(
            {
                "date": str(book.index[end - 1].date()),
                **regression(
                    sample.total - sample.cash, sample.market - sample.cash
                ),
            }
        )
    result = {
        "daily": frequency_metrics(book.total, book.cash, book.market, 252),
        "monthly": frequency_metrics(
            monthly.total, monthly.cash, monthly.market, 12
        ),
        "maximum_daily_drawdown": float(np.max(1 - wealth / high)),
        "worst_day": float(book.total.min()),
        "cumulative_return": float(wealth.iloc[-1] - 1),
        "cash_treatment": "cash interest is posted inside daily execution NAV"
        if returns_include_cash
        else "reporting adds daily risk-free to an excess trading-return book",
        "benchmark": "TB3MS + 4 percent annually",
        "sp500_coverage_sessions": int(book.market.notna().sum()),
        "sp500_missing_sessions": int(book.market.isna().sum()),
        "regression_policy": "Use only observed official S&P returns; other performance statistics retain every session",
        "rolling_252_day_beta": rolling,
    }
    if turnover is not None:
        one_way = np.asarray(turnover, float) / 2
        result["annualized_one_way_turnover"] = float(
            one_way.sum() * 252 / len(book)
        )
    return result, book, monthly
