"""Market preparation and replay for agent-proposed schedules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .training_execution import replay_at_observed_prices
from .training_search import risk_designs


@dataclass
class DailyMarket:
    dates: pd.DatetimeIndex
    returns: np.ndarray
    tradable: np.ndarray
    formation: np.ndarray
    designs: list
    frames: list | None = None
    cash_returns: np.ndarray | None = None



def daily_market(panel, settings=None):
    """Retain price warmup and only the declared development return history."""
    source = panel.daily
    prices = source.pivot(index="date", columns="permno", values="prc")
    returns = source.pivot(
        index="date", columns="permno", values="daily_return"
    )
    dates = prices.index.sort_values()
    prices = prices.reindex(index=dates, columns=panel.securities).to_numpy()
    returns = returns.reindex(index=dates, columns=panel.securities).to_numpy()
    # The same-day month-end snapshot is available for decisions at the close.
    formation = panel.dates.searchsorted(dates, side="right") - 1
    return DailyMarket(
        dates,
        returns,
        np.isfinite(prices) & (prices > 0),
        formation,
        risk_designs(panel, settings),
        panel.frames,
    )



def replay_daily(market, schedule, settings, *, keep_holdings=False):
    """Earn on carried shares and execute at close.

    Schedule keys identify the observed close used for each decision.
    Missing held observations exit at the last known price and remain cash.
    """
    return replay_at_observed_prices(
        market, schedule, settings, keep_holdings=keep_holdings
    )



def unit_return_month_ends(panel):
    """Calendar unit returns are known at month end, unlike interval labels."""
    return pd.DatetimeIndex(panel.dates[:-1] + pd.offsets.MonthEnd(1))



def monthly_net(daily_net, market, panel):
    """Use calendar return months; exclude pre-strategy warmup sessions."""
    series = pd.Series(daily_net, index=market.dates)
    monthly = (1 + series).groupby(series.index.to_period("M")).prod() - 1
    return np.asarray(
        [
            monthly.loc[(date + pd.offsets.MonthEnd(1)).to_period("M")]
            for date in panel.dates[:-1]
        ]
    )



def threshold_positions(score, previous, entry, exit_level):
    """Hold existing membership between separate entry and exit boundaries."""
    if not 0 <= exit_level < entry <= 1:
        raise ValueError("Thresholds require 0 <= exit < entry <= 1")
    desired = np.asarray(previous, dtype=float).copy()
    desired[(previous > 0) & (score < exit_level)] = 0
    desired[(previous < 0) & (score > -exit_level)] = 0
    desired[score >= entry] = 1
    desired[score <= -entry] = -1
    desired[~np.isfinite(score)] = 0
    return desired
