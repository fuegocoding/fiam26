"""Daily/weekly and threshold-driven rules with next-close execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .training_data import market_return_series
from .training_execution import replay_capacity
from .training_search import (
    Candidate,
    objective,
    project_target,
    risk_designs,
    summarize,
)


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
    """Retain 2014 warmup for trailing signals and 2015–2020 return books."""
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
    """Earn on carried shares, execute at close, charge actual net turnover.

    Schedule keys are execution sessions, after information-day decisions.
    Missing held observations exit at the last known price and remain cash.
    """
    if settings.get("capacity_execution", False):
        return replay_capacity(
            market, schedule, settings, keep_holdings=keep_holdings
        )
    weights = np.zeros(market.returns.shape[1])
    net = np.zeros(len(market.dates))
    turnover = np.zeros(len(net))
    held = (
        np.zeros_like(market.returns, dtype=np.float32)
        if keep_holdings
        else None
    )
    cost_rate = settings["cost_bps_one_way"] / 10_000
    previous_date = market.dates[0]
    for d, date in enumerate(market.dates):
        raw = market.returns[d]
        valid = np.isfinite(raw) & (raw >= -1)
        exits = (~valid) & (np.abs(weights) > 1e-12)
        exit_cost = np.abs(weights[exits]).sum() * cost_rate
        cash = (
            (1 - weights.sum()) * market.cash_returns[d]
            if market.cash_returns is not None
            and d >= min(schedule, default=len(market.dates))
            else 0.0
        )
        pnl = float(weights[valid] @ raw[valid]) + cash
        elapsed = (date - previous_date).days
        borrow = np.abs(weights[weights < 0]).sum() * (
            settings["borrow_bps_annual"] / 10_000 * elapsed / 365.25
        )
        day_return = pnl - borrow - exit_cost
        if day_return <= -1:
            raise ValueError("Insolvent daily research book")
        carried = np.where(valid, weights * (1 + np.nan_to_num(raw)), 0)
        carried /= 1 + day_return
        trade = 0.0
        if d in schedule:
            target = np.asarray(schedule[d], dtype=float) * market.tradable[d]
            trade = float(np.abs(target - carried).sum())
            # Targets are fractions of post-cost NAV: solve the self-financing
            # transaction-cost equation by fixed point iteration.
            fee = trade * cost_rate
            for _ in range(8):
                fee = cost_rate * float(
                    np.abs(target * (1 - fee) - carried).sum()
                )
            day_return = (1 + day_return) * (1 - fee) - 1
            weights = target
        else:
            weights = carried
        if d == len(net) - 1 and settings.get("liquidate_at_end", True):
            liquidation = np.abs(weights).sum() * cost_rate
            day_return = (1 + day_return) * (1 - liquidation) - 1
            trade += np.abs(weights).sum()
            weights = np.zeros_like(weights)
        net[d], turnover[d] = (
            day_return,
            trade + exit_cost / max(cost_rate, 1e-30),
        )
        if held is not None:
            held[d] = weights
        previous_date = date
    return {"net": net, "turnover": turnover, "holdings": held}


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


def panel_schedule(weights, panel, market, rebalance_months=1):
    """Allow quarterly/semiannual alpha reviews with mandatory universe exits."""
    result = {}
    for t, date in enumerate(panel.dates[:-1]):
        if t and date.month % rebalance_months != 0:
            continue
        d = market.dates.searchsorted(date, side="right")
        if d < len(market.dates):
            result[d] = weights[t]
    return result


def tactical_signals(panel, market, family, lookback):
    """Price-only signals use trailing total returns, never future adjusted prices."""
    valid = np.isfinite(market.returns) & (market.returns > -1)
    logs = pd.DataFrame(
        np.where(valid, np.log1p(np.maximum(market.returns, -0.999999)), np.nan)
    )
    if family == "reversal":
        signal = -logs.rolling(lookback, min_periods=lookback).sum()
    elif family == "momentum":
        # Skip the last five sessions to separate trend from short reversal.
        signal = logs.shift(5).rolling(lookback, min_periods=lookback).sum()
    elif family == "low_volatility":
        signal = -logs.rolling(lookback, min_periods=lookback).std()
    elif family == "breakout":
        growth = logs.fillna(0).cumsum()
        peak = growth.rolling(lookback, min_periods=lookback).max()
        signal = growth - peak
        signal = signal.where(logs.rolling(lookback).count() >= lookback)
    else:
        raise ValueError("Unknown tactical rule")
    values = signal.to_numpy()
    for d, t in enumerate(market.formation):
        if t < 0:
            values[d] = np.nan
        else:
            values[d, ~panel.mask[t]] = np.nan
    ranks = pd.DataFrame(values).rank(axis=1, pct=True).to_numpy() * 2 - 1
    return ranks


def tactical_schedule(panel, market, settings, recipe):
    """Daily or weekly decisions, with separate rank entry/exit thresholds."""
    signals = (
        tactical_signals(
            panel, market, recipe["family"], recipe["lookback_sessions"]
        )
        * recipe["expected_return_sign"]
    )
    positions = np.zeros(len(panel.securities))
    result = {}
    for d, t in enumerate(market.formation[:-1]):
        if t < 0:
            continue
        date = market.dates[d]
        if recipe["review"] == "weekly":
            following = market.dates[d + 1]
            if date.isocalendar()[:2] == following.isocalendar()[:2]:
                continue
        score = signals[d]
        entry, exit_level = recipe["entry_rank"], recipe["exit_rank"]
        if entry == 0:
            desired = np.nan_to_num(score)
        else:
            desired = threshold_positions(score, positions, entry, exit_level)
        desired[~panel.mask[t]] = 0
        mask, design, inverse = market.designs[t]
        values = desired[mask]
        values = values - design @ (inverse @ values)
        target = np.zeros(len(positions))
        values *= getattr(market.designs[t], "liquidity_scale", 1)
        short_scale = getattr(
            market.designs[t], "short_cost_scale", np.ones(len(values))
        )
        values[values < 0] *= short_scale[values < 0]
        target[mask] = project_target(
            values,
            design,
            settings,
            normalize=True,
            shortable=getattr(market.designs[t], "shortable", None),
        )
        # Threshold changes trigger execution; unchanged membership can persist.
        changed = entry == 0 or np.any(desired != positions)
        if changed:
            result[d + 1] = target.astype(np.float32)
        positions = desired
    return result


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


def schedule_for(candidate, panel, market, settings):
    recipe = candidate.recipe
    if "family" in recipe:
        return tactical_schedule(panel, market, settings, recipe)
    return panel_schedule(
        candidate.weights, panel, market, recipe.get("rebalance_months", 1)
    )


def search_frequencies(panel, settings, candidates, market, progress=None):
    """Compare fixed slower panel rules and faster dynamics rules honestly."""
    additions = []
    slow_features = {
        "fundamentals": {"niq_su", "gp_at", "cash_at"},
        "events": {"eqnetis_at", "taccruals_at", "at_gr1"},
        "valuation": {"be_me", "ni_me", "ocf_me"},
    }
    for base in candidates:
        if base.recipe.get("feature") not in slow_features.get(base.pod, set()):
            continue
        if base.recipe["smoothing_months"] != 1:
            continue
        for months in settings["slow_rebalance_months"]:
            recipe = {**base.recipe, "rebalance_months": months}
            schedule = panel_schedule(base.weights, panel, market, months)
            book = replay_daily(market, schedule, settings)
            net = monthly_net(book["net"], market, panel)
            metrics = summarize(net, market_return_series(panel))
            additions.append(
                Candidate(
                    base.strategy_id + f":review{months}m",
                    base.pod,
                    recipe,
                    base.weights,
                    net,
                    book["turnover"],
                    metrics,
                    objective(metrics),
                )
            )
    for family, lookbacks in settings["tactical_lookbacks"].items():
        for lookback in lookbacks:
            for review in ("daily", "weekly"):
                for entry, exit_level in settings["threshold_pairs"]:
                    for sign in (1, -1):
                        recipe = {
                            "family": family,
                            "lookback_sessions": lookback,
                            "review": review,
                            "entry_rank": entry,
                            "exit_rank": exit_level,
                            "expected_return_sign": sign,
                            "scaling": "cross_sectional_rank",
                        }
                        schedule = tactical_schedule(
                            panel, market, settings, recipe
                        )
                        book = replay_daily(market, schedule, settings)
                        net = monthly_net(book["net"], market, panel)
                        metrics = summarize(net, market_return_series(panel))
                        # Month-end snapshots support the shared allocation
                        # search; the final combined book replays exact schedules.
                        snapshots = np.zeros_like(panel.mask, dtype=np.float32)
                        ordered = sorted(schedule)
                        for t, date in enumerate(panel.dates):
                            d = market.dates.searchsorted(date, side="right")
                            previous = [i for i in ordered if i <= d]
                            if previous:
                                snapshots[t] = schedule[previous[-1]]
                        name = f"dynamics:{family}{lookback}:{review}:entry{entry}:exit{exit_level}:sign{sign:+d}"
                        additions.append(
                            Candidate(
                                name,
                                "dynamics",
                                recipe,
                                snapshots,
                                net,
                                book["turnover"],
                                metrics,
                                objective(metrics),
                            )
                        )
        if progress:
            progress(family, len(additions))
    return additions


def combined_schedule(
    selected, allocations, panel, market, settings, cache=None
):
    """Net pod targets in one shared book, retaining each rule's own schedule."""
    cache = {} if cache is None else cache
    books = {}
    events = set()
    by_id = {
        s.strategy_id: s for strategies in selected.values() for s in strategies
    }
    if not by_id:
        return {}
    for identity, strategy in by_id.items():
        if identity not in cache:
            schedule = schedule_for(strategy, panel, market, settings)
            # These are desired model quantities. The shared fund applies its
            # own participation, borrow and costs once, after netting pods.
            book = replay_daily(
                market,
                schedule,
                {
                    **settings,
                    "capacity_execution": False,
                    "cost_bps_one_way": 0,
                    "borrow_bps_annual": 0,
                    "liquidate_at_end": False,
                },
                keep_holdings=True,
            )
            cache[identity] = (schedule, book)
        schedule, book = cache[identity]
        # Orders for close d can use marks through d-1. A rule scheduled for d
        # supplies its already formed target; other rules retain yesterday's
        # known model weights. Using holdings[d] would use execution-day returns.
        known = np.zeros_like(book["holdings"])
        known[1:] = book["holdings"][:-1]
        for day, target in schedule.items():
            known[day] = target
        books[identity] = known
        events.update(schedule)
    for day in panel.execution_dates:
        events.add(int(market.dates.searchsorted(day)))
    capital = {}
    for record in allocations:
        capital.setdefault(record["formation_date"], {})[
            record["strategy_id"]
        ] = record["capital_weight"]
    result = {}
    for d in sorted(events):
        if d <= 0 or d >= len(market.dates):
            continue
        # Month-end may be a weekend; its snapshot still precedes Monday's
        # execution even though Friday is the previous trading session.
        t = (
            panel.dates.searchsorted(
                market.dates[d] - pd.Timedelta(days=1), side="right"
            )
            - 1
        )
        if t < 0:
            continue
        allocation = capital.get(str(panel.dates[t].date()), {})
        target = sum(
            allocation.get(identity, 0) * book[d]
            for identity, book in books.items()
        )
        mask, design, inverse = market.designs[t]
        values = target[mask].astype(float)
        beta = panel.frames[t].beta_60m.fillna(1).to_numpy(dtype=float)
        sectors = panel.frames[t].gics.fillna("unknown").astype(str).str[:2]
        sector_net = (
            pd.Series(target).groupby(sectors.to_numpy()).sum().abs().max()
        )
        shortable = getattr(
            market.designs[t], "shortable", np.ones(len(values), dtype=bool)
        )
        needs_risk_trade = (
            np.count_nonzero(np.abs(target) > 1e-10) > settings["universe_size"]
            or abs(target.sum()) > settings.get("target_net_band", 0.01)
            or abs(target @ beta) > settings.get("target_beta_band", 0.05)
            or sector_net > settings.get("target_sector_band", 0.02)
            or np.abs(target).sum() > settings["gross_limit"]
            or np.abs(target).max() > settings["single_name_limit"]
            or np.any((values < -1e-10) & ~shortable)
        )
        if needs_risk_trade:
            values -= design @ (inverse @ values)
            target = np.zeros(len(panel.securities))
            target[mask] = project_target(
                values, design, settings, shortable=shortable
            )
        result[d] = target.astype(np.float32)
    return result
