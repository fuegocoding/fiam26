"""Measure unit returns for agent-proposed schedules."""

import numpy as np


def measure_edge(
    market, schedule, settings, *, holdings_path=None, keep_holdings=True,
    snapshot_days=(),
):
    """Replay unit weights without capital.

    Schedules fill in full at their observed close after the decision.
    Neither fund size nor trading volume changes the measured edge.
    """
    days, names = market.returns.shape
    weights = np.zeros(names)
    returns = np.zeros(days)
    long_contribution, short_contribution = np.zeros(days), np.zeros(days)
    turnover = np.zeros(days)
    held = None
    if keep_holdings:
        held = (
            np.lib.format.open_memmap(
                holdings_path, mode="w+", dtype=np.float32, shape=(days, names)
            )
            if holdings_path is not None
            else np.zeros((days, names), dtype=np.float32)
        )
    rejected_short_orders = 0
    requested_snapshots = set(snapshot_days)
    prior_holdings = {}
    for day in range(days):
        if day in requested_snapshots:
            prior_holdings[day] = weights.copy()
        formation = int(market.formation[day])
        if formation < 0:
            continue
        cap = market.frames[formation].me.fillna(0).to_numpy(dtype=float) * 1_000_000
        observed = market.returns[day]
        valid = np.isfinite(observed) & (observed >= -1)
        missing = ~valid & (weights != 0)
        forced_turnover = float(np.abs(weights[missing]).sum())
        long_contribution[day] = float(
            np.maximum(weights[valid], 0) @ observed[valid]
        )
        short_contribution[day] = float(
            np.minimum(weights[valid], 0) @ observed[valid]
        )
        pnl = long_contribution[day] + short_contribution[day]
        if pnl <= -1:
            raise ValueError("Unit exposure edge replay became insolvent")
        carried = np.where(
            valid, weights * (1 + np.nan_to_num(observed)), 0
        ) / (1 + pnl)
        delta = np.zeros(names)
        if day in schedule:
            target = np.asarray(schedule[day], dtype=float).copy()
            target[~market.tradable[day]] = 0
            ineligible = (target < -1e-10) & (cap < 500_000_000)
            if ineligible.any():
                if not settings.get("reject_ineligible_short_orders", False):
                    raise ValueError(
                        "Edge rule proposes a short below the mandate floor"
                    )
                # A quote/eligibility rejection at execution leaves cash. Never
                # promote another stock using execution-day knowledge.
                rejected_short_orders += int(ineligible.sum())
                target[ineligible] = 0
                long_total = target[target > 0].sum()
                short_total = -target[target < 0].sum()
                if long_total > short_total and long_total > 0:
                    target[target > 0] *= short_total / long_total
            delta = target - carried
            weights = target
        else:
            weights = carried
        trade = np.abs(delta)
        returns[day] = pnl
        turnover[day] = float(trade.sum() + forced_turnover)
        if held is not None:
            held[day] = weights
    return {
        "returns": returns,
        "holdings": held,
        "long_contribution": long_contribution,
        "short_contribution": short_contribution,
        "turnover_two_way": turnover,
        "capital_assumption": None,
        "rejected_short_orders": rejected_short_orders,
        "prior_holdings": prior_holdings,
    }
