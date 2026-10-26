"""Capital-independent edge measurement and continuous marginal-edge allocation."""

from __future__ import annotations

import numpy as np


def measure_edge(market, schedule, settings, *, holdings_path=None):
    """Replay unit weights with spread and borrow, without capital or impact.

    This is the zero-size return edge, not the return achievable by a fund.
    All schedules execute after the decision day. Dollar capacity is measured
    separately, and never changes the rule's signal or its edge score.
    """
    days, names = market.returns.shape
    weights, gross_weights = np.zeros(names), np.zeros(names)
    net, gross = np.zeros(days), np.zeros(days)
    borrow_series, spread_series = np.zeros(days), np.zeros(days)
    impact_terms = np.zeros(days)
    hard_caps = np.full(days, np.inf)
    held = (
        np.lib.format.open_memmap(
            holdings_path, mode="w+", dtype=np.float32, shape=(days, names)
        )
        if holdings_path is not None
        else np.zeros((days, names), dtype=np.float32)
    )
    context = []
    for frame in market.frames:
        cap = frame.me.fillna(0).to_numpy(dtype=float) * 1_000_000
        adv = frame.dolvol_126d.fillna(0).to_numpy(dtype=float)
        sigma = np.clip(
            frame.rvol_21d.fillna(0.04).to_numpy(dtype=float), 0.005, 0.3
        )
        spread = np.maximum(
            settings["cost_bps_one_way"] / 10_000,
            np.clip(
                frame.bidaskhl_21d.fillna(0.01).to_numpy(dtype=float) / 2,
                0,
                0.05,
            ),
        )
        fee = (
            np.where(
                cap >= settings["large_cap_borrow_threshold_usd"],
                settings["large_cap_borrow_bps"],
                settings["other_borrow_bps"],
            )
            / 10_000
        )
        context.append((cap, adv, sigma, spread, fee))
    previous = market.dates[0]
    rejected_short_orders = 0
    for day, date in enumerate(market.dates):
        formation = int(market.formation[day])
        if formation < 0:
            previous = date
            continue
        cap, adv, sigma, spread, fee = context[formation]
        observed = market.returns[day]
        valid = np.isfinite(observed) & (observed >= -1)
        gross_pnl = float(gross_weights[valid] @ observed[valid])
        if gross_pnl <= -1:
            raise ValueError("Gross unit exposure became insolvent")
        gross_weights = np.where(
            valid, gross_weights * (1 + np.nan_to_num(observed)), 0
        ) / (1 + gross_pnl)
        missing = ~valid & (weights != 0)
        forced_cost = float(np.abs(weights[missing]) @ spread[missing])
        borrow = (
            float(np.abs(np.minimum(weights, 0)) @ fee)
            * (date - previous).days
            / 365.25
        )
        raw_pnl = float(weights[valid] @ observed[valid])
        pnl = raw_pnl - borrow - forced_cost
        if pnl <= -1:
            raise ValueError("Unit exposure edge replay became insolvent")
        carried = np.where(
            valid, weights * (1 + np.nan_to_num(observed)), 0
        ) / (1 + pnl)
        delta = np.zeros(names)
        cost = 0.0
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
            gross_weights = target.copy()
            # Post-cost target weights must pay for their own transaction costs.
            for _ in range(12):
                delta = target * (1 - cost) - carried
                cost = float(np.abs(delta) @ spread)
            weights = target
        else:
            weights = carried
        trade = np.abs(delta)
        impact_terms[day] = float(
            np.sum(
                trade
                * settings["impact_eta"]
                * sigma
                * np.sqrt(
                    np.divide(trade, adv, out=np.zeros(names), where=adv > 0)
                )
            )
        )
        # These are constraints in dollars per unit exposure. They do not
        # change the rule or its measured unit returns.
        limits = []
        traded = trade > 1e-10
        if traded.any():
            limits.append(
                float(
                    np.min(
                        adv[traded]
                        * settings["max_daily_participation"]
                        * settings["order_ttl_sessions"]
                        / trade[traded]
                    )
                )
            )
        shorts = weights < -1e-10
        if shorts.any():
            inventory = np.minimum(
                cap * settings["max_borrow_fraction_market_cap"],
                adv * settings["max_borrow_adv_days"],
            )
            limits.append(float(np.min(inventory[shorts] / -weights[shorts])))
        if limits:
            hard_caps[day] = max(0.0, min(limits))
        net[day] = (1 + pnl) * (1 - cost) - 1
        gross[day] = gross_pnl
        borrow_series[day], spread_series[day] = borrow, forced_cost + cost
        held[day] = gross_weights
        previous = date
    return {
        "net": net,
        "gross": gross,
        "holdings": held,
        "borrow": borrow_series,
        "spread": spread_series,
        "impact_coefficient_daily": impact_terms,
        "hard_capacity_usd": hard_caps,
        "capital_assumption": None,
        "rejected_short_orders": rejected_short_orders,
    }


def optimal_capital(edge, impact, hard_cap, *, shadow_price=0.0):
    """Maximize A*edge - impact*A**1.5 on a continuous capital interval.

    Edge includes fixed spread and borrow. Capital grows until marginal
    expected return meets the cash opportunity cost, or a hard limit binds.
    """
    if (
        not np.isfinite([edge, impact, hard_cap, shadow_price]).all()
        or impact < 0
        or hard_cap < 0
    ):
        raise ValueError("Finite nonnegative impact and capital limit required")
    available_edge = edge - shadow_price
    if available_edge <= 0:
        return 0.0
    if impact == 0:
        return float(hard_cap)
    return float(min(hard_cap, (available_edge / (1.5 * impact)) ** 2))


def allocate_continuously(edges, impacts, hard_caps, total_capital):
    """Allocate continuously across strategies, with cash when edge is exhausted."""
    edges, impacts, caps = (
        np.asarray(x, dtype=float) for x in (edges, impacts, hard_caps)
    )
    if not (edges.shape == impacts.shape == caps.shape) or total_capital < 0:
        raise ValueError("Invalid capital allocation inputs")
    if edges.ndim != 1 or not np.isfinite(total_capital):
        raise ValueError("Capital inputs must be finite vectors")

    def demand(shadow):
        return np.array(
            [
                optimal_capital(
                    e, i, min(c, total_capital), shadow_price=shadow
                )
                for e, i, c in zip(edges, impacts, caps)
            ]
        )

    unconstrained = demand(0)
    if unconstrained.sum() <= total_capital:
        return unconstrained
    low, high = 0.0, max(0.0, float(edges.max()))
    for _ in range(70):
        mid = (low + high) / 2
        if demand(mid).sum() > total_capital:
            low = mid
        else:
            high = mid
    return demand(high)


def current_capacity(target, carried, frame, settings, budget):
    """Current inventory and staged-turnover bounds, independent of rule selection."""
    target, carried = np.asarray(target), np.asarray(carried)
    adv = frame.dolvol_126d.fillna(0).to_numpy(dtype=float)
    cap = frame.me.fillna(0).to_numpy(dtype=float) * 1_000_000
    delta = np.abs(target - carried)
    traded = delta > 1e-10
    limit = float(budget)
    if traded.any():
        limit = min(
            limit,
            float(
                np.min(
                    adv[traded]
                    * settings["max_daily_participation"]
                    * settings["order_ttl_sessions"]
                    / delta[traded]
                )
            ),
        )
    shorts = target < -1e-10
    if shorts.any():
        inventory = np.minimum(
            cap * settings["max_borrow_fraction_market_cap"],
            adv * settings["max_borrow_adv_days"],
        )
        inventory[cap < 500_000_000] = 0
        limit = min(limit, float(np.min(inventory[shorts] / -target[shorts])))
    return max(0.0, limit)
