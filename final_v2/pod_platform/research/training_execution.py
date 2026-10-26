"""Capacity-aware research fills with compounding NAV and explicit borrow proxies."""

from __future__ import annotations

import numpy as np


def replay_capacity(market, schedule, settings, *, keep_holdings=False):
    """Execute pending targets over sessions at bounded trailing-ADV participation.

    This uses the existing execution planner's square-root impact equation:
    eta * daily_sigma * sqrt(order_dollars / ADV), plus spread and explicit cost.
    Borrow availability/fees are modeled assumptions, not historical locate data.
    Only actual fills enter holdings and earn returns.
    """
    days, names = market.returns.shape
    weights, desired = np.zeros(names), np.zeros(names)
    position_age = np.zeros(names)
    closed_dollar_days = closed_dollars = 0.0
    (
        long_pnl_series,
        short_pnl_series,
        cash_series,
        fees_series,
        borrow_series,
    ) = (np.zeros(days) for _ in range(5))
    observer = settings.get("execution_observer")
    net, turnover = np.zeros(days), np.zeros(days)
    costs, borrow_costs, pending = (
        np.zeros(days),
        np.zeros(days),
        np.zeros(days),
    )
    counts, gross_series, net_series = (
        np.zeros(days),
        np.zeros(days),
        np.zeros(days),
    )
    held = None
    if keep_holdings:
        held = (
            np.lib.format.open_memmap(
                settings["holdings_storage_path"],
                mode="w+",
                dtype=np.float32,
                shape=(days, names),
            )
            if settings.get("holdings_storage_path")
            else np.zeros((days, names), dtype=np.float32)
        )
    recorded_targets = {} if keep_holdings else None
    recorded_renewals = {} if keep_holdings else None
    nav = float(
        settings.get("execution_start_nav", settings["strategy_capital_usd"])
    )
    initial_nav = nav
    active_from = min(schedule) if schedule else days
    largest_name = np.zeros(days)
    transaction_dollars = borrow_dollars = cash_income_dollars = 0.0
    last_order = np.full(names, -10_000, dtype=int)
    maximum_participation, expired_orders, rejected_shorts = 0.0, 0, 0
    snapshot_key = (
        settings["cost_bps_one_way"],
        settings["large_cap_borrow_threshold_usd"],
        settings["large_cap_borrow_bps"],
        settings["other_borrow_bps"],
        settings.get("execution_spread_multiplier", 1.0),
    )
    snapshot_cache = getattr(market, "_execution_snapshot_cache", {})
    market._execution_snapshot_cache = snapshot_cache
    snapshots = snapshot_cache.get(snapshot_key, [])
    previous_cap = previous_adv = previous_sigma = previous_spread = None
    for frame in [] if snapshots else market.frames:
        cap = frame.me.to_numpy(dtype=float) * 1_000_000
        adv = frame.dolvol_126d.to_numpy(dtype=float)
        sigma = frame.rvol_21d.to_numpy(dtype=float)
        spread = frame.bidaskhl_21d.to_numpy(dtype=float) / 2
        cap = np.where(
            np.isfinite(cap),
            cap,
            previous_cap if previous_cap is not None else 0,
        )
        adv = np.where(
            np.isfinite(adv),
            adv,
            previous_adv if previous_adv is not None else 0,
        )
        sigma = np.clip(
            np.where(
                np.isfinite(sigma),
                sigma,
                previous_sigma if previous_sigma is not None else 0.04,
            ),
            0.005,
            0.3,
        )
        spread = np.clip(
            np.where(
                np.isfinite(spread),
                spread,
                previous_spread if previous_spread is not None else 0.005,
            ),
            0,
            0.05,
        )
        previous_cap, previous_adv, previous_sigma, previous_spread = (
            cap,
            adv,
            sigma,
            spread,
        )
        base = np.maximum(
            settings["cost_bps_one_way"] / 10_000, spread
        ) * settings.get("execution_spread_multiplier", 1.0)
        fee = (
            np.where(
                cap >= settings["large_cap_borrow_threshold_usd"],
                settings["large_cap_borrow_bps"],
                settings["other_borrow_bps"],
            )
            / 10_000
        )
        snapshots.append((cap, adv, sigma, base, fee))
    snapshot_cache[snapshot_key] = snapshots
    previous_date = market.dates[0]
    for d, date in enumerate(market.dates):
        t = int(market.formation[d])
        if t < 0:
            previous_date = date
            continue
        # Adaptive allocators make their dollar decision from yesterday's NAV
        # and holdings, before this execution session's return is inspected.
        decision_nav = nav
        instruction = schedule.get(d)
        adaptive = callable(instruction)
        decision = instruction(nav, weights.copy()) if adaptive else instruction
        cap, adv, sigma, base, fee = snapshots[t]
        raw = market.returns[d]
        # Valuation uses observed returns; execution separately requires a quote.
        valid = np.isfinite(raw) & (raw >= -1)
        exits = (~valid) & (np.abs(weights) > 1e-12)
        exit_turnover = float(np.abs(weights[exits]).sum())
        exit_fee = float(np.abs(weights[exits]) @ base[exits])
        elapsed = (date - previous_date).days
        position_age += elapsed
        borrow = float(np.abs(np.minimum(weights, 0)) @ fee) * elapsed / 365.25
        borrow_dollars += borrow * nav
        transaction_dollars += exit_fee * nav
        cash_return = (
            (1 - weights.sum()) * market.cash_returns[d]
            if getattr(market, "cash_returns", None) is not None
            and d >= active_from
            else 0.0
        )
        cash_income_dollars += cash_return * nav
        long_pnl = float(np.maximum(weights[valid], 0) @ raw[valid])
        short_pnl = float(np.minimum(weights[valid], 0) @ raw[valid])
        (
            long_pnl_series[d],
            short_pnl_series[d],
            cash_series[d],
            borrow_series[d],
        ) = long_pnl, short_pnl, cash_return, borrow
        closed_dollar_days += (
            float(np.abs(weights[exits]) @ position_age[exits]) * nav
        )
        closed_dollars += float(np.abs(weights[exits]).sum()) * nav
        position_age[exits] = 0
        pnl = long_pnl + short_pnl + cash_return - borrow - exit_fee
        if pnl <= -1:
            raise ValueError("Capacity replay became insolvent")
        carried = np.where(valid, weights * (1 + np.nan_to_num(raw)), 0) / (
            1 + pnl
        )
        # Pending orders represent fixed return-adjusted quantities. Mark their
        # notional with the price path; reaching a target ends that order rather
        # than silently rebalancing the same weights every subsequent day.
        desired = np.where(valid, desired * (1 + np.nan_to_num(raw)), 0) / (
            1 + pnl
        )
        nav *= 1 + pnl
        if observer is not None:
            observer.mark(d, raw, valid, nav, carried)
        if d in schedule and decision is not None:
            composed = hasattr(decision, "resolve")
            desired = np.asarray(
                decision.resolve(nav) if composed else decision, dtype=float
            ).copy()
            if adaptive and not composed:
                desired *= decision_nav / nav
            if recorded_targets is not None:
                recorded_targets[d] = desired.copy()
                recorded_renewals[d] = (
                    decision.updated_mask.copy()
                    if composed
                    else np.ones(names, dtype=bool)
                )
            desired *= settings.get("execution_gross_buffer", 0.98)
            desired[np.abs(desired) <= 1e-10] = 0
            resizing = (carried * desired > 0) & (np.abs(carried) > 1e-10)
            skip_resize = resizing & (
                np.abs(desired - carried) < settings["execution_no_trade_band"]
            )
            desired[skip_resize] = carried[skip_resize]
            if composed:
                last_order[decision.updated_mask] = d
            else:
                last_order[:] = d
        short_limit = (
            np.minimum(
                cap * settings["max_borrow_fraction_market_cap"],
                adv * settings["max_borrow_adv_days"],
            )
            / nav
        )
        shortable = (cap >= 500_000_000) & (adv > 0)
        rejected_shorts += int(np.count_nonzero((desired < 0) & ~shortable))
        target = np.maximum(desired, -short_limit)
        target[(target < 0) & ~shortable] = 0
        target = np.clip(
            target,
            -settings["single_name_limit"],
            settings["single_name_limit"],
        )
        if np.abs(carried).sum() > settings["gross_limit"] * 0.99:
            target *= min(
                1,
                settings["gross_limit"]
                * 0.95
                / max(np.abs(target).sum(), 1e-30),
            )
        delta = target - carried
        increasing = (
            (np.abs(target) > np.abs(carried)) & (target * carried >= 0)
        ) | (target * carried < 0)
        expired = d - last_order >= settings["order_ttl_sessions"]
        if expired.any():
            expired_orders += int(
                np.count_nonzero(expired & increasing & (np.abs(delta) > 1e-8))
            )
            close = expired & (carried * target < 0)
            delta[close] = -carried[close]
            delta[expired & increasing & ~close] = 0
        daily_cap = adv * settings["max_daily_participation"] / nav
        fills = np.sign(delta) * np.minimum(np.abs(delta), daily_cap)
        fills[~market.tradable[d]] = 0
        # Finish exits before opening a new name when all 500 slots are occupied.
        after = carried + fills
        incumbent = carried != 0
        remaining = int(np.count_nonzero(incumbent & (after != 0)))
        entries = np.flatnonzero(~incumbent & (fills != 0))
        slots = max(0, settings["universe_size"] - remaining)
        if len(entries) > slots:
            ranked = entries[np.argsort(-np.abs(delta[entries]), kind="stable")]
            fills[ranked[slots:]] = 0
        # Keep partial long/short fills inside the mandate's net exposure band.
        proposed_net = float((carried + fills).sum())
        net_limit = settings.get("execution_max_absolute_net", 0.49)
        if abs(proposed_net) > net_limit:
            direction = 1 if proposed_net > 0 else -1
            adjustable = fills * direction > 0
            excess = abs(proposed_net) - net_limit
            available = np.abs(fills[adjustable]).sum()
            if available > 0:
                fills[adjustable] *= max(0, 1 - excess / available)
        if (
            np.abs(carried + fills).sum() > settings["gross_limit"] * 0.995
            and np.abs(carried).sum() <= settings["gross_limit"] * 0.995
        ):
            low, high = 0.0, 1.0
            for _ in range(20):
                mid = (low + high) / 2
                if (
                    np.abs(carried + mid * fills).sum()
                    > settings["gross_limit"] * 0.995
                ):
                    high = mid
                else:
                    low = mid
            fills *= low
        # Risk clipping can turn a complete exit into a partial exit. Recheck
        # slots after those adjustments before admitting replacement names.
        while True:
            after = carried + fills
            remaining = int(np.count_nonzero(incumbent & (after != 0)))
            entries = np.flatnonzero(~incumbent & (fills != 0))
            slots = max(0, settings["universe_size"] - remaining)
            if len(entries) <= slots:
                break
            ranked = entries[np.argsort(-np.abs(delta[entries]), kind="stable")]
            fills[ranked[slots:]] = 0
            proposed_net = float((carried + fills).sum())
            if abs(proposed_net) > net_limit:
                direction = 1 if proposed_net > 0 else -1
                adjustable = fills * direction > 0
                available = np.abs(fills[adjustable]).sum()
                if available > 0:
                    fills[adjustable] *= max(
                        0, 1 - (abs(proposed_net) - net_limit) / available
                    )
        dollars = np.abs(fills) * nav
        participation = np.divide(
            dollars, adv, out=np.zeros(names), where=adv > 0
        )
        maximum_participation = max(
            maximum_participation, float(participation.max())
        )
        impact = settings["impact_eta"] * sigma * np.sqrt(participation)
        fee_paid = float(np.abs(fills) @ (base + impact))
        if fee_paid >= 1:
            raise ValueError("Execution costs exhausted NAV")
        closed = np.where(
            carried * fills < 0, np.minimum(np.abs(carried), np.abs(fills)), 0
        )
        closed_dollar_days += float(closed @ position_age) * nav
        closed_dollars += float(closed.sum()) * nav
        after = carried + fills
        old_remaining = np.maximum(0, np.abs(carried) - closed)
        same_sign = after * carried > 0
        position_age = np.divide(
            position_age * old_remaining,
            np.abs(after),
            out=np.zeros(names),
            where=same_sign & (np.abs(after) > 1e-12),
        )
        if observer is not None:
            observer.fills(d, nav, carried, fills, after)
        weights = after / (1 - fee_paid)
        if np.count_nonzero(weights) > settings["universe_size"]:
            raise ValueError("Executed holdings exceed the position ceiling")
        desired /= 1 - fee_paid
        transaction_dollars += fee_paid * nav
        nav *= 1 - fee_paid
        net[d] = (1 + pnl) * (1 - fee_paid) - 1
        fees_series[d] = exit_fee + fee_paid * (1 + pnl)
        turnover[d] = np.abs(fills).sum() + exit_turnover
        costs[d], borrow_costs[d] = fee_paid + exit_fee, borrow
        pending[d] = float(np.abs(target - weights).sum() * nav)
        counts[d] = np.count_nonzero(np.abs(weights) > 1e-10)
        gross_series[d], net_series[d] = np.abs(weights).sum(), weights.sum()
        largest_name[d] = np.abs(weights).max()
        if held is not None:
            held[d] = weights
        previous_date = date
    return {
        "net": net,
        "turnover": turnover,
        "holdings": held,
        "execution_targets": recorded_targets,
        "execution_renewal_masks": recorded_renewals,
        "attribution": {
            "long_return": long_pnl_series,
            "short_return": short_pnl_series,
            "cash_return": cash_series,
            "trading_cost_return": fees_series,
            "borrow_cost_return": borrow_series,
        },
        "execution": {
            "initial_nav_usd": initial_nav,
            "cash_income_usd": cash_income_dollars,
            "average_closed_notional_holding_calendar_days": closed_dollar_days
            / closed_dollars
            if closed_dollars > 0
            else None,
            "closed_notional_usd": closed_dollars,
            "final_nav_usd": nav,
            "max_daily_participation": maximum_participation,
            "maximum_positions": int(counts.max()),
            "maximum_gross": float(gross_series.max()),
            "maximum_absolute_net": float(np.abs(net_series).max()),
            "largest_position_weight": float(largest_name.max()),
            "minimum_invested_positions": int(counts[counts > 0].min())
            if np.any(counts > 0)
            else 0,
            "gross_limit_breach_sessions": int(
                (gross_series > settings["gross_limit"] + 1e-6).sum()
            ),
            "net_limit_breach_sessions": int(
                (np.abs(net_series) > 0.5 + 1e-6).sum()
            ),
            "single_name_limit_breach_sessions": int(
                (largest_name > settings["single_name_limit"] + 1e-6).sum()
            ),
            "position_count_breach_sessions": int(
                (
                    (counts > settings["universe_size"])
                    | (
                        (np.arange(days) >= active_from)
                        & (counts < settings.get("min_positions", 0))
                    )
                ).sum()
            ),
            "transaction_cost_usd": transaction_dollars,
            "borrow_cost_usd": borrow_dollars,
            "transaction_cost_nav_fractions_sum": float(costs.sum()),
            "borrow_cost_nav_fractions_sum": float(borrow_costs.sum()),
            "pending_notional_at_end_usd": float(pending[-1]),
            "expired_entry_order_observations": expired_orders,
            "rejected_ineligible_short_observations": rejected_shorts,
            "borrow_source": "disclosed_size_fee_and_inventory_proxy_no_locate_quotes",
            "cost_model": "spread_plus_eta_sigma_sqrt_participation",
            "adv_source": "formation_known_126_session_dollar_volume",
            "open_positions_marked_at_training_end": True,
        },
    }
