"""Full observed-price fills and daily accounting, without execution modeling."""

from __future__ import annotations

import numpy as np

from .mandate import check_book, frame_sectors, sector_codes


def trim_observed_risk(target, tradable, settings, *, sectors=None):
    """Scale observed holdings down at the close to respect fund risk limits."""
    adjustable = np.asarray(tradable, bool)
    fixed, variable = target[~adjustable], target[adjustable]
    upper = 1.0
    gross = np.abs(variable).sum()
    if gross:
        upper = min(upper, (settings.get("gross_limit", 2.0) - np.abs(fixed).sum()) / gross)
    largest = np.abs(variable).max(initial=0)
    if largest:
        upper = min(upper, settings.get("single_name_limit", .05) / largest)
    fixed_net, variable_net = fixed.sum(), variable.sum()
    lower = 0.0
    if abs(variable_net) > 1e-14:
        bounds = sorted(((-.5-fixed_net)/variable_net, (.5-fixed_net)/variable_net))
        lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
    if sectors is not None:
        codes = sector_codes(sectors)
        for code in np.unique(codes):
            local = codes == code
            fixed_sector = target[local & ~adjustable].sum()
            variable_sector = target[local & adjustable].sum()
            limit = settings.get('sector_net_limit', .2)
            if abs(variable_sector) > 1e-14:
                bounds = sorted(((-limit-fixed_sector)/variable_sector,
                                 (limit-fixed_sector)/variable_sector))
                lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
            elif abs(fixed_sector) > limit + 1e-7:
                return target, 1.0
    # Missing quotes cannot be used to invent a feasible rebalance. Any
    # remaining fixed-position breach is reported by the normal daily audit.
    if upper < max(0.0, lower):
        return target, 1.0
    scale = min(1.0, max(0.0, upper))
    if scale >= 1.0 - 1e-12:
        return target, 1.0
    result = target.copy()
    result[adjustable] *= scale
    return result, scale


def replay_at_observed_prices(market, schedule, settings, *, keep_holdings=False):
    """Mark carried shares, then fill scheduled targets in full at the close.

    Only supplied quotes may be traded. No volume, inventory, latency, expiry,
    partial-fill or price adjustment is assumed. A fill earns subsequent returns.
    """
    days, names = market.returns.shape
    weights, age = np.zeros(names), np.zeros(names)
    trading_age = np.zeros(names)
    closed_dollar_trading_days = 0.0
    closed_dollar_days = closed_dollars = 0.0
    net, turnover, long_returns, short_returns, cash_returns = (
        np.zeros(days) for _ in range(5)
    )
    counts, gross, absolute_net, largest = (np.zeros(days) for _ in range(4))
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
    recorded = {} if keep_holdings else None
    nav = float(settings.get(
        "execution_start_nav", settings.get("strategy_capital_usd", 1.0)
    ))
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("Starting NAV must be finite and positive")
    initial_nav = nav
    observer = settings.get("execution_observer")
    active_from = min(schedule, default=days)
    rejected_shorts = unavailable_quotes = 0
    risk_trim_sessions = 0
    mandate_rejections = []
    cash_income = 0.0
    previous_date = market.dates[0] if days else None
    for day, date in enumerate(market.dates):
        # Adaptive decisions see the previous NAV, before today's return.
        decision_nav = nav
        instruction = schedule.get(day)
        adaptive = callable(instruction)
        decision = instruction(nav, weights.copy()) if adaptive else instruction
        raw = market.returns[day]
        valid = np.isfinite(raw) & (raw >= -1)
        missing = ~valid & (weights != 0)
        exit_turnover = float(np.abs(weights[missing]).sum())
        age += (date - previous_date).days
        trading_age += 1
        closed_dollar_trading_days += float(np.abs(weights[missing]) @ trading_age[missing]) * nav
        closed_dollar_days += float(np.abs(weights[missing]) @ age[missing]) * nav
        closed_dollars += exit_turnover * nav
        age[missing] = 0
        trading_age[missing] = 0
        long_returns[day] = float(np.maximum(weights[valid], 0) @ raw[valid])
        short_returns[day] = float(np.minimum(weights[valid], 0) @ raw[valid])
        if market.cash_returns is not None and day >= active_from:
            cash_returns[day] = (1 - weights.sum()) * market.cash_returns[day]
        cash_income += cash_returns[day] * nav
        pnl = long_returns[day] + short_returns[day] + cash_returns[day]
        if pnl <= -1:
            raise ValueError("Observed-price replay became insolvent")
        carried = np.where(
            valid, weights * (1 + np.nan_to_num(raw)), 0
        ) / (1 + pnl)
        nav *= 1 + pnl
        if observer is not None:
            observer.mark(day, raw, valid, nav, carried)
        target = carried.copy()
        formation = int(market.formation[day])
        frame = market.frames[formation] if market.frames is not None and formation >= 0 else None
        sectors = frame_sectors(frame, names) if settings.get('enforce_fund_mandate') else None
        if decision is not None:
            composed = hasattr(decision, "resolve")
            target = np.asarray(
                decision.resolve(nav) if composed else decision, dtype=float
            ).copy()
            if target.shape != (names,) or not np.isfinite(target).all():
                raise ValueError("Targets must be finite security vectors")
            if adaptive and not composed:
                target *= decision_nav / nav
            if sectors is not None:
                proposed_check = check_book(target, sectors, settings)
                if not proposed_check['pass']:
                    mandate_rejections.append({'date': str(date.date()), 'stage': 'proposed',
                                               'failures': proposed_check['failures']})
                    target = carried.copy()
                    if observer is not None and hasattr(observer, 'reject_targets'):
                        observer.reject_targets(day, nav, carried)
            # An absent quote cannot supply an invented transaction price.
            unavailable = ~market.tradable[day]
            unavailable_quotes += int(np.count_nonzero(
                unavailable & (target != carried)
            ))
            target[unavailable] = carried[unavailable]
            formation = int(market.formation[day])
            if (
                settings.get("validate_short_eligibility", True)
                and market.frames is not None
                and formation >= 0
            ):
                cap = (
                    market.frames[formation].me.fillna(0).to_numpy(float) * 1e6
                )
                ineligible = (target < 0) & (cap < 500e6)
                rejected_shorts += int(ineligible.sum())
                target[ineligible] = 0
            if sectors is not None:
                actual_check = check_book(target, sectors, settings)
                if not actual_check['pass'] and not np.array_equal(target, carried):
                    mandate_rejections.append({'date': str(date.date()), 'stage': 'post_quote',
                                               'failures': actual_check['failures']})
                    target = carried.copy()
                    if observer is not None and hasattr(observer, 'reject_targets'):
                        observer.reject_targets(day, nav, carried)
        original_target = target
        target, risk_scale = trim_observed_risk(target, market.tradable[day], settings, sectors=sectors)
        if risk_scale < 1:
            risk_trim_sessions += 1
            if observer is not None and hasattr(observer, "risk_adjustment"):
                observer.risk_adjustment(day, nav, original_target, target)
        if recorded is not None and (decision is not None or risk_scale < 1):
            recorded[day] = target.copy()
        if day == days - 1 and settings.get("liquidate_at_end", False):
            target[market.tradable[day]] = 0
            if recorded is not None:
                recorded[day] = target.copy()
        movement = target - carried
        closed = np.where(
            carried * movement < 0,
            np.minimum(np.abs(carried), np.abs(movement)),
            0,
        )
        closed_dollar_days += float(closed @ age) * nav
        closed_dollar_trading_days += float(closed @ trading_age) * nav
        closed_dollars += float(closed.sum()) * nav
        remaining = np.maximum(0, np.abs(carried) - closed)
        age = np.divide(
            age * remaining,
            np.abs(target),
            out=np.zeros(names),
            where=(target * carried > 0) & (np.abs(target) > 1e-12),
        )
        trading_age = np.divide(trading_age * remaining, np.abs(target),
            out=np.zeros(names), where=(target * carried > 0) & (np.abs(target) > 1e-12))
        if observer is not None:
            observer.fills(day, nav, carried, movement, target)
        weights = target
        net[day] = pnl
        turnover[day] = float(np.abs(movement).sum() + exit_turnover)
        counts[day] = np.count_nonzero(np.abs(weights) > 1e-10)
        gross[day] = np.abs(weights).sum()
        absolute_net[day] = abs(weights.sum())
        largest[day] = np.abs(weights).max(initial=0)
        if held is not None:
            held[day] = weights
        previous_date = date
    return {
        "net": net,
        "turnover": turnover,
        "holdings": held,
        "execution_targets": recorded,
        "attribution": {
            "long_return": long_returns,
            "short_return": short_returns,
            "cash_return": cash_returns,
        },
        "execution": {
            "fill_convention": "full_order_at_observed_price",
            "initial_nav_usd": initial_nav,
            "final_nav_usd": nav,
            "cash_income_usd": cash_income,
            "average_closed_notional_holding_calendar_days": (
                closed_dollar_days / closed_dollars if closed_dollars else None
            ),
            "closed_notional_usd": closed_dollars,
            "average_closed_notional_holding_trading_days": (
                closed_dollar_trading_days / closed_dollars if closed_dollars else None
            ),
            "maximum_positions": int(counts.max(initial=0)),
            "maximum_gross": float(gross.max(initial=0)),
            "maximum_absolute_net": float(absolute_net.max(initial=0)),
            "largest_position_weight": float(largest.max(initial=0)),
            "minimum_invested_positions": (
                int(counts[counts > 0].min()) if np.any(counts > 0) else 0
            ),
            "gross_limit_breach_sessions": int(
                (gross > settings.get("gross_limit", 2) + 1e-6).sum()
            ),
            "net_limit_breach_sessions": int(
                (absolute_net > 0.5 + 1e-6).sum()
            ),
            "single_name_limit_breach_sessions": int(
                (largest > settings.get("single_name_limit", 0.05) + 1e-6).sum()
            ),
            "position_count_breach_sessions": int((
                (counts > settings.get("universe_size", 500))
                | ((np.arange(days) >= active_from)
                   & (counts < settings.get("min_positions", 0)))
            ).sum()),
            "rejected_ineligible_short_observations": rejected_shorts,
            "unavailable_quote_observations": unavailable_quotes,
            "risk_trim_sessions": risk_trim_sessions,
            "rejected_mandate_targets": len(mandate_rejections),
            "mandate_rejections": mandate_rejections,
            "open_positions_marked_at_training_end": True,
        },
    }
