"""Stock-specific dollar capacity constraints for portfolio backtests.

The module separates three ideas that should not be conflated:

* market-impact capacity is estimated from each security's own dollar volume;
* a short additionally needs borrow/locate capacity; and
* portfolio beta can be hedged elsewhere, but that does not make an illiquid
  position executable.

All inputs are expected to be point-in-time values known at the rebalance.  The
participation rates are research assumptions, not claims about a universal
market standard, and should be calibrated to the intended execution process.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CapacityConfig:
    """Configuration for per-security dollar capacity.

    With the defaults, a long can trade or be liquidated over five sessions at
    5% of ADV per session, while a short uses 2%.  Thus a stock with $1 million
    ADV has a $250,000 long capacity and a $100,000 short market-volume capacity.
    A confirmed locate may reduce the short capacity further.
    """

    nav: float
    execution_days: int = 5
    liquidation_days: int = 5
    long_participation: float = 0.05
    short_participation: float = 0.02
    min_long_price: float = 1.0
    min_short_price: float = 5.0
    max_borrow_fee_bps: float | None = None
    require_locate: bool = False
    adv_col: str = "dolvol_126d"
    locate_col: str = "locate_dollars"
    borrow_fee_col: str = "borrow_fee_bps"

    def __post_init__(self) -> None:
        if not isfinite(self.nav) or self.nav <= 0:
            raise ValueError("nav must be a positive finite dollar amount")
        if self.execution_days <= 0 or self.liquidation_days <= 0:
            raise ValueError("execution_days and liquidation_days must be positive")
        for name, value in (
            ("long_participation", self.long_participation),
            ("short_participation", self.short_participation),
        ):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.min_long_price < 0 or self.min_short_price < 0:
            raise ValueError("minimum prices cannot be negative")
        if self.max_borrow_fee_bps is not None and self.max_borrow_fee_bps < 0:
            raise ValueError("max_borrow_fee_bps cannot be negative")


def _finite_nonnegative(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return max(number, 0.0)


def _clip_delta(delta: float, capacity: float) -> float:
    return float(np.clip(delta, -capacity, capacity))


def _short_capacity(
    row: pd.Series,
    market_position_cap: float,
    market_trade_cap: float,
    config: CapacityConfig,
) -> tuple[float, float, str, str | None]:
    """Return short position/trade caps, borrow status, and blocking reason."""
    locate = (
        _finite_nonnegative(row.get(config.locate_col))
        if config.locate_col in row.index
        else None
    )
    fee = (
        _finite_nonnegative(row.get(config.borrow_fee_col))
        if config.borrow_fee_col in row.index
        else None
    )

    if config.max_borrow_fee_bps is not None and fee is not None:
        if fee > config.max_borrow_fee_bps:
            return 0.0, market_trade_cap, "fee_blocked", "borrow_fee"

    if locate is None:
        if config.require_locate:
            return 0.0, market_trade_cap, "blocked_missing", "missing_locate"
        return market_position_cap, market_trade_cap, "proxy_unverified", None

    # Locate is treated as a cap on the total short holding, not as daily flow.
    return min(market_position_cap, locate), market_trade_cap, "confirmed", None


def _apply_row(
    row: pd.Series,
    previous_dollars: Mapping[int, float],
    config: CapacityConfig,
) -> dict[str, object]:
    permno = int(row["permno"])
    target_weight = float(row["target_weight"])
    if not np.isfinite(target_weight):
        raise ValueError(f"target_weight must be finite for permno {permno}")

    previous = float(previous_dollars.get(permno, 0.0))
    if not np.isfinite(previous):
        raise ValueError(f"previous dollars must be finite for permno {permno}")

    requested = target_weight * config.nav
    requested_trade = requested - previous
    adv = _finite_nonnegative(row.get(config.adv_col))
    price = _finite_nonnegative(row.get("prc"))
    price = price if price is not None else 0.0

    if adv is None or adv <= 0:
        market_position_cap = 0.0
        trade_cap = 0.0
        position_cap = 0.0
        borrow_status = "not_applicable" if requested >= 0 else (
            "blocked_missing" if config.require_locate else "proxy_unverified"
        )
        final = previous
        executed_trade = 0.0
        reason = "missing_adv"
    else:
        target_is_short = requested < 0
        participation = (
            config.short_participation if target_is_short else config.long_participation
        )
        market_position_cap = adv * participation * config.liquidation_days
        trade_cap = adv * participation * config.execution_days
        position_cap = market_position_cap
        borrow_status = "not_applicable"
        borrow_reason: str | None = None

        if target_is_short:
            position_cap, trade_cap, borrow_status, borrow_reason = _short_capacity(
                row, market_position_cap, trade_cap, config
            )

        price_reason: str | None = None
        if requested > 0 and price < config.min_long_price:
            position_cap = 0.0
            price_reason = "long_price"
        elif requested < 0 and price < config.min_short_price:
            position_cap = 0.0
            price_reason = "short_price"

        # A lower current capacity estimate blocks added exposure; it does not
        # make a risk-reducing trade impossible. Existing borrowed shares may be
        # held or covered even when no fresh locate is available today.
        if target_is_short and previous < 0:
            position_cap = max(position_cap, abs(previous))

        crosses_zero = previous != 0 and requested != 0 and np.sign(previous) != np.sign(requested)
        if crosses_zero:
            # Use one shared ordinary-liquidity budget for the whole crossing.
            # Closing exposure comes first; only the unused dollars may open the
            # opposite side, which still has its own holding/borrow constraints.
            entry_trade_cap = trade_cap
            crossing_cap = adv * config.long_participation * config.execution_days
            close_dollars = min(abs(previous), crossing_cap)
            after_close = previous - np.sign(previous) * close_dollars
            if abs(after_close) > 1e-12:
                final = after_close
                reason = "crossing_exit_capacity"
            else:
                remaining = max(crossing_cap - close_dollars, 0.0)
                entry_dollars = min(
                    abs(requested), position_cap, entry_trade_cap, remaining
                )
                final = float(np.sign(requested) * entry_dollars)
                if np.isclose(final, requested):
                    reason = "none"
                elif borrow_reason is not None and entry_dollars == 0:
                    reason = borrow_reason
                elif price_reason is not None and entry_dollars == 0:
                    reason = price_reason
                elif target_is_short:
                    locate = _finite_nonnegative(row.get(config.locate_col))
                    if locate is not None and locate < market_position_cap:
                        reason = "locate"
                    else:
                        reason = "crossing_trade_capacity"
                else:
                    reason = "crossing_trade_capacity"
            executed_trade = final - previous
            trade_cap = crossing_cap
        else:
            capped_target = float(np.clip(requested, -position_cap, position_cap))
            desired_trade = capped_target - previous
            if previous < 0 and desired_trade > 0:
                # Buying to cover reduces risk and requires no fresh borrow.
                trade_cap = adv * config.long_participation * config.execution_days
            executed_trade = _clip_delta(desired_trade, trade_cap)
            final = previous + executed_trade

            constrained_by_position = not np.isclose(capped_target, requested)
            constrained_by_trade = not np.isclose(executed_trade, desired_trade)
            if borrow_reason is not None and (constrained_by_position or constrained_by_trade):
                reason = borrow_reason
            elif price_reason is not None and (constrained_by_position or constrained_by_trade):
                reason = price_reason
            elif target_is_short and constrained_by_position:
                locate = _finite_nonnegative(row.get(config.locate_col))
                if locate is not None and locate < market_position_cap:
                    reason = "locate"
                else:
                    reason = "short_liquidity"
            elif constrained_by_position or constrained_by_trade:
                reason = "short_liquidity" if target_is_short else "long_liquidity"
            else:
                reason = "none"

    result = row.to_dict()
    result.update(
        {
            "adv_dollars": float(adv) if adv is not None else np.nan,
            "requested_dollars": float(requested),
            "previous_dollars": float(previous),
            "requested_trade_dollars": float(requested_trade),
            "market_position_cap_dollars": float(market_position_cap),
            "position_cap_dollars": float(position_cap),
            "trade_cap_dollars": float(trade_cap),
            "executed_trade_dollars": float(executed_trade),
            "final_dollars": float(final),
            "final_weight": float(final / config.nav),
            "unfilled_dollars": float(requested - final),
            "borrow_status": borrow_status,
            "bind_reason": reason,
        }
    )
    return result


def apply_capacity_constraints(
    targets: pd.DataFrame,
    previous_dollars: Mapping[int, float],
    config: CapacityConfig,
) -> pd.DataFrame:
    """Apply stock-specific holding and trading limits to one rebalance.

    Parameters
    ----------
    targets:
        One row per security with ``permno``, ``target_weight``, ``prc``, and
        the configured ADV column. Optional locate and borrow-fee columns make
        short results executable rather than merely research proxies.
    previous_dollars:
        Actual dollar holdings immediately before this rebalance. Positive is
        long and negative is short. Capacity applies to the change from these
        holdings.
    config:
        Portfolio NAV and execution assumptions.

    Returns
    -------
    pandas.DataFrame
        The original fields plus requested, executable, unfilled, and binding-
        constraint columns, all in dollars, and the resulting weight.
    """
    required = {"permno", "target_weight", "prc", config.adv_col}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"targets is missing required columns: {sorted(missing)}")
    if targets["permno"].duplicated().any():
        duplicates = targets.loc[targets["permno"].duplicated(), "permno"].tolist()
        raise ValueError(f"targets contains duplicate permno values: {duplicates[:5]}")

    rows = [_apply_row(row, previous_dollars, config) for _, row in targets.iterrows()]
    return pd.DataFrame(rows, index=targets.index)
