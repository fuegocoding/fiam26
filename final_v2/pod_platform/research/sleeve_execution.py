"""Actual sleeve positions and native-review order composition for one fund."""

from __future__ import annotations

import numpy as np


class ComposedInstruction:
    """New reviewed dollars plus carried pending quantities of other sleeves."""

    def __init__(self, book, updated_mask):
        self.book = book
        self.updated_mask = updated_mask

    def resolve(self, current_nav):
        self.book.compress(current_nav)
        return (
            sum(self.book.targets.values(), np.zeros(self.book.names))
            / current_nav
        )


class SleeveExecutionBook:
    def __init__(
        self,
        names,
        *,
        gross_buffer=1.0,
        order_ttl_sessions=10,
        maximum_positions=None,
    ):
        if not 0 < gross_buffer <= 1 or order_ttl_sessions < 1:
            raise ValueError("Positive gross buffer and order TTL are required")
        self.names = names
        self.maximum_positions = maximum_positions
        self.compression_events = []
        self.positions, self.targets, self.updated = {}, {}, {}
        self.capital = {}
        self.gross_buffer = gross_buffer
        self.order_ttl_sessions = order_ttl_sessions
        self.pnl = {}
        self.reconciliation_max_usd = 0.0

    def set_targets(self, day, changes, capital=None):
        self.expire(day)
        mask = np.zeros(self.names, bool)
        for identity, target in changes.items():
            target = np.asarray(target, float).copy()
            old = self.targets.get(identity, np.zeros(self.names))
            position = self.positions.setdefault(identity, np.zeros(self.names))
            mask |= (
                (np.abs(old) > 1e-8)
                | (np.abs(target) > 1e-8)
                | (np.abs(position) > 1e-8)
            )
            self.targets[identity] = target
            self.updated[identity] = day
        if capital is not None:
            self.capital = dict(capital)
        return ComposedInstruction(self, mask)

    def compress(self, nav):
        """Net sleeves first, then select actual fund slots without new signs.

        Compression is an explicit construction overlay, not an alteration of
        the research forecast. Kept long/short dollars are scaled only downward
        to preserve dollar neutrality, and all removed exposure is recorded.
        """
        if self.maximum_positions is None or not self.targets:
            return
        combined = sum(self.targets.values(), np.zeros(self.names))
        active = np.abs(combined) > 1e-8
        if active.sum() <= self.maximum_positions:
            return
        longs, shorts = (
            np.flatnonzero(combined > 1e-8),
            np.flatnonzero(combined < -1e-8),
        )
        half = self.maximum_positions // 2
        chosen_long = longs[np.argsort(-combined[longs], kind="stable")][:half]
        chosen_short = shorts[np.argsort(combined[shorts], kind="stable")][
            :half
        ]
        scale = np.zeros(self.names)
        long_sum, short_sum = (
            combined[chosen_long].sum(),
            -combined[chosen_short].sum(),
        )
        common = min(long_sum, short_sum)
        if common > 0:
            scale[chosen_long] = common / long_sum
            scale[chosen_short] = common / short_sum
        removed = float(np.abs(combined * (1 - scale)).sum())
        for target in self.targets.values():
            target *= scale
        self.compression_events.append(
            {
                "pre_net_positions": int(active.sum()),
                "post_net_positions": int(np.count_nonzero(scale)),
                "gross_removed_usd": removed,
                "nav_usd": nav,
            }
        )

    def expire(self, day):
        """Cancel expired sleeve entries before unrelated orders are netted.

        Actual positions are buffered dollars, while targets precede the fund's
        gross buffer. Keep reductions executable, including a reversal's close
        to zero, but never revive an expired entry through a shared symbol.
        """
        for identity, target in self.targets.items():
            if day - self.updated[identity] < self.order_ttl_sessions:
                continue
            position = self.positions.get(identity, np.zeros(self.names))
            desired = target * self.gross_buffer
            same_side = desired * position >= 0
            retained = np.where(
                same_side,
                np.sign(desired)
                * np.minimum(np.abs(desired), np.abs(position)),
                0,
            )
            target[:] = retained / self.gross_buffer

    def mark(self, day, raw, valid, nav, carried):
        growth = np.where(valid, 1 + np.nan_to_num(raw), 0)
        for identity, position in self.positions.items():
            profit = float(position[valid] @ raw[valid])
            self.pnl[identity] = self.pnl.get(identity, 0.0) + profit
            position *= growth
        for identity, target in self.targets.items():
            if self.updated.get(identity) != day:
                target *= growth
        reconciled = sum(self.positions.values(), np.zeros(self.names))
        gap = nav * carried - reconciled
        self.reconciliation_max_usd = max(
            self.reconciliation_max_usd, float(np.abs(gap).max())
        )
        # Roundoff and execution risk overrides belong to an explicit overlay,
        # rather than being silently credited to an alpha sleeve.
        self.positions.setdefault("risk_overlay", np.zeros(self.names))[:] += (
            gap
        )
        self.expire(day)

    def fills(self, day, nav, carried, fills, after):
        identities = sorted(set(self.positions) | set(self.targets))
        if not identities:
            return
        positions = np.array(
            [self.positions.get(i, np.zeros(self.names)) for i in identities]
        )
        desired = np.array(
            [
                self.targets.get(i, np.zeros(self.names)) * self.gross_buffer
                for i in identities
            ]
        )
        delta = desired - positions
        buys, sells = np.maximum(delta, 0), np.maximum(-delta, 0)
        buy_sum, sell_sum = buys.sum(axis=0), sells.sum(axis=0)
        crossed = np.minimum(buy_sum, sell_sum)
        allocated = (
            np.divide(buys, buy_sum, out=np.zeros_like(buys), where=buy_sum > 0)
            * crossed
            - np.divide(
                sells, sell_sum, out=np.zeros_like(sells), where=sell_sum > 0
            )
            * crossed
        )
        outstanding = delta - allocated
        buy, sell = np.maximum(outstanding, 0), np.maximum(-outstanding, 0)
        buy_total, sell_total = buy.sum(axis=0), sell.sum(axis=0)
        actual = fills * nav
        allocated += np.divide(
            buy, buy_total, out=np.zeros_like(buy), where=buy_total > 0
        ) * np.maximum(actual, 0)
        allocated -= np.divide(
            sell, sell_total, out=np.zeros_like(sell), where=sell_total > 0
        ) * np.maximum(-actual, 0)
        for identity, position, movement in zip(
            identities, positions, allocated
        ):
            self.positions[identity] = position + movement
        discrepancy = after * nav - sum(
            self.positions.values(), np.zeros(self.names)
        )
        self.positions.setdefault("risk_overlay", np.zeros(self.names))[:] += (
            discrepancy
        )


class RecordedInstruction:
    """A saved close instruction with the exact per-name order renewal mask."""

    def __init__(self, target, updated_mask):
        self.target = target
        self.updated_mask = updated_mask

    def resolve(self, current_nav):
        return self.target
