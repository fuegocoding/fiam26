"""Actual sleeve positions and native-review order composition for one fund."""

from __future__ import annotations

import numpy as np


class ComposedInstruction:
    """New reviewed dollars plus carried positions of other sleeves."""

    def __init__(self, book):
        self.book = book

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
        maximum_positions=None,
    ):
        self.names = names
        self.maximum_positions = maximum_positions
        self.compression_events = []
        self.positions, self.targets, self.updated = {}, {}, {}
        self.capital = {}
        self.pnl = {}
        self.reconciliation_max_usd = 0.0

    def set_targets(self, day, changes, capital=None):
        for identity, target in changes.items():
            target = np.asarray(target, float).copy()
            old = self.targets.get(identity, np.zeros(self.names))
            position = self.positions.setdefault(identity, np.zeros(self.names))
            self.targets[identity] = target
            self.updated[identity] = day
        if capital is not None:
            self.capital = dict(capital)
        return ComposedInstruction(self)

    def compress(self, nav):
        """Net sleeves first, then select actual fund slots without new signs.

        Compression is an explicit construction overlay. Keep the largest
        aggregate positions and scale only downward, preserving the intended
        long/short notional ratio. Directional books retain their funded side.
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
        indices = np.flatnonzero(active)
        chosen = indices[np.argsort(-np.abs(combined[indices]), kind="stable")][:self.maximum_positions]
        if len(longs) and len(shorts) and len(chosen) >= 2:
            for side in (longs, shorts):
                if not np.isin(chosen, side).any():
                    replacement = side[np.argmax(np.abs(combined[side]))]
                    chosen[np.argmin(np.abs(combined[chosen]))] = replacement
        chosen_long = chosen[combined[chosen] > 0]
        chosen_short = chosen[combined[chosen] < 0]
        scale = np.zeros(self.names)
        long_sum, short_sum = (
            combined[chosen_long].sum(),
            -combined[chosen_short].sum(),
        )
        full_long, full_short = combined[longs].sum(), -combined[shorts].sum()
        if full_long > 0 and full_short > 0:
            retained = min(long_sum / full_long, short_sum / full_short)
            if long_sum > 0:scale[chosen_long] = retained * full_long / long_sum
            if short_sum > 0:scale[chosen_short] = retained * full_short / short_sum
        else:
            scale[chosen] = 1.0
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

    def risk_adjustment(self, day, nav, before, after):
        """Attribute observed-price risk trims to the sleeves being reduced."""
        scale = np.divide(after, before, out=np.ones(self.names), where=np.abs(before)>1e-14)
        for identity, target in self.targets.items():
            target *= scale
            self.updated[identity] = day

    def reject_targets(self, day, nav, carried):
        """Cancel rejected orders while preserving actual sleeve positions."""
        self.targets = {identity: position.copy() for identity, position in self.positions.items()}
        self.updated = dict.fromkeys(self.targets, day)

    def fills(self, day, nav, carried, fills, after):
        identities = sorted(set(self.positions) | set(self.targets))
        if not identities:
            return
        positions = np.array(
            [self.positions.get(i, np.zeros(self.names)) for i in identities]
        )
        desired = np.array(
            [
                self.targets.get(i, np.zeros(self.names))
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
