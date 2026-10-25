"""Measured economics for the challenge bridge (no refitting, no invented constants).

* ``tail_membership`` reproduces the engine's declared tail-portfolio rule (long top ``tail_fraction`` by signed score,
  short bottom ``tail_fraction`` among market-cap-eligible names) so integrity can price trading costs on real turnover.
* ``calibrate_return_bridge`` fits the prediction -> decimal-return slope ONLY on the persisted walk-forward design
  predictions (each row was predicted by a model trained on strictly earlier, resolved months). It never scores the
  final frozen model on its own training months, which would be an in-sample fit.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Mapping, Sequence

from .builtins import stats
from .builtins.timeutil import end_of_day, month_end_known_at

MIN_ROWS_PER_MONTH = 10


class CalibrationError(ValueError):
    """The persisted predictions cannot support an honest return conversion."""


def tail_membership(
    rows: Sequence[Mapping[str, Any]],
    panel: Any,
    dataset: Any,
    bridge_sign: int,
) -> list[dict[str, Any]]:
    """Per formation date: the securities traded long / short by the declared tail rule (engine.evaluate semantics)."""
    ev = dataset.spec.evaluation
    cap_col = dataset.spec.market_cap_column
    cap = panel.num[cap_col] if cap_col else None
    floor = ev.short_min_market_cap_usd / dataset.spec.market_cap_to_usd
    by_month: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_month.setdefault(int(row["m"]), []).append(row)
    out: list[dict[str, Any]] = []
    for month in sorted(by_month):
        signed = {r["s"]: bridge_sign * float(r["p"]) for r in by_month[month]}
        q = max(1, int(len(signed) * ev.tail_fraction))
        longs = sorted(signed, key=lambda s: (-signed[s], s))[:q]

        def eligible(s: str) -> bool:
            if cap is None:
                return True
            i = panel.row_index.get((s, month))
            return i is not None and cap[i] is not None and cap[i] >= floor

        shorts = sorted(
            (s for s in signed if eligible(s)), key=lambda s: (signed[s], s)
        )[:q]
        if shorts:
            out.append(
                {
                    "formation_at": month_end_known_at(month).isoformat(),
                    "longs": longs,
                    "shorts": shorts,
                }
            )
    return out


def calibrate_return_bridge(
    predictions: Sequence[Mapping[str, Any]],
    *,
    bridge_sign: int,
    design_end: dt.date,
    min_months: int,
) -> dict[str, Any]:
    """Month-demeaned OLS of ``bridge_sign x next-period excess return`` on the out-of-training prediction.

    Returns ``slope`` (decimal return per unit native prediction), ``slope_se`` (across monthly slopes) and the measured
    ``residual_sd`` (within-month residual, degrees of freedom adjusted). Raises ``CalibrationError`` when rows are not
    out-of-training / not resolved inside the design window, or when the evidence is too thin.
    """
    cutoff = end_of_day(design_end)
    by_month: dict[int, list[tuple[float, float]]] = {}
    for row in predictions:
        p, r = row.get("p"), row.get("r")
        if (
            p is None
            or r is None
            or not math.isfinite(p)
            or not math.isfinite(r)
        ):
            raise CalibrationError(
                "a persisted design row has no finite prediction / realised return"
            )
        if dt.datetime.fromisoformat(row["x"]) > cutoff:
            raise CalibrationError(
                "a design row resolves after the design window; calibration would leak"
            )
        by_month.setdefault(int(row["m"]), []).append(
            (float(p), bridge_sign * float(r))
        )
    sxx = sxy = 0.0
    slopes: list[float] = []
    demeaned: list[tuple[float, float]] = []
    for month in sorted(by_month):
        pts = by_month[month]
        if len(pts) < MIN_ROWS_PER_MONTH:
            continue
        mp, my = (
            math.fsum(x for x, _ in pts) / len(pts),
            math.fsum(y for _, y in pts) / len(pts),
        )
        mxx = math.fsum((x - mp) ** 2 for x, _ in pts)
        mxy = math.fsum((x - mp) * (y - my) for x, y in pts)
        if mxx <= 1e-300:
            continue
        sxx, sxy = sxx + mxx, sxy + mxy
        slopes.append(mxy / mxx)
        demeaned.extend((x - mp, y - my) for x, y in pts)
    n = len(demeaned)
    if len(slopes) < max(3, min_months) or sxx <= 0.0 or n <= len(slopes) + 1:
        raise CalibrationError(
            f"only {len(slopes)} usable design months; need {max(3, min_months)} for a measured return conversion"
        )
    slope = sxy / sxx
    sse = math.fsum((y - slope * x) ** 2 for x, y in demeaned)
    resid_sd = math.sqrt(sse / (n - len(slopes) - 1))
    se = stats.stdev(slopes)
    if not (
        math.isfinite(slope)
        and math.isfinite(resid_sd)
        and resid_sd > 0.0
        and se is not None
    ):
        raise CalibrationError(
            "return conversion is degenerate (non-finite slope or zero residual dispersion)"
        )
    return {
        "slope": slope,
        "slope_se": se / math.sqrt(len(slopes)),
        "residual_sd": resid_sd,
        "n_rows": n,
        "n_months": len(slopes),
        "method": "month_demeaned_ols_on_persisted_walk_forward_design_predictions",
        "in_sample_fit": False,
    }
