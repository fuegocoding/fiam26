"""Judge-facing attribution diagnostics (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 18.6, 19.8, 19.11, 19.12, 28.1).

* factor-versus-idiosyncratic return, from holdings and diagnostic factor
  exposures, and from a regression of the portfolio series on factor series;
* selection versus sizing with a stated baseline, an explicit interaction
  convention and a seeded random-sizing control;
* hit rate and win/loss magnitude;
* an alpha-decay curve of dated post-entry payoff windows with explicit
  prerequisites - it refuses (``INSUFFICIENT``) rather than inventing a curve.

Factors here are *internal diagnostics*; none of this is a Fama-French
attribution. Selection/sizing is an accounting split with a declared order, not a
unique causal decomposition (spec 19.12). Inputs are caller-supplied; nothing is
fabricated or interpolated.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse

STANDARD_FACTOR_NOTE = "Internal diagnostic factors; not standard external (e.g. Fama-French) definitions."


def _fail(code: ErrorCode, summary: str, stop: str) -> None:
    raise PodPlatformError(ToolResponse.failure(summary=summary, code=code, stop_condition=stop))


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


# ------------------------------------------------- holdings factor attribution


@dataclass(frozen=True)
class HoldingsPeriod:
    label: str
    weights: Mapping[str, float]  # signed weights of NAV carried into the period
    returns: Mapping[str, float]  # security returns over the period, same basis as the factors


def holdings_factor_attribution(
    periods: Sequence[HoldingsPeriod],
    exposures: Mapping[str, Mapping[str, float]],
    factor_returns: Mapping[str, Mapping[str, float]],
    *,
    factor_set_label: str,
    period_exposures: Optional[Mapping[str, Mapping[str, Mapping[str, float]]]] = None,
) -> dict[str, Any]:
    """Per period: total = sum_f factor contribution + idiosyncratic, exactly.

    factor contribution_f = sum_i w_i * beta_{i,f} * F_f ; idiosyncratic = sum_i w_i * (r_i - sum_f beta_{i,f} F_f).
    A held name without an exposure vector, or a period without every factor return, fails closed
    (spec 16.3: missing exposures are never silently assigned precise values).
    """
    if not periods:
        raise ValueError("at least one period is required")
    if not factor_set_label:
        raise ValueError("a factor set label is required so the factors are never mistaken for a standard set")
    rows: list[dict[str, Any]] = []
    factor_names: Optional[list[str]] = None
    totals = {"total": 0.0, "idiosyncratic": 0.0}
    factor_totals: dict[str, float] = {}
    for period in periods:
        returns_f = factor_returns.get(period.label)
        if returns_f is None:
            _fail(ErrorCode.SCHEMA_MISMATCH, f"no factor returns for period {period.label}", "Attribution needs factor returns for every period.")
        names = sorted(returns_f)
        if factor_names is None:
            factor_names = names
        elif names != factor_names:
            _fail(ErrorCode.SCHEMA_MISMATCH, f"factor set changed in period {period.label}", "Use one factor set across periods.")
        table = (period_exposures or {}).get(period.label, exposures)
        contributions = {f: 0.0 for f in names}
        total = 0.0
        idio = 0.0
        for sid in sorted(period.weights):
            w = _finite(period.weights[sid])
            if w == 0.0:
                continue
            if sid not in period.returns:
                _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, f"no return for held {sid} in {period.label}",
                      "Held names are never zero-filled.")
            loadings = table.get(sid)
            if loadings is None or any(f not in loadings for f in names):
                _fail(ErrorCode.SCHEMA_MISMATCH, f"no complete exposure vector for held {sid} in {period.label}",
                      "Missing exposures are never silently assigned.")
            r = _finite(period.returns[sid])
            explained = 0.0
            for f in names:
                part = w * _finite(loadings[f]) * _finite(returns_f[f])
                contributions[f] += part
                explained += part
            total += w * r
            idio += w * r - explained
        factor_sum = sum(contributions.values())
        if abs(total - factor_sum - idio) > 1e-12 * max(1.0, abs(total)):
            _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, f"attribution identity failed in {period.label}",
                  "Factor plus idiosyncratic must reconcile to the period contribution.")
        rows.append({"period": period.label, "total": total, "factor_total": factor_sum, "idiosyncratic": idio,
                     "factor_contributions": contributions})
        totals["total"] += total
        totals["idiosyncratic"] += idio
        for f, v in contributions.items():
            factor_totals[f] = factor_totals.get(f, 0.0) + v
    factor_sum_all = sum(factor_totals.values())
    return {
        "method": "holdings_based_factor_vs_idiosyncratic",
        "factor_set_label": factor_set_label,
        "standard_factor_definitions": False,
        "note": STANDARD_FACTOR_NOTE,
        "periods": rows,
        "aggregate": {"sum_of_period_contributions": totals["total"], "factor": factor_sum_all, "idiosyncratic": totals["idiosyncratic"],
                      "factor_by_name": dict(sorted(factor_totals.items())),
                      "convention": "sum of period contributions (weights x returns); no compounding"},
    }


# ----------------------------------------------- regression factor attribution


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(matrix)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("factor series are collinear or constant; regression is undefined")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (a[i][n] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    cols = []
    for k in range(n):
        e = [1.0 if i == k else 0.0 for i in range(n)]
        cols.append(_solve(matrix, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def regression_factor_attribution(
    labels: Sequence[str],
    portfolio_excess: Sequence[float],
    factors: Mapping[str, Sequence[float]],
    *,
    factor_set_label: str,
) -> dict[str, Any]:
    """OLS with intercept; per observation: portfolio = alpha + sum beta_k f_k + residual (exact)."""
    n = len(portfolio_excess)
    names = sorted(factors)
    k = len(names)
    if not names:
        raise ValueError("at least one factor is required")
    if not factor_set_label:
        raise ValueError("a factor set label is required")
    if len(labels) != n or any(len(factors[f]) != n for f in names):
        raise ValueError("series length mismatch")
    if n <= k + 1:
        raise ValueError("more observations than parameters are required")
    y = [_finite(v) for v in portfolio_excess]
    x = [[1.0] + [_finite(factors[f][t]) for f in names] for t in range(n)]
    p = k + 1
    xtx = [[sum(x[t][i] * x[t][j] for t in range(n)) for j in range(p)] for i in range(p)]
    xty = [sum(x[t][i] * y[t] for t in range(n)) for i in range(p)]
    coef = _solve(xtx, xty)
    fitted = [sum(coef[i] * x[t][i] for i in range(p)) for t in range(n)]
    resid = [y[t] - fitted[t] for t in range(n)]
    sse = sum(r * r for r in resid)
    sigma2 = sse / (n - p)
    inv = _invert(xtx)
    se = [math.sqrt(max(0.0, sigma2 * inv[i][i])) for i in range(p)]
    mean_y = sum(y) / n
    sst = sum((v - mean_y) ** 2 for v in y)
    r2 = 1.0 - sse / sst if sst > 0 else None
    alpha = coef[0]
    betas = {f: coef[i + 1] for i, f in enumerate(names)}
    rows = []
    for t in range(n):
        factor_part = sum(betas[f] * _finite(factors[f][t]) for f in names)
        rows.append({"period": labels[t], "total": y[t], "factor": factor_part, "alpha": alpha, "idiosyncratic_residual": resid[t]})
        if abs(y[t] - factor_part - alpha - resid[t]) > 1e-12 * max(1.0, abs(y[t])):
            _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, f"regression identity failed at {labels[t]}", "Components must reconcile.")
    return {
        "method": "regression_factor_vs_idiosyncratic",
        "factor_set_label": factor_set_label,
        "standard_factor_definitions": False,
        "note": STANDARD_FACTOR_NOTE,
        "n": n,
        "alpha_per_period": alpha,
        "alpha_se": se[0],
        "betas": betas,
        "beta_se": {f: se[i + 1] for i, f in enumerate(names)},
        "r_squared": r2,
        "rows": rows,
        "aggregate": {
            "total": sum(r["total"] for r in rows),
            "factor": sum(r["factor"] for r in rows),
            "alpha": alpha * n,
            "idiosyncratic_residual": sum(r["idiosyncratic_residual"] for r in rows),
        },
        "uncertainty": "classical OLS standard errors; not dependence-adjusted",
    }


# ------------------------------------------------------- selection vs sizing


@dataclass(frozen=True)
class SizingPeriod:
    label: str
    weights: Mapping[str, float]  # signed actual weights
    returns: Mapping[str, float]


def _equal_within_side(weights: Mapping[str, float]) -> dict[str, float]:
    longs = [s for s, w in weights.items() if w > 0]
    shorts = [s for s, w in weights.items() if w < 0]
    long_gross = sum(weights[s] for s in longs)
    short_gross = sum(-weights[s] for s in shorts)
    out: dict[str, float] = {}
    for s in longs:
        out[s] = long_gross / len(longs)
    for s in shorts:
        out[s] = -short_gross / len(shorts)
    return out


def selection_vs_sizing(periods: Sequence[SizingPeriod], *, seed: int = 0, n_draws: int = 200) -> dict[str, Any]:
    """Selection = equal weight within each side (side gross preserved); sizing = actual - selection.

    The split is exact by construction (selection + sizing = actual) and its *order* (selection
    first) is a stated convention - not a unique causal decomposition. A seeded control permutes the
    absolute sizes among the same names within each side ``n_draws`` times to show how much of the
    sizing effect random sizing would also produce.
    """
    if not periods:
        raise ValueError("at least one period is required")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    rows: list[dict[str, Any]] = []
    sums = {"actual": 0.0, "selection": 0.0, "sizing": 0.0, "control_mean": 0.0}
    for period in periods:
        weights = {s: _finite(w) for s, w in period.weights.items() if _finite(w) != 0.0}
        if not weights:
            _fail(ErrorCode.SCHEMA_MISMATCH, f"period {period.label} has no positions", "Nothing to attribute.")
        for s in weights:
            if s not in period.returns:
                _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, f"no return for held {s} in {period.label}", "Held names are never zero-filled.")
        actual = sum(w * _finite(period.returns[s]) for s, w in weights.items())
        equal = _equal_within_side(weights)
        selection = sum(w * _finite(period.returns[s]) for s, w in equal.items())
        sizing = actual - selection
        rng = random.Random(f"{seed}:{period.label}")
        longs = sorted(s for s, w in weights.items() if w > 0)
        shorts = sorted(s for s, w in weights.items() if w < 0)
        draws: list[float] = []
        for _ in range(n_draws):
            total = 0.0
            for side, names in ((1.0, longs), (-1.0, shorts)):
                sizes = [abs(weights[s]) for s in names]
                rng.shuffle(sizes)
                total += sum(side * size * _finite(period.returns[s]) for s, size in zip(names, sizes))
            draws.append(total)
        control_mean = sum(draws) / len(draws)
        at_or_below = sum(1 for d in draws if d <= actual) / len(draws)
        rows.append({
            "period": period.label,
            "actual": actual,
            "selection": selection,
            "sizing": sizing,
            "random_sizing_control_mean": control_mean,
            "sizing_vs_random_control": actual - control_mean,
            "actual_percentile_in_control": at_or_below,
            "names": len(weights),
        })
        sums["actual"] += actual
        sums["selection"] += selection
        sums["sizing"] += sizing
        sums["control_mean"] += control_mean
    return {
        "method": "selection_vs_sizing",
        "baseline": "equal weight within each side, side gross preserved, same names",
        "order_convention": "selection first, sizing = actual - selection; interaction is inside sizing",
        "causal_claim": False,
        "control": {"kind": "seeded permutation of absolute sizes within side", "seed": seed, "draws": n_draws},
        "periods": rows,
        "aggregate": {**{k: v for k, v in sums.items()}, "sizing_vs_random_control": sums["actual"] - sums["control_mean"],
                      "convention": "sum of period contributions (weights x returns); no compounding"},
    }


# ----------------------------------------------------- hit rate and win/loss


@dataclass(frozen=True)
class PositionContribution:
    security_id: str
    period: str
    side: str  # "long" | "short"
    pnl: float  # contribution in a stated unit (return contribution or USD)


def _side_stats(values: Sequence[float]) -> dict[str, Any]:
    n = len(values)
    wins = [v for v in values if v > 0]
    losses = [v for v in values if v < 0]
    zeros = n - len(wins) - len(losses)
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = -sum(losses) / len(losses) if losses else None
    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "flat": zeros,
        "hit_rate": len(wins) / n if n else None,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "win_loss_ratio": (avg_win / avg_loss) if (avg_win is not None and avg_loss) else None,
        "expectancy": sum(values) / n if n else None,
        "total": sum(values),
    }


def hit_rate_win_loss(contributions: Sequence[PositionContribution], *, unit: str) -> dict[str, Any]:
    """Hit rate (share of position-periods with positive P&L) and win/loss magnitudes, all and by side."""
    if not contributions:
        raise ValueError("no position contributions were supplied")
    values = [_finite(c.pnl) for c in contributions]
    out: dict[str, Any] = {"method": "hit_rate_win_loss", "unit": unit, "all": _side_stats(values), "by_side": {}}
    for side in ("long", "short"):
        subset = [_finite(c.pnl) for c in contributions if c.side == side]
        if subset:
            out["by_side"][side] = _side_stats(subset)
    unknown = sorted({c.side for c in contributions} - {"long", "short"})
    if unknown:
        raise ValueError(f"unknown sides: {unknown}")
    out["note"] = "Position-periods are not independent bets; hit rate is descriptive, not a significance claim."
    return out


# --------------------------------------------------------- alpha-decay curve


@dataclass(frozen=True)
class DecayCohort:
    label: str
    entry_date: dt.date  # close of the executable entry; first earned return is the next trading day
    signals: Mapping[str, float]  # signed forecast per security (sign sets the position direction)


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            ranks[i] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


def alpha_decay_curve(
    cohorts: Sequence[DecayCohort],
    daily_returns: Mapping[tuple[str, dt.date], float],
    trading_days: Sequence[dt.date],
    windows: Sequence[tuple[int, int]],
    *,
    min_cohorts: int = 4,
    min_names: int = 20,
    drop_incomplete_names: bool = False,
) -> dict[str, Any]:
    """Signed payoff after executable entry by trading-day window (offset 1 = first earned day).

    Prerequisites (all reported): an explicit trading calendar; every cohort's entry date on it with
    enough later days; a supplied daily return for every (name, day) in the widest window (names
    missing any are dropped only under ``drop_incomplete_names`` and are then counted); at least
    ``min_names`` names per cohort and ``min_cohorts`` cohorts. Otherwise the result is
    ``INSUFFICIENT`` - no curve is produced. Cohorts at adjacent formations overlap in time, so the
    effective sample is smaller than the number of cohorts; standard errors are descriptive.
    """
    days = list(trading_days)
    if any(b <= a for a, b in zip(days, days[1:])):
        raise ValueError("trading_days must be strictly increasing")
    if not windows or any(a < 1 or b < a for a, b in windows):
        raise ValueError("windows are (start_offset, end_offset) with 1 <= start <= end")
    widest = max(b for _a, b in windows)
    index = {d: i for i, d in enumerate(days)}
    prerequisites = [
        "explicit trading calendar",
        f"each cohort entry date on the calendar with at least {widest} later trading days",
        f"a supplied daily return for every name on each of those days (no interpolation)",
        f"at least {min_names} names per cohort and {min_cohorts} cohorts",
    ]
    used: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    prepared: list[tuple[DecayCohort, dict[str, int], list[str]]] = []
    for cohort in cohorts:
        if cohort.entry_date not in index:
            excluded.append({"cohort": cohort.label, "reason": "entry date is not a trading day on the calendar"})
            continue
        i0 = index[cohort.entry_date]
        if i0 + widest >= len(days):
            excluded.append({"cohort": cohort.label, "reason": f"fewer than {widest} trading days after entry"})
            continue
        names = []
        dropped = 0
        for sid, signal in sorted(cohort.signals.items()):
            if _finite(signal) == 0.0:
                continue
            complete = all((sid, days[i0 + o]) in daily_returns for o in range(1, widest + 1))
            if complete:
                names.append(sid)
            elif drop_incomplete_names:
                dropped += 1
            else:
                _fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                      f"missing daily return for {sid} after entry {cohort.entry_date} (cohort {cohort.label})",
                      "Alpha decay never interpolates; supply the returns or opt in to dropping incomplete names.")
        if len(names) < min_names:
            excluded.append({"cohort": cohort.label, "reason": f"only {len(names)} complete names (< {min_names}); {dropped} dropped"})
            continue
        prepared.append((cohort, {"i0": i0, "dropped": dropped}, names))
    if len(prepared) < min_cohorts:
        return {"status": "INSUFFICIENT", "method": "alpha_decay_curve", "prerequisites": prerequisites,
                "cohorts_used": len(prepared), "cohorts_required": min_cohorts, "excluded": excluded,
                "note": "No curve is produced without the prerequisites; nothing is inferred."}

    def window_return(sid: str, i0: int, start: int, end: int) -> float:
        growth = 1.0
        for o in range(start, end + 1):
            growth *= 1.0 + _finite(daily_returns[(sid, days[i0 + o])])
        return growth - 1.0

    curve: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    for start, end in windows:
        per_cohort: list[float] = []
        per_cohort_ic: list[float] = []
        names_total = 0
        first_date: Optional[dt.date] = None
        last_date: Optional[dt.date] = None
        for cohort, meta, names in prepared:
            i0 = meta["i0"]
            signed = []
            signals = []
            realized = []
            for sid in names:
                r = window_return(sid, i0, start, end)
                direction = 1.0 if cohort.signals[sid] > 0 else -1.0
                signed.append(direction * r)
                signals.append(_finite(cohort.signals[sid]))
                realized.append(r)
            per_cohort.append(sum(signed) / len(signed))
            ic = _spearman(signals, realized)
            if ic is not None:
                per_cohort_ic.append(ic)
            names_total += len(names)
            d0, d1 = days[i0 + start], days[i0 + end]
            first_date = d0 if first_date is None or d0 < first_date else first_date
            last_date = d1 if last_date is None or d1 > last_date else last_date
        mean = sum(per_cohort) / len(per_cohort)
        sd = math.sqrt(sum((v - mean) ** 2 for v in per_cohort) / (len(per_cohort) - 1)) if len(per_cohort) > 1 else None
        curve.append({
            "start_offset": start, "end_offset": end,
            "first_date": first_date.isoformat() if first_date else None,
            "last_date": last_date.isoformat() if last_date else None,
            "mean_signed_payoff": mean,
            "se_across_cohorts": (sd / math.sqrt(len(per_cohort))) if sd is not None else None,
            "mean_rank_ic": (sum(per_cohort_ic) / len(per_cohort_ic)) if per_cohort_ic else None,
            "n_cohorts": len(per_cohort), "n_name_observations": names_total,
        })
    for _s, end in sorted({(1, e) for _a, e in windows}):
        per_cohort = []
        for cohort, meta, names in prepared:
            i0 = meta["i0"]
            vals = [(1.0 if cohort.signals[sid] > 0 else -1.0) * window_return(sid, i0, 1, end) for sid in names]
            per_cohort.append(sum(vals) / len(vals))
        cumulative.append({"through_offset": end, "mean_signed_payoff": sum(per_cohort) / len(per_cohort), "n_cohorts": len(per_cohort)})
    return {
        "status": "COMPUTED",
        "method": "alpha_decay_curve",
        "convention": "signed payoff = sign(signal) x compounded daily return over the window; offset 1 is the first earned day after entry",
        "prerequisites": prerequisites,
        "windows": curve,
        "cumulative": cumulative,
        "cohorts_used": [{"cohort": c.label, "entry_date": c.entry_date.isoformat(), "names": len(n), "dropped_incomplete": m["dropped"]}
                         for c, m, n in prepared],
        "excluded": excluded,
        "caveat": "Adjacent cohorts overlap in time, so the effective sample is below the number of cohorts; standard errors are descriptive.",
    }


def earnings_vs_ex_earnings(contributions: Sequence[PositionContribution],
                           earnings_dates: Mapping[str, Sequence[dt.date]],
                           trading_days: Sequence[dt.date], *, window=(-1,1)) -> dict[str, Any]:
    """Exact additive contribution split around an explicitly supplied event calendar.

    Each contribution's period is an ISO daily date. Explicit empty event lists
    mean no scheduled events; absent calendars fail closed. Overlapping event
    windows count a contribution once. This is descriptive, not causal.
    """
    days=list(trading_days)
    if any(b<=a for a,b in zip(days,days[1:])) or not days:raise ValueError('calendar must strictly increase')
    if window[0]>window[1]:raise ValueError('invalid event window')
    index={d:i for i,d in enumerate(days)};events={}
    for sid,dates in earnings_dates.items():
        if any(d not in index for d in dates):raise ValueError('earnings date is outside explicit trading calendar')
        events[sid]={offset for d in dates for offset in range(max(0,index[d]+window[0]),min(len(days),index[d]+window[1]+1))}
    near=[];away=[]
    for c in contributions:
        day=dt.date.fromisoformat(c.period)
        if day not in index:raise ValueError('contribution date is outside explicit trading calendar')
        if c.security_id not in events:raise ValueError(f'no earnings calendar supplied for {c.security_id}')
        (near if index[day] in events[c.security_id] else away).append(_finite(c.pnl))
    return {'method':'dated_earnings_vs_ex_earnings','window_trading_days':list(window),
        'earnings_contribution':sum(near),'ex_earnings_contribution':sum(away),'total':sum(near)+sum(away),
        'earnings_observations':len(near),'ex_earnings_observations':len(away),
        'convention':'sum of supplied daily position contributions; overlapping windows counted once; no causal claim'}
