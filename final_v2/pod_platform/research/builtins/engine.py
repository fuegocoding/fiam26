"""Walk-forward evaluation engine for reviewed recipes (pure Python).

For each design-window month ``m`` (information date = its month end):

* features use month ``m`` and earlier rows only;
* a ridge model is refit using only months ``s < m`` whose label had RESOLVED by the information date of ``m``
  (label maturity = purge; the target horizon is the embargo) and inside the design window;
* the belief prediction is scored by cross-sectional rank IC against the recipe's target, paired against the
  declared cheap baseline on identical rows;
* a separate return bridge scores the same predictions against the dataset's next-month excess-return label:
  rank IC, equal-weight tail spread (short leg restricted to market-cap-eligible names), flat one-way cost on
  turnover, plus sector / exposure diagnostics.

Nothing is tuned on the evaluated months: the recipe, method and penalty are fixed before the data is read.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional

from . import stats
from .catalog import LABEL, FeatureSpec, Recipe, VariantConfig
from .errors import BuiltinConfigError, EvidenceInsufficient
from .loader import Panel
from .manifest import ApprovedDataset
from .timeutil import end_of_day, month_end_known_at

ENGINE_VERSION = "builtins.engine.v1"
EVALUATION_LABEL = "retrospective_design_partition_not_fresh_holdout"
MIN_SECTOR_GROUP = 5
MIN_EXPOSURE_ROWS = 30


@dataclass
class MonthData:
    month: int
    resolve_at: dt.datetime
    usable: bool
    target: dict[int, float] = field(default_factory=dict)
    feats: dict[str, dict[int, float]] = field(default_factory=dict)
    baseline: dict[int, float] = field(default_factory=dict)


def physical(column: str, aliases: dict[str, str]) -> str:
    return aliases.get(column, column)


def required_columns(recipe: Recipe, dataset: ApprovedDataset) -> set[str]:
    """Physical columns to load: recipe inputs (after operator aliases) + label + declared context/exposure columns."""
    spec = dataset.spec
    aliases = spec.column_aliases
    cols = {physical(c, aliases) for c in recipe.logical_columns(spec.label.column) if c != spec.label.column}
    cols.add(spec.label.column)
    if recipe.uses_sector():
        if spec.sector_column is None:
            raise BuiltinConfigError(f"recipe {recipe.recipe_id} needs a sector_column but the dataset declares none")
        cols.add(spec.sector_column)
    elif spec.sector_column:
        cols.add(spec.sector_column)  # sector diagnostics
    if spec.market_cap_column:
        cols.add(spec.market_cap_column)
    cols.update(spec.exposure_columns)
    for column in cols:
        declared = spec.columns.get(column)
        if declared is None:
            raise BuiltinConfigError(f"recipe {recipe.recipe_id} needs column {column!r} which the manifest does not declare (no silent substitution)")
        if column != spec.label.column and declared.role == "label":
            raise BuiltinConfigError(f"column {column!r} has role=label and cannot be used as a feature")
    return cols


def _sector_key(panel: Panel, row: int, sector_column: Optional[str], digits: int) -> Optional[str]:
    if sector_column is None:
        return None
    value = panel.text[sector_column][row]
    return value[:digits] if value else None


def _centered(raw: dict[int, float]) -> dict[int, float]:
    rows = sorted(raw)
    ranks = stats.centered_rank([raw[r] for r in rows])
    return dict(zip(rows, ranks))


def feature_values(panel: Panel, spec: FeatureSpec, month: int, dataset: ApprovedDataset) -> dict[int, float]:
    ds = dataset.spec
    aliases = ds.column_aliases
    rows = panel.rows_by_month.get(month, [])
    x = panel.num[physical(spec.column, aliases)]
    raw: dict[int, float] = {}
    if spec.op in ("rank", "sector_rank"):
        for r in rows:
            if x[r] is not None:
                raw[r] = x[r]
    elif spec.op == "rank_product":
        y = panel.num[physical(spec.column2 or "", aliases)]
        both = {r: (x[r], y[r]) for r in rows if x[r] is not None and y[r] is not None}
        first = _centered({r: v[0] for r, v in both.items()})
        second = _centered({r: v[1] for r, v in both.items()})
        raw = {r: first[r] * second[r] for r in both}
    elif spec.op in ("lag_rank", "change_rank", "growth_rank"):
        if spec.months < 1:
            raise BuiltinConfigError(f"feature {spec.name}: {spec.op} needs months >= 1")
        for r in rows:
            j = panel.row_index.get((panel.security[r], month - spec.months))
            if j is None or x[j] is None:
                continue
            past = x[j]
            if spec.op == "lag_rank":
                raw[r] = past
            elif x[r] is None:
                continue
            elif spec.op == "change_rank":
                raw[r] = x[r] - past
            elif past > 0.0:  # growth only where the denominator is positive
                raw[r] = x[r] / past - 1.0
    else:
        raise BuiltinConfigError(f"unknown feature op {spec.op!r}")
    if spec.op == "sector_rank":
        groups: dict[str, dict[int, float]] = {}
        for r, v in raw.items():
            key = _sector_key(panel, r, ds.sector_column, ds.evaluation.sector_digits)
            if key is not None:
                groups.setdefault(key, {})[r] = v
        out: dict[int, float] = {}
        for members in groups.values():
            if len(members) >= MIN_SECTOR_GROUP:
                out.update(_centered(members))
        raw_out = out
    elif spec.op == "rank_product":
        raw_out = _centered(raw)
    else:
        raw_out = _centered(raw)
    sign = float(spec.sign)
    return {r: sign * v for r, v in raw_out.items()}


def target_values(panel: Panel, recipe: Recipe, month: int, dataset: ApprovedDataset) -> dict[int, float]:
    ds = dataset.spec
    aliases = ds.column_aliases
    t = recipe.target
    rows = panel.rows_by_month.get(month, [])
    raw: dict[int, float] = {}
    if t.kind == "lead":
        x = panel.num[physical(t.column, aliases)]
        for r in rows:
            j = panel.row_index.get((panel.security[r], month + t.months))
            if j is not None and x[j] is not None:
                raw[r] = x[j]
    else:  # cum_lead over the label column: rows m .. m+h-1 (label of row j is the return after month j)
        x = panel.num[ds.label.column if t.column == LABEL else physical(t.column, aliases)]
        for r in rows:
            total = 0.0
            ok = True
            for step in range(t.months):
                j = panel.row_index.get((panel.security[r], month + step))
                if j is None or x[j] is None:
                    ok = False
                    break
                total += x[j]
            if ok:
                raw[r] = total
    if t.demean == "sector":
        groups: dict[str, list[int]] = {}
        for r in raw:
            key = _sector_key(panel, r, ds.sector_column, ds.evaluation.sector_digits)
            if key is not None:
                groups.setdefault(key, []).append(r)
        demeaned: dict[int, float] = {}
        for members in groups.values():
            if len(members) >= MIN_SECTOR_GROUP:
                mu = math.fsum(raw[r] for r in members) / len(members)
                for r in members:
                    demeaned[r] = raw[r] - mu
        raw = demeaned
    return _centered(raw)


def target_resolve_at(recipe: Recipe, month: int, dataset: ApprovedDataset) -> dt.datetime:
    lag = dataset.spec.label.resolve_lag_days
    t = recipe.target
    if t.kind == "lead":
        return month_end_known_at(month + t.months, lag)
    return month_end_known_at(month + t.months - 1 + dataset.spec.label.resolve_months, lag)


def _fit_ridge(train: list[tuple[int, list[float], float, list[list[float]], list[float]]], lam: float, k: int):
    n = sum(t[0] for t in train)
    if n < 10:
        return None
    sx = [math.fsum(t[1][j] for t in train) for j in range(k)]
    sy = math.fsum(t[2] for t in train)
    xbar = [v / n for v in sx]
    ybar = sy / n
    cov = [
        [math.fsum(t[3][i][j] for t in train) / n - xbar[i] * xbar[j] + (lam if i == j else 0.0) for j in range(k)]
        for i in range(k)
    ]
    cxy = [math.fsum(t[4][i] for t in train) / n - xbar[i] * ybar for i in range(k)]
    beta = stats.solve_linear(cov, cxy)
    if beta is None:
        return None
    return xbar, ybar, beta


def _month_stats(rows: list[int], names: list[str], md: MonthData):
    k = len(names)
    xs = [[md.feats[name][r] for name in names] for r in rows]
    ys = [md.target[r] for r in rows]
    n = len(rows)
    sx = [math.fsum(x[j] for x in xs) for j in range(k)]
    sy = math.fsum(ys)
    sxx = [[math.fsum(x[i] * x[j] for x in xs) for j in range(k)] for i in range(k)]
    sxy = [math.fsum(x[i] * y for x, y in zip(xs, ys)) for i in range(k)]
    return (n, sx, sy, sxx, sxy)


def _tail_weights(order: list[int], q: int) -> dict[int, float]:
    return {r: 1.0 / q for r in order[:q]}


def evaluate(panel: Panel, dataset: ApprovedDataset, recipe: Recipe, variant: VariantConfig, *, research_asof: dt.datetime) -> dict:
    """Run the walk-forward evaluation and return metrics plus full diagnostics. Raises EvidenceInsufficient on thin data."""
    ds = dataset.spec
    ev = ds.evaluation
    features = list(recipe.features(variant.feature_set))
    names = [f.name for f in features]
    k = len(features)
    window_start, window_end = dataset.design_window
    window_end_dt = end_of_day(window_end)
    if window_end_dt > research_asof:
        raise EvidenceInsufficient("LABEL_NOT_MATURE: design window has not ended at the research clock")
    horizon = recipe.target.months
    label_col = ds.label.column
    ret = panel.num[label_col]
    months = sorted(panel.rows_by_month)

    data: dict[int, MonthData] = {}
    purged_rows = 0
    for m in months:
        resolve = target_resolve_at(recipe, m, dataset)
        usable = resolve <= window_end_dt and resolve <= research_asof
        md = MonthData(month=m, resolve_at=resolve, usable=usable)
        md.feats = {f.name: feature_values(panel, f, m, dataset) for f in features}
        md.baseline = feature_values(panel, recipe.baseline, m, dataset)
        if usable:
            md.target = target_values(panel, recipe, m, dataset)
        else:
            purged_rows += len(panel.rows_by_month[m])
        data[m] = md

    def complete_rows(md: MonthData, need_baseline: bool) -> list[int]:
        rows = [r for r in md.target if all(r in md.feats[n] for n in names)]
        if need_baseline:
            rows = [r for r in rows if r in md.baseline]
        return sorted(rows)

    train_stats = {}
    for m in months:
        md = data[m]
        if md.usable:
            rows = complete_rows(md, False)
            if len(rows) >= ev.min_names_per_month:
                train_stats[m] = _month_stats(rows, names, md)

    cap_col = ds.market_cap_column
    cap = panel.num[cap_col] if cap_col else None
    sector_col = ds.sector_column
    digits = ev.sector_digits
    cost = ev.cost_bps_one_way / 1e4
    tail = ev.tail_fraction
    belief_lag = max(0, horizon - 1)

    monthly: list[dict] = []
    predictions: list[dict] = []  # exact out-of-sample design rows; persisted as immutable candidate provenance
    prev_w = {"cand": {}, "base": {}}
    fits = 0
    warmup_months = 0
    degenerate = 0
    short_ineligible_months = 0
    exposures_acc: dict[str, dict[str, list[float]]] = {c: {"cand": [], "base": []} for c in ds.exposure_columns}
    sector_cand = {"long_max": [], "short_max": [], "net_max": [], "sn_ic_c": [], "sn_ic_b": []}
    missing_by_feature = {n: 0 for n in names}
    total_target_rows = 0
    total_eval_rows = 0

    for m in months:
        md = data[m]
        formation = month_end_known_at(m)
        train_months = [s for s in months if s < m and s in train_stats and data[s].resolve_at <= formation]
        if len(train_months) < ev.min_train_months:
            warmup_months += 1
            continue
        if not md.usable:
            continue  # label unresolved inside the design window: purged
        if month_end_known_at(m + ds.label.resolve_months, ds.label.resolve_lag_days) > window_end_dt:
            purged_rows += len(panel.rows_by_month[m])  # the bridge return has not resolved inside the window
            continue
        rows_target = sorted(md.target)
        total_target_rows += len(rows_target)
        for n in names:
            missing_by_feature[n] += sum(1 for r in rows_target if r not in md.feats[n])
        eval_rows = [r for r in complete_rows(md, True) if ret[r] is not None]
        if len(eval_rows) < ev.min_names_per_month:
            continue
        if variant.method_id == "ridge":
            fit = _fit_ridge([train_stats[s] for s in train_months], float(variant.ridge_lambda), k)
            if fit is None:
                continue
            fits += 1
            xbar, ybar, beta = fit
            pred = {r: ybar + math.fsum(b * (md.feats[n][r] - xb) for b, n, xb in zip(beta, names, xbar)) for r in eval_rows}
        else:
            pred = {r: math.fsum(md.feats[n][r] for n in names) / k for r in eval_rows}
        base = {r: md.baseline[r] for r in eval_rows}
        tgt = [md.target[r] for r in eval_rows]
        pv = [pred[r] for r in eval_rows]
        bv = [base[r] for r in eval_rows]
        rv = [ret[r] for r in eval_rows]
        total_eval_rows += len(eval_rows)
        ic_c, ic_b = stats.spearman(pv, tgt), stats.spearman(bv, tgt)
        if ic_c is None or ic_b is None:
            degenerate += 1
            continue
        sign = float(recipe.bridge_sign)
        bic_c = stats.spearman([sign * v for v in pv], rv)
        bic_b = stats.spearman([sign * v for v in bv], rv)
        if bic_c is None or bic_b is None:
            degenerate += 1
            continue
        rec: dict = {
            "month": m, "n": len(eval_rows), "ic_cand": ic_c, "ic_base": ic_b,
            "bridge_ic_cand": bic_c, "bridge_ic_base": bic_b,
        }
        # --- tail portfolios (long top q by signed score; short bottom q among cap-eligible names)
        n_rows = len(eval_rows)
        q = max(1, int(n_rows * tail))
        eligible = set(eval_rows)
        short_filter = False
        if cap is not None:
            floor = ev.short_min_market_cap_usd / ds.market_cap_to_usd
            eligible = {r for r in eval_rows if cap[r] is not None and cap[r] >= floor}
            short_filter = True
        rec["short_eligible_share"] = len(eligible) / n_rows
        port_ok = len(eligible) >= 1
        if port_ok:
            qs = min(q, len(eligible))
            for label, score in (("cand", pred), ("base", base)):
                signed = {r: sign * score[r] for r in eval_rows}
                desc = sorted(eval_rows, key=lambda r: (-signed[r], panel.security[r]))
                asc = sorted((r for r in eval_rows if r in eligible), key=lambda r: (signed[r], panel.security[r]))
                longs, shorts = desc[:q], asc[:qs]
                gross = math.fsum(ret[r] for r in longs) / len(longs) - math.fsum(ret[r] for r in shorts) / len(shorts)
                weights = {panel.security[r]: 1.0 / len(longs) for r in longs}
                for r in shorts:
                    weights[panel.security[r]] = weights.get(panel.security[r], 0.0) - 1.0 / len(shorts)
                keys = set(weights) | set(prev_w[label])
                turn = math.fsum(abs(weights.get(s, 0.0) - prev_w[label].get(s, 0.0)) for s in keys)
                prev_w[label] = weights
                rec[f"gross_{label}"] = gross
                rec[f"turnover_{label}"] = turn
                rec[f"net_{label}"] = gross - cost * turn
                if label == "cand":
                    rec["n_long"], rec["n_short"] = len(longs), len(shorts)
                    if sector_col:
                        sl, ss = {}, {}
                        for r in longs:
                            key = _sector_key(panel, r, sector_col, digits) or "_none"
                            sl[key] = sl.get(key, 0) + 1
                        for r in shorts:
                            key = _sector_key(panel, r, sector_col, digits) or "_none"
                            ss[key] = ss.get(key, 0) + 1
                        sector_cand["long_max"].append(max(sl.values()) / len(longs))
                        sector_cand["short_max"].append(max(ss.values()) / len(shorts))
                        union = set(sl) | set(ss)
                        sector_cand["net_max"].append(max(abs(sl.get(s, 0) / len(longs) - ss.get(s, 0) / len(shorts)) for s in union))
        else:
            short_ineligible_months += 1
        # --- sector-neutral belief IC (within-sector demeaned prediction and target)
        if sector_col:
            groups: dict[str, list[int]] = {}
            for r in eval_rows:
                key = _sector_key(panel, r, sector_col, digits)
                if key is not None:
                    groups.setdefault(key, []).append(r)
            xs_c, xs_b, ys = [], [], []
            for members in groups.values():
                if len(members) >= MIN_SECTOR_GROUP:
                    mc = math.fsum(pred[r] for r in members) / len(members)
                    mb = math.fsum(base[r] for r in members) / len(members)
                    my = math.fsum(md.target[r] for r in members) / len(members)
                    for r in members:
                        xs_c.append(pred[r] - mc)
                        xs_b.append(base[r] - mb)
                        ys.append(md.target[r] - my)
            snc, snb = stats.spearman(xs_c, ys), stats.spearman(xs_b, ys)
            if snc is not None and snb is not None:
                sector_cand["sn_ic_c"].append(snc)
                sector_cand["sn_ic_b"].append(snb)
        # --- exposure correlations (score vs characteristic, cross-sectional rank correlation)
        for column in ds.exposure_columns:
            col = panel.num[column]
            idx = [r for r in eval_rows if col[r] is not None]
            if len(idx) >= MIN_EXPOSURE_ROWS:
                ev_col = [col[r] for r in idx]
                for label, score in (("cand", pred), ("base", base)):
                    corr = stats.spearman([score[r] for r in idx], ev_col)
                    if corr is not None:
                        exposures_acc[column][label].append(corr)
        monthly.append(rec)
        resolved = max(md.resolve_at, month_end_known_at(m + ds.label.resolve_months, ds.label.resolve_lag_days))
        for r in eval_rows:
            predictions.append({"m": m, "s": panel.security[r], "p": pred[r], "b": base[r], "t": md.target[r], "r": ret[r],
                                "x": resolved.isoformat()})

    if len(monthly) < ev.min_eval_months:
        raise EvidenceInsufficient(
            f"INSUFFICIENT_HISTORY: {len(monthly)} evaluable design months < required {ev.min_eval_months} "
            f"(warm-up months {warmup_months}, degenerate {degenerate}, purged label rows {purged_rows})"
        )
    n_m = len(monthly)
    metrics: dict[str, float] = {}

    def paired(key_a: str, key_b: str, lag: int, prefix: str, rows: list[dict]) -> Optional[list[float]]:
        diffs = [r[key_a] - r[key_b] for r in rows if key_a in r and key_b in r]
        if len(diffs) < 3:
            return None
        mu = stats.mean(diffs)
        se = stats.hac_se(diffs, lag)
        metrics[f"{prefix}_improvement"] = mu
        if se is not None and se > 0:
            metrics[f"{prefix}_improvement_se"] = se
            metrics[f"{prefix}_improvement_t"] = mu / se
            metrics[f"{prefix}_improvement_ci95_low"] = mu - 1.96 * se
            metrics[f"{prefix}_improvement_ci95_high"] = mu + 1.96 * se
        metrics[f"{prefix}_improvement_hit_rate"] = sum(1 for d in diffs if d > 0) / len(diffs)
        return diffs

    metrics["belief_months"] = float(n_m)
    metrics["belief_ic_candidate"] = stats.mean([r["ic_cand"] for r in monthly])
    metrics["belief_ic_baseline"] = stats.mean([r["ic_base"] for r in monthly])
    belief_diffs = paired("ic_cand", "ic_base", belief_lag, "belief_ic", monthly)
    half = n_m // 2
    if belief_diffs and half >= 2:
        metrics["belief_ic_improvement_first_half"] = stats.mean(belief_diffs[:half])
        metrics["belief_ic_improvement_second_half"] = stats.mean(belief_diffs[half:])
    by_year: dict[int, list[float]] = {}
    for rec_, diff in zip(monthly, belief_diffs or []):
        by_year.setdefault(rec_["month"] // 12, []).append(diff)
    year_means = {y: stats.mean(v) for y, v in by_year.items() if len(v) >= 3}
    if year_means:
        metrics["belief_years_counted"] = float(len(year_means))
        metrics["belief_positive_year_share"] = sum(1 for v in year_means.values() if v > 0) / len(year_means)
        metrics["belief_max_year_abs_share"] = (
            max(abs(v) for v in year_means.values()) / sum(abs(v) for v in year_means.values())
            if any(v != 0 for v in year_means.values()) else 0.0
        )
    metrics["bridge_ic_candidate"] = stats.mean([r["bridge_ic_cand"] for r in monthly])
    metrics["bridge_ic_baseline"] = stats.mean([r["bridge_ic_base"] for r in monthly])
    paired("bridge_ic_cand", "bridge_ic_base", 0, "bridge_ic", monthly)
    port = [r for r in monthly if "net_cand" in r]
    metrics["bridge_portfolio_months"] = float(len(port))
    if port:
        metrics["bridge_gross_spread_candidate"] = stats.mean([r["gross_cand"] for r in port])
        metrics["bridge_gross_spread_baseline"] = stats.mean([r["gross_base"] for r in port])
        metrics["bridge_net_spread_candidate"] = stats.mean([r["net_cand"] for r in port])
        metrics["bridge_net_spread_baseline"] = stats.mean([r["net_base"] for r in port])
        metrics["bridge_turnover_candidate"] = stats.mean([r["turnover_cand"] for r in port])
        metrics["bridge_turnover_baseline"] = stats.mean([r["turnover_base"] for r in port])
        metrics["bridge_cost_bps_one_way_assumed"] = ev.cost_bps_one_way
        metrics["bridge_names_long_mean"] = stats.mean([float(r["n_long"]) for r in port])
        metrics["bridge_names_short_mean"] = stats.mean([float(r["n_short"]) for r in port])
        metrics["bridge_short_eligible_share"] = stats.mean([r["short_eligible_share"] for r in port])
        net_diffs = paired("net_cand", "net_base", 0, "bridge_net_spread", port)
        if net_diffs:
            metrics["bridge_net_spread_candidate_positive_share"] = sum(1 for r in port if r["net_cand"] > 0) / len(port)
    metrics["coverage_eval_months"] = float(n_m)
    metrics["coverage_eval_rows"] = float(total_eval_rows)
    metrics["coverage_names_per_month_mean"] = stats.mean([float(r["n"]) for r in monthly])
    metrics["coverage_names_per_month_min"] = float(min(r["n"] for r in monthly))
    metrics["coverage_warmup_months"] = float(warmup_months)
    metrics["coverage_purged_label_rows"] = float(purged_rows)
    metrics["coverage_degenerate_months"] = float(degenerate)
    if total_target_rows:
        metrics["coverage_feature_complete_share"] = total_eval_rows / total_target_rows
    metrics["trial_fold_evaluations"] = float(fits)
    if sector_col and sector_cand["long_max"]:
        metrics["sector_long_max_share"] = stats.mean(sector_cand["long_max"])
        metrics["sector_short_max_share"] = stats.mean(sector_cand["short_max"])
        metrics["sector_net_exposure_max_abs"] = stats.mean(sector_cand["net_max"])
    if sector_cand["sn_ic_c"]:
        metrics["sector_neutral_belief_ic_candidate"] = stats.mean(sector_cand["sn_ic_c"])
        metrics["sector_neutral_belief_ic_baseline"] = stats.mean(sector_cand["sn_ic_b"])
    for column, acc in exposures_acc.items():
        if acc["cand"]:
            metrics[f"exposure_corr_candidate_{column}"] = stats.mean(acc["cand"])
        if acc["base"]:
            metrics[f"exposure_corr_baseline_{column}"] = stats.mean(acc["base"])
    for key, value in list(metrics.items()):
        if value is None or not math.isfinite(value):
            raise EvidenceInsufficient(f"metric {key} is undefined or non-finite; refusing to report it")

    return {
        "metrics": {k_: float(v) for k_, v in sorted(metrics.items())},
        "predictions": predictions,
        "frozen_model": freeze_model(variant, names, [train_stats[s] for s in sorted(train_stats)], k, sorted(train_stats)),
        "monthly": monthly,
        "diagnostics": {
            "feature_names": names,
            "missing_target_rows_by_feature": missing_by_feature,
            "fits": fits,
            "belief_dependence_lag_months": belief_lag,
            "yearly_belief_improvement": {str(y): v for y, v in sorted(year_means.items())},
            "short_ineligible_months": short_ineligible_months,
            "short_eligibility_enforced": cap is not None,
        },
    }


def freeze_model(variant: VariantConfig, names: list[str], stats_list: list, k: int, train_months: list[int]) -> dict:
    """Final design-trained model: ridge fit once on every design month whose target resolved inside the window.

    A rank blend has no fitted parameters. The result is plain JSON so it can be hashed and replayed without refitting.
    """
    model: dict = {"method_id": variant.method_id, "feature_names": list(names)}
    if variant.method_id == "ridge":
        fit = _fit_ridge(stats_list, float(variant.ridge_lambda), k)
        if fit is None:
            raise EvidenceInsufficient("INSUFFICIENT_HISTORY: the final design-trained ridge model is underdetermined")
        model.update({"ridge_lambda": float(variant.ridge_lambda), "xbar": list(fit[0]), "ybar": fit[1], "beta": list(fit[2]),
                      "train_months": list(train_months)})
    return model


def score_frozen(
    panel: Panel, dataset: ApprovedDataset, recipe: Recipe, variant: VariantConfig, model: dict, months: list[int],
    *, resolved_by: Optional[dt.datetime] = None,
) -> list[dict]:
    """Apply a frozen model to ``months`` WITHOUT refitting. ``resolved_by`` set => labelled rows only (target and bridge resolved by it).

    ``resolved_by=None`` produces unlabeled forecast rows (features and baseline only).
    """
    ds = dataset.spec
    features = list(recipe.features(variant.feature_set))
    names = [f.name for f in features]
    if model.get("feature_names") != names or model.get("method_id") != variant.method_id:
        raise BuiltinConfigError("frozen model does not match the recipe features / method; refusing to score")
    ret = panel.num.get(ds.label.column, [None] * len(panel))
    out: list[dict] = []
    for m in months:
        if not panel.rows_by_month.get(m):
            continue
        resolved = None
        target: dict[int, float] = {}
        if resolved_by is not None:
            resolved = max(target_resolve_at(recipe, m, dataset), month_end_known_at(m + ds.label.resolve_months, ds.label.resolve_lag_days))
            if resolved > resolved_by:
                continue  # label not mature at the evaluation window / clock
            target = target_values(panel, recipe, m, dataset)
        feats = {f.name: feature_values(panel, f, m, dataset) for f in features}
        base = feature_values(panel, recipe.baseline, m, dataset)
        for r in sorted(panel.rows_by_month[m]):
            if any(r not in feats[n] for n in names) or r not in base:
                continue
            if resolved_by is not None and (r not in target or ret[r] is None):
                continue
            if model["method_id"] == "ridge":
                pred = model["ybar"] + math.fsum(b * (feats[n][r] - xb) for b, n, xb in zip(model["beta"], names, model["xbar"]))
            else:
                pred = math.fsum(feats[n][r] for n in names) / len(names)
            out.append({"m": m, "s": panel.security[r], "p": pred, "b": base[r], "t": target.get(r), "r": ret[r] if resolved else None,
                        "x": resolved.isoformat() if resolved else None})
    return out
