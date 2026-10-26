"""Normalized, direction-aware strategy search inside four pod scopes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .desk_metrics import compute_desk_metrics
from .training_data import POD_FEATURES, TrainingPanel, market_return_series


@dataclass
class Candidate:
    """An exact executable weight rule and its complete development evidence."""

    strategy_id: str
    pod: str
    recipe: dict
    weights: np.ndarray
    net: np.ndarray
    turnover: np.ndarray
    metrics: dict
    score: float


@dataclass
class RiskDesign:
    """Formation controls and construction-only liquidity/short restrictions."""

    mask: np.ndarray
    matrix: np.ndarray
    inverse: np.ndarray
    shortable: np.ndarray
    liquidity_scale: np.ndarray
    short_cost_scale: np.ndarray

    def __iter__(self):
        return iter((self.mask, self.matrix, self.inverse))


def smooth_signal(signal, window):
    """Average only current and earlier formation ranks, preserving missingness."""
    result = np.full_like(signal, np.nan)
    for t in range(len(signal)):
        values = signal[max(0, t - window + 1) : t + 1]
        count = np.isfinite(values).sum(axis=0)
        result[t] = np.divide(
            np.nansum(values, axis=0),
            count,
            out=np.full(signal.shape[1], np.nan),
            where=count > 0,
        )
    return result


def risk_designs(panel, settings=None):
    """Precompute formation-time sector, dollar, beta and size projections."""
    result = []
    settings = settings or {}
    for frame, mask in zip(panel.frames, panel.mask):
        current = frame.loc[mask]
        sectors = current.gics.fillna("unknown").astype(str).str[:2]
        sector = pd.get_dummies(sectors, dtype=float).to_numpy()
        beta = current.beta_60m.fillna(1.0).to_numpy(dtype=float)
        size = np.log(current.me.to_numpy(dtype=float))
        controls = np.column_stack(
            [beta, size] if settings.get("neutralize_size", True) else [beta]
        )
        controls -= controls.mean(axis=0)
        controls /= np.maximum(controls.std(axis=0), 1e-8)
        design = np.column_stack([sector, controls])
        adv = (
            current.dolvol_126d.fillna(0).to_numpy(dtype=float)
            if "dolvol_126d" in current
            else np.full(len(current), 20_000_000)
        )
        large_fee = max(1, settings.get("large_cap_borrow_bps", 50))
        other_fee = max(1, settings.get("other_borrow_bps", 500))
        fee = np.where(
            current.me.to_numpy(dtype=float) * 1_000_000
            >= settings.get("large_cap_borrow_threshold_usd", 10_000_000_000),
            large_fee,
            other_fee,
        )
        result.append(
            RiskDesign(
                mask,
                design,
                np.linalg.pinv(design),
                current.me.to_numpy(dtype=float) >= 500,
                np.clip(adv / 20_000_000, 0, 1),
                np.minimum(1, large_fee / fee),
            )
        )
    return result


def make_weights(signal, designs, settings, *, construction_scaling=True):
    """Project normalized signals, then enforce gross and single-name limits."""
    weights = np.zeros_like(signal, dtype=np.float32)
    for t, (mask, design, inverse) in enumerate(designs):
        values = np.nan_to_num(signal[t, mask], nan=0.0)
        values = values - design @ (inverse @ values)
        if construction_scaling:
            values *= getattr(designs[t], "liquidity_scale", 1)
            short_scale = getattr(
                designs[t], "short_cost_scale", np.ones(len(values))
            )
            values[values < 0] *= short_scale[values < 0]
        values = project_target(
            values,
            design,
            settings,
            normalize=True,
            shortable=getattr(designs[t], "shortable", None),
        )
        weights[t, mask] = values
    return weights


def project_target(
    values, design, settings, *, normalize=False, shortable=None
):
    """Select at most 500 names, then neutralize inside that executable subset."""
    output = np.zeros(len(values), dtype=float)
    maximum = settings.get("universe_size", 500)
    if np.abs(values).sum() < 1e-10:
        return output
    values = values.copy()
    if shortable is not None:
        values[(values < 0) & ~shortable] = 0
    # Cost penalties must not fill the entire shortlist with one forecast side.
    # Preserve both tails before enforcing neutrality inside the selected book.
    longs = np.flatnonzero(values > 1e-12)
    shorts = np.flatnonzero(values < -1e-12)
    longs = longs[np.argsort(-values[longs], kind="stable")]
    shorts = shorts[np.argsort(values[shorts], kind="stable")]
    chosen = np.concatenate([longs[: maximum // 2], shorts[: maximum // 2]])
    if len(chosen) < min(maximum, len(values)):
        used = np.zeros(len(values), dtype=bool)
        used[chosen] = True
        remaining = np.flatnonzero(~used & (np.abs(values) > 1e-12))
        remaining = remaining[
            np.argsort(-np.abs(values[remaining]), kind="stable")
        ]
        chosen = np.concatenate([chosen, remaining[: maximum - len(chosen)]])
    for _ in range(maximum + 1):
        local = design[chosen]
        alpha = values[chosen].astype(float)
        alpha -= local @ (np.linalg.pinv(local) @ alpha)
        bad = (
            (alpha < -1e-12) & ~shortable[chosen]
            if shortable is not None
            else np.zeros(len(chosen), dtype=bool)
        )
        if not bad.any():
            break
        chosen = chosen[~bad]
        if len(chosen) < 2:
            return output
    gross = np.abs(alpha).sum()
    if gross < 1e-10:
        return output
    scale = min(
        settings["gross_limit"] / gross,
        settings["single_name_limit"] / max(np.abs(alpha).max(), 1e-10),
    )
    if not normalize:
        scale = min(1, scale)
    output[chosen] = alpha * scale
    return output


def measure_weights(weights, panel, settings):
    """Charge drift-aware turnover, forced exits and borrow for both directions."""
    count = len(panel.returns)
    net, turns = np.zeros(count), np.zeros(count)
    previous = np.zeros(weights.shape[1])
    cost_rate = settings["cost_bps_one_way"] / 10_000
    for t in range(count):
        target = weights[t].astype(float) * panel.entry_mask[t]
        forced = np.abs(target[panel.exit_mask[t]]).sum()
        turnover = np.abs(target - previous).sum() + forced
        # Exit all remaining shares at the development boundary.
        ending = target * (1 + panel.returns[t])
        ending[panel.exit_mask[t]] = 0
        if t == count - 1:
            turnover += np.abs(ending).sum()
        elapsed = (panel.settlement_dates[t] - panel.execution_dates[t]).days
        borrow = (
            np.abs(target[target < 0]).sum()
            * settings["borrow_bps_annual"]
            / 10_000
            * elapsed
            / 365.25
        )
        value = float(target @ panel.returns[t]) - cost_rate * turnover - borrow
        if not np.isfinite(value) or value <= -1:
            raise ValueError("Invalid or insolvent candidate return book")
        net[t], turns[t] = value, turnover
        previous = ending / (1 + value)
    return net, turns


def summarize(net, market=None, **kwargs):
    """Reuse the platform's PSR, DSR, drawdown, HAC and stability metrics."""
    metrics = compute_desk_metrics(net, periods_per_year=12, **kwargs)
    if market is not None:
        design = np.column_stack([np.ones(len(market)), market])
        intercept, beta = np.linalg.lstsq(design, net, rcond=None)[0]
        metrics["market_alpha_annualized"] = float(intercept * 12)
        metrics["realized_market_beta"] = float(beta)
    return metrics


def objective(metrics):
    """Reward return quality with explicit penalties for unstable halves."""
    sharpe = metrics["sharpe_annualized"]
    if sharpe is None:
        return -1e9
    halves = metrics["halves"]
    first = halves["first_sharpe_annualized"] or 0
    second = halves["second_sharpe_annualized"] or 0
    return (
        sharpe - 0.15 * abs(first - second) - 0.5 * metrics["maximum_drawdown"]
    )


def candidate(strategy_id, pod, recipe, weights, panel, settings):
    net, turnover = measure_weights(weights, panel, settings)
    metrics = summarize(net, market_return_series(panel))
    metrics["mean_monthly_turnover"] = float(turnover.mean())
    return Candidate(
        strategy_id,
        pod,
        recipe,
        weights,
        net,
        turnover,
        metrics,
        objective(metrics),
    )


def search(panel: TrainingPanel, settings: dict, progress=None):
    """Record every direction, smoothing choice and ensemble as a trial."""
    designs = risk_designs(panel, settings)
    candidates = []
    for pod, features in POD_FEATURES.items():
        for feature in features:
            raw = panel.features[feature]
            coverage = np.isfinite(raw[panel.mask]).mean()
            if coverage < settings["min_feature_coverage"]:
                continue
            for window in settings["smoothing_months"]:
                weights = make_weights(
                    smooth_signal(raw, window), designs, settings
                )
                for sign in (1, -1):
                    name = f"{pod}:{feature}:smooth{window}:sign{sign:+d}"
                    candidates.append(
                        candidate(
                            name,
                            pod,
                            {
                                "feature": feature,
                                "smoothing_months": window,
                                "expected_return_sign": sign,
                                "scaling": "cross_sectional_rank",
                                "risk_controls": ["sector", "beta_60m"]
                                + (
                                    ["log_me"]
                                    if settings.get("neutralize_size", True)
                                    else []
                                ),
                            },
                            weights * sign,
                            panel,
                            settings,
                        )
                    )
        singles = sorted(
            [c for c in candidates if c.pod == pod],
            key=lambda c: c.score,
            reverse=True,
        )
        # Avoid disguising smoothing variants of one feature as diversification.
        unique = []
        seen = set()
        for trial in singles:
            if trial.recipe["feature"] not in seen:
                unique.append(trial)
                seen.add(trial.recipe["feature"])
        for size in settings["ensemble_sizes"]:
            components = unique[:size]
            if len(components) < size:
                continue
            combined = np.mean([c.weights for c in components], axis=0)
            combined = make_weights(
                combined, designs, settings, construction_scaling=False
            )
            gross = np.abs(combined).sum(axis=1, keepdims=True)
            combined *= np.minimum(
                np.divide(
                    settings["gross_limit"],
                    gross,
                    out=np.zeros_like(gross),
                    where=gross > 1e-10,
                ),
                np.divide(
                    settings["single_name_limit"],
                    np.abs(combined).max(axis=1, keepdims=True),
                    out=np.zeros_like(gross),
                    where=gross > 1e-10,
                ),
            )
            candidates.append(
                candidate(
                    f"{pod}:ensemble_top{size}",
                    pod,
                    {
                        "components": [c.strategy_id for c in components],
                        "combination": "equal_signal_weight_then_gross_normalize",
                    },
                    combined,
                    panel,
                    settings,
                )
            )
        if progress:
            progress(pod, len([c for c in candidates if c.pod == pod]))
    if not candidates:
        raise ValueError(
            "No research candidates have sufficient input coverage"
        )
    sharpes = [c.metrics["sharpe_periodic"] for c in candidates]
    complete = all(value is not None for value in sharpes)
    for trial in candidates:
        trial.metrics.update(
            summarize(
                trial.net,
                market_return_series(panel),
                trial_sharpes=[v for v in sharpes if v is not None],
                total_attempts=len(candidates),
                trial_periods_per_year=12,
                attempts_complete=complete,
            )
        )
    return candidates


def select_strategies(candidates, settings):
    """Keep one to three complementary positive-return strategies per pod."""
    selected = {}
    for pod in POD_FEATURES:
        ranked = sorted(
            [
                c
                for c in candidates
                if c.pod == pod
                and float(np.mean(c.net)) > 0
                and c.metrics["cagr"] > 0
            ],
            key=lambda c: c.score,
            reverse=True,
        )
        chosen = []
        for trial in ranked:
            if not chosen:
                chosen.append(trial)
                if settings["max_strategies_per_pod"] == 1:
                    break
                continue
            correlations = [
                abs(float(np.corrcoef(trial.net, other.net)[0, 1]))
                for other in chosen
            ]
            if max(correlations) > settings["max_strategy_correlation"]:
                continue
            old = summarize(np.mean([c.net for c in chosen], axis=0))
            new = summarize(np.mean([c.net for c in chosen + [trial]], axis=0))
            if (
                objective(new)
                > objective(old) + settings["min_incremental_score"]
            ):
                chosen.append(trial)
            if len(chosen) >= settings["max_strategies_per_pod"]:
                break
        selected[pod] = chosen
    return selected
