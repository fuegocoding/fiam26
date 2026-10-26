"""One forecast-to-portfolio mapping shared by diagnostics and execution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pod_plans import construction_designs


def prepare_forecast(score, design):
    active = design.mask & np.isfinite(score) & (np.abs(score) > 1e-12)
    indices = np.flatnonzero(active)
    matrix = design.matrix[active[design.mask]]
    alpha = np.asarray(score[active], float)
    if len(alpha):
        alpha -= matrix @ (np.linalg.pinv(matrix) @ alpha)
    return indices, matrix, alpha, design.shortable[active[design.mask]]


def supported_target(
    score,
    design,
    settings,
    *,
    maximum,
    weighting="score",
    prepared=None,
    sizing_factors=None,
):
    """One supported, sign-preserving constructor for diagnostics and trading."""
    output = np.zeros(len(score), dtype=np.float32)
    indices, matrix, alpha, shortable = (
        prepared if prepared is not None else prepare_forecast(score, design)
    )
    if len(indices) < 2:
        return output
    longs = np.flatnonzero(alpha > 1e-12)
    shorts = np.flatnonzero((alpha < -1e-12) & shortable)
    longs = longs[np.argsort(-alpha[longs], kind="stable")][: maximum // 2]
    shorts = shorts[np.argsort(alpha[shorts], kind="stable")][: maximum // 2]
    if not len(longs) or not len(shorts):
        return output
    chosen = np.r_[longs, shorts]
    desired = (
        np.abs(alpha[chosen]) if weighting == "score" else np.ones(len(chosen))
    )
    if sizing_factors is not None:
        desired *= sizing_factors[chosen]
    if (
        desired[: len(longs)].sum() <= 1e-12
        or desired[len(longs) :].sum() <= 1e-12
    ):
        return output
    desired[: len(longs)] /= desired[: len(longs)].sum()
    desired[len(longs) :] /= -desired[len(longs) :].sum()
    signs = np.sign(desired)
    free = np.ones(len(chosen), bool)
    final = np.zeros(len(chosen))
    controls = matrix[chosen]
    for _ in range(len(chosen) + 1):
        if free.sum() < 2:
            return output
        local = controls[free]
        value = desired[free] - local @ (np.linalg.pinv(local) @ desired[free])
        wrong = value * signs[free] < -1e-12
        if not wrong.any():
            final[free] = value
            break
        remaining = np.flatnonzero(free)
        free[remaining[wrong]] = False
    gross = np.abs(final).sum()
    if gross <= 1e-10:
        return output
    scale = min(
        settings["gross_limit"] / gross,
        settings["single_name_limit"] / max(np.abs(final).max(), 1e-12),
    )
    output[indices[chosen]] = final * scale
    return output


def monthly_concentration(strategy, decisions, panel, market, settings):
    """All deployable counts, same constructor and native monthly retention.

    Labels cover execution close to the next execution close, including the
    old book's final return before its replacement fills. Calendar-month
    headline returns have a different boundary. The label is the exact
    fixed-share zero-size interval. Faster rules
    must use daily replay and cannot select a count from this monthly curve.
    """
    if strategy.review != "monthly":
        return None
    designs = construction_designs(panel, strategy.controls)
    by_month = {formation: score for _, (formation, score) in decisions.items()}
    prepared = {
        t: prepare_forecast(score, designs[t]) for t, score in by_month.items()
    }
    rows = []
    for maximum in range(2, settings["universe_size"] + 1, 2):
        carried = np.zeros(len(panel.securities))
        for t, date in enumerate(panel.dates[:-1]):
            if t in by_month:
                carried = construct_target(
                    strategy,
                    by_month[t],
                    designs[t],
                    panel.frames[t],
                    settings,
                    maximum,
                    prepared=prepared[t],
                ).astype(float)
                carried[~panel.entry_mask[t]] = 0
            gross = float(carried @ panel.returns[t])
            if gross <= -1:
                raise ValueError("Exact monthly concentration became insolvent")
            rows.append(
                (
                    date,
                    panel.settlement_dates[t],
                    maximum,
                    gross,
                    int(np.count_nonzero(carried)),
                    float(np.abs(carried).sum()),
                )
            )
            carried *= 1 + panel.returns[t]
            carried[panel.exit_mask[t]] = 0
            carried /= 1 + gross
    result = pd.DataFrame(
        rows,
        columns=[
            "formation_date",
            "settlement_date",
            "names",
            "controlled_spread",
            "actual_positions",
            "gross_exposure",
        ],
    )
    result["gross_spread"] = result.controlled_spread
    result["return_per_unit_gross"] = np.divide(
        result.controlled_spread,
        result.gross_exposure,
        out=np.full(len(result), np.nan),
        where=result.gross_exposure > 1e-10,
    )
    return result


def liquid_hedged_target(score, design, frame, settings, *, maximum):
    """Rank the alpha long book, then hedge sectors using liquid large caps.

    Large-cap hedge membership is explicit construction, not a negative alpha
    prediction about those companies. Raw liquidity is never an outcome filter.
    Dollar-neutral sector buckets retain transparent size/beta residual risks.
    """
    output = np.zeros(len(score), dtype=np.float32)
    active = design.mask & np.isfinite(score) & (np.abs(score) > 1e-12)
    indices = np.flatnonzero(active)
    if len(indices) < 2 or maximum < 4:
        return output
    values = score[indices]
    longs = indices[np.argsort(-values, kind="stable")][: maximum // 2]
    sectors = frame.gics.fillna("unknown").astype(str).str[:2].to_numpy()
    cap = frame.me.fillna(0).to_numpy(dtype=float)
    adv = frame.dolvol_126d.fillna(0).to_numpy(dtype=float)
    eligible = (
        design.mask
        & (cap >= settings["large_cap_borrow_threshold_usd"] / 1e6)
        & (adv >= 20_000_000)
    )
    eligible[longs] = False
    hedge = np.flatnonzero(eligible)
    hedge = hedge[np.argsort(-adv[hedge], kind="stable")][: maximum // 2]
    if not len(hedge):
        return output
    # Allocate each long sector only when a matched liquid hedge exists. This
    # records an explicit coverage limitation rather than adding implicit shorts.
    supported = np.isin(sectors[longs], sectors[hedge])
    longs = longs[supported]
    if not len(longs):
        return output
    alpha_weight = np.ones(len(longs))
    # Smooth capacity/cost construction: every ranked long remains eligible,
    # while a tiny ADV name cannot absorb the same dollars as a liquid one.
    alpha_weight *= np.minimum(
        1, np.sqrt(np.maximum(adv[longs], 0) / 20_000_000)
    )
    if alpha_weight.sum() <= 0:
        return output
    alpha_weight /= alpha_weight.sum()
    for sector in np.unique(sectors[longs]):
        local_long = sectors[longs] == sector
        local_hedge = hedge[sectors[hedge] == sector]
        total = alpha_weight[local_long].sum()
        output[longs[local_long]] = alpha_weight[local_long]
        risk = (
            frame.rvol_21d.iloc[local_hedge]
            .fillna(0.02)
            .clip(0.005, 0.3)
            .to_numpy()
        )
        inverse = 1 / risk
        output[local_hedge] = -total * inverse / inverse.sum()
    gross = np.abs(output).sum()
    scale = min(
        settings["gross_limit"] / max(gross, 1e-12),
        settings["single_name_limit"] / max(np.abs(output).max(), 1e-12),
    )
    return output * scale


def construct_target(
    strategy, score, design, frame, settings, maximum, prepared=None
):
    if strategy.hedge_policy == "liquid_sector":
        return liquid_hedged_target(
            score, design, frame, settings, maximum=maximum
        )
    sizing_factors = None
    if strategy.construction == "liquidity_tilt":
        prepared = (
            prepare_forecast(score, design) if prepared is None else prepared
        )
        indices = prepared[0]
        adv = frame.dolvol_126d.fillna(0).to_numpy(float)[indices]
        sizing_factors = np.clip(adv / 20_000_000, 0, 1)
    return supported_target(
        score,
        design,
        settings,
        maximum=maximum,
        weighting=strategy.weighting,
        prepared=prepared,
        sizing_factors=sizing_factors,
    )
