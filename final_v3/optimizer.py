"""Selects and sizes long and short positions from monthly scores."""

import json
import hashlib
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import risk
import timing
import part2
import deployment
import turnover
import neutrality
import config
from config import NAV

# Every dollar limit derives from NAV (config.py) or a stock's ADV.
GROSS = 1.00  # Long plus short notional over NAV.
# Maximum unused own capital at each monthly rebalance. The short leg is
# separate collateral; gross equity exposure may never exceed 200% of NAV.
MAX_TREASURY_WEIGHT = .05
MAX_ONE_WAY_TURNOVER = None
MAX_ANNUAL_ONE_WAY_TURNOVER = 4.0
# Trading headroom for involuntary holding-month exits, not a cash allocation.
FORCED_EXIT_TURNOVER_RESERVE = .05
EXEMPT_INITIAL_DEPLOYMENT = True
N_PER_SIDE = 100
MIN_PRICE = 5.00  # Share price, in dollars, for longs and shorts.
EXECUTION_DAYS = 5
LONG_ADV_PARTICIPATION = 0.05
SHORT_ADV_PARTICIPATION = 0.02
REQUIRE_VERIFIED_LABELS = True
# Match long and short name counts within each GICS sector.
SECTOR_NEUTRAL = True
MIN_SECTOR_NAMES = 4
# A holding stays while it still ranks within (1 + BUFFER) x its quota.
BUFFER = 0.5
# Optional legacy cutoff. Disabled: signals are retained and actual positions
# must pass the per-name capacity and short-universe rules below.
ENABLE_MICRO_SLEEVE_SWITCH = False
MICRO_SLEEVES_MAX_NAV = 50_000_000.0
# A stock with this share of its shares outstanding already sold short cannot
# be borrowed at the size or fee the cost model assumes, so it is never
# shorted when crowding is observed. Official data has no such observations.
SHORT_INTEREST_CAP = 0.15
# The shortable universe is restricted to names with a market capitalisation of
# at least US$500m at the formation date (Instructions 3.1). This hard floor
# applies to the short leg only. Longs still pass the other eligibility and
# capacity checks; there is no competition rule banning a valuation signal.
# `me` is market equity in millions of dollars.
SHORT_ME_FLOOR = 500.0
# Long and short notionals are scaled so their 60-month betas offset; each
# side stays within BETA_TILT of half the gross, keeping net exposure small.
BETA_NEUTRAL = True
MAX_ABS_BETA = 0.05  # Hard daily AND monthly S&P 500 forecast-beta bands after turnover.
BETA_TILT = 0.25
BETA_BOUNDS = (0.2, 3.0)

_UNVERIFIED_LABEL = "no_verified_label_for_observation_period"
_ATTRIBUTES = [
    "final_score",
    "ticker",
    "company_name",
    "ticker_name_reference_date",
    "ticker_name_source",
    "ticker_name_status",
    "prc",
    "me",
    "size_grp",
    "dolvol_126d",
    "sector",
    "beta_60m",
    *neutrality.COLUMNS,
    'beta_sp500_daily_observations', 'beta_sp500_monthly_observations',
    'beta_sp500_daily_source', 'beta_sp500_monthly_source', 'beta_information_cutoff',
]


def dollar_limits(nav: float = NAV) -> dict[str, float]:
    """Book-level and per-name dollar sizes implied by a NAV."""
    side = nav * GROSS / 2
    return {
        "nav": float(nav),
        "gross_notional_dollars": float(nav * GROSS),
        "side_notional_dollars": float(side),
        "position_dollars": float(side / N_PER_SIDE),
    }


def scored_for_nav(predictions: pd.DataFrame, nav: float | pd.Series) -> pd.DataFrame:
    """Use the liquid score only when the optional legacy cutoff is enabled.

    The cutoff keys off the endowed fund size, not the compounded path, so
    the blend is fixed for the whole out-of-sample period.
    """
    frame = predictions.copy()
    endowment = (nav.endowment if hasattr(nav, 'endowment') else
                 float(nav.iloc[0]) if isinstance(nav, pd.Series) else float(nav))
    if (
        ENABLE_MICRO_SLEEVE_SWITCH
        and endowment > MICRO_SLEEVES_MAX_NAV
        and "final_score_liquid" in frame
    ):
        frame["final_score"] = frame.final_score_liquid
        frame["available_count"] = frame.available_count_liquid
    return frame


def optimize(
    predictions: pd.DataFrame,
    panel_path: pathlib.Path,
    nav: float | pd.Series = NAV,
    short_interest: pd.DataFrame | None = None,
    risk_cache: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds each month's book from the final score, then applies risk rules.

    A holding absent from a month's predictions has already exited at its
    last price, so only names present that month can be kept.

    Args:
        predictions: One row per stock-month with return_month, permno,
            final_score, price, liquidity, sector, beta and labels.
        panel_path: Characteristics panel holding the risk rules' history.
        nav: Capital that sets every dollar size. A scalar is the endowed
            amount; a Series indexed by holding month is the compounding
            capital the mandate requires (Instructions 3.1, 3.3), so the
            liquidity limits tighten as the fund grows.
        short_interest: Optional borrow.build table (permno, return_month,
            short_pct); stocks at or above SHORT_INTEREST_CAP are not shorted.

    Returns:
        Target positions per month, and per-month exposures and coverage.
    """
    timing.validate_predictions(predictions)
    frame = scored_for_nav(predictions, nav)
    frame = neutrality.attach(frame,panel_path,config.DATA_DIR,config.WORK_DIR)
    if not isinstance(frame.return_month.dtype,pd.PeriodDtype):
        frame["return_month"] = pd.to_datetime(frame.return_month).dt.to_period("M")
    if short_interest is not None:
        crowded = short_interest[["permno", "return_month", "short_pct"]].copy()
        crowded["return_month"] = pd.to_datetime(crowded.return_month).dt.to_period("M")
        frame = frame.merge(crowded, on=["permno", "return_month"], how="left")
    else:
        frame["short_pct"] = np.nan
    panel = pd.read_parquet(panel_path, columns=risk.PANEL_COLUMNS)
    panel["eom"] = pd.to_datetime(panel.eom).dt.to_period("M")
    scores = frame[["return_month", "permno", "final_score", "prc"]]
    cached_features = None
    if risk_cache is not None:
        panel_stat = panel_path.stat()
        score_hash = hashlib.sha256(pd.util.hash_pandas_object(scores, index=False).to_numpy().tobytes()).hexdigest()
        signature = (str(panel_path.resolve()), panel_stat.st_size, panel_stat.st_mtime_ns,
                     score_hash, risk.feature_cache_signature())
        if risk_cache.get('signature') != signature:
            risk_cache.clear()
            risk_cache.update(signature=signature, months={})
        cached_features = risk_cache['months']
    closed = set()
    if "ret_exc_lead1m" in pq.ParquetFile(panel_path).schema.names:
        outcomes = pd.read_parquet(
            panel_path, columns=["permno", "eom", "ret_exc_lead1m"]
        )
        outcomes = outcomes.loc[outcomes.ret_exc_lead1m.isna()]
        settled = pd.to_datetime(outcomes.eom).dt.to_period("M") + 1
        closed = set(zip(settled, outcomes.permno, strict=True))
    previous: dict[int, float] = {}
    positions, diagnostics = [], []
    annual_budget = (turnover.AnnualBudget(MAX_ANNUAL_ONE_WAY_TURNOVER,FORCED_EXIT_TURNOVER_RESERVE)
                     if MAX_ANNUAL_ONE_WAY_TURNOVER is not None else None)
    if (MAX_ONE_WAY_TURNOVER is not None or annual_budget is not None) and MAX_TREASURY_WEIGHT is None:
        raise ValueError('Turnover-constrained sizing currently requires a Treasury deployment target')
    for month, current in frame.groupby("return_month", sort=True):
        capital = (float(nav.capital(month)) if hasattr(nav, 'capital') else
                   float(nav.loc[month]) if isinstance(nav, pd.Series) else float(nav))
        # Only settled prior-month closures can clear the retention buffer.
        previous = {p: w for p, w in previous.items() if (month - 1, p) not in closed}
        prev_long = {p for p, w in previous.items() if w > 0}
        prev_short = {p for p, w in previous.items() if w < 0}
        weights, coverage = _book(current, capital, prev_long, prev_short)
        if MAX_TREASURY_WEIGHT is not None:
            long_target = 1.0 - MAX_TREASURY_WEIGHT + 1e-8
            initial = pd.Series(weights, dtype=float)
            beta = neutrality.matrix(current,initial.index).mean(axis=1).clip(*BETA_BOUNDS)
            lb = float((initial.clip(lower=0)*beta).sum()/initial.clip(lower=0).sum())
            sb = float((-initial.clip(upper=0)*beta).sum()/-initial.clip(upper=0).sum())
            hedge_target = long_target*lb/sb if BETA_NEUTRAL else long_target
            short_target = float(np.clip(hedge_target, .50, 2.0-long_target-1e-8))
            weights = deployment.resize(weights, current, capital, long_target, short_target,
                long_participation=LONG_ADV_PARTICIPATION, short_participation=SHORT_ADV_PARTICIPATION,
                execution_days=EXECUTION_DAYS,
                beta_limit=MAX_ABS_BETA if BETA_NEUTRAL else None)
            before_risk = pd.Series(weights, dtype=float)
        weights, squeeze, rules = risk.protect(weights, month, panel, scores, cache=cached_features)
        if MAX_TREASURY_WEIGHT is not None:
            # A hard cash ceiling supersedes global ES/edge cash scaling.
            # Individual peer and run-up protections still bound every short.
            global_scale = rules['risk_scale'] * rules['edge_scale']
            weights = {p: value/global_scale for p,value in weights.items()}
            shorts = before_risk.loc[before_risk < 0].abs()
            safe_caps = pd.Series(np.inf, index=before_risk.index)
            safe_caps.loc[shorts.index] = np.where(
                squeeze.loc[shorts.index,'peer_flagged'], shorts*risk.PEER_SCALE,
                risk.MAX_SHORT_MULTIPLE*shorts.mean()) * squeeze.loc[shorts.index,'runup_factor']
            short_target = max(long_target-.5+1e-8, -sum(v for v in weights.values() if v < 0))
            weights = deployment.resize(weights, current, capital, long_target, short_target,
                long_participation=LONG_ADV_PARTICIPATION, short_participation=SHORT_ADV_PARTICIPATION,
                execution_days=EXECUTION_DAYS, short_caps=safe_caps,
                beta_limit=MAX_ABS_BETA if BETA_NEUTRAL else None)
            squeeze['risk_multiplier'] = pd.Series(weights)/before_risk
            rules.update(treasury_cap=MAX_TREASURY_WEIGHT, global_cash_scaling_applied=False,
                treasury_weight=max(1-sum(v for v in weights.values() if v>0),0),
                short_weight_in_flagged=float(-pd.Series(weights).where(squeeze.peer_flagged,0).sum()))
            if MAX_ONE_WAY_TURNOVER is not None or annual_budget is not None:
                if not hasattr(nav, 'pretrade'):
                    raise ValueError('Hard turnover sizing requires causal pretrade holdings')
                drift = nav.pretrade(month)
                initial_exemption = not nav.books and EXEMPT_INITIAL_DEPLOYMENT
                rules['turnover_initial_deployment_exempt'] = initial_exemption
                rules['preferred_long_notional'] = sum(v for v in weights.values() if v>0)
                rules['preferred_short_notional'] = -sum(v for v in weights.values() if v<0)
                if not initial_exemption:
                    if annual_budget is not None:
                        preferred_budget,hard_budget = annual_budget.limits(month)
                    else:
                        preferred_budget = MAX_ONE_WAY_TURNOVER-FORCED_EXIT_TURNOVER_RESERVE
                        hard_budget = MAX_ONE_WAY_TURNOVER
                    if MAX_ONE_WAY_TURNOVER is not None:
                        hard_budget = min(hard_budget,MAX_ONE_WAY_TURNOVER)
                        preferred_budget = min(preferred_budget,hard_budget)
                    rules.update(turnover_preferred_budget=preferred_budget,turnover_hard_budget=hard_budget)
                    lo,so,_ = _eligible(current,capital)
                    reference = before_risk.to_dict()
                    for p,value in drift.items():
                        eligible = lo if value>0 else so
                        if p not in reference and p in set(eligible.permno):
                            reference[p] = (long_target if value>0 else -float(shorts.sum()))/N_PER_SIDE
                    _,expanded,_ = risk.protect(reference,month,panel,scores,cache=cached_features)
                    basis = pd.Series(reference,dtype=float)
                    limits = np.minimum(.05-1e-10,
                        current.set_index('permno').dolvol_126d.reindex(basis.index)
                        *EXECUTION_DAYS*np.where(basis>0,LONG_ADV_PARTICIPATION,SHORT_ADV_PARTICIPATION)/capital)
                    limits = pd.Series(limits,index=basis.index)
                    negative = basis.index[basis<0]
                    limits.loc[negative] = np.minimum(limits.loc[negative],
                        np.where(expanded.loc[negative,'peer_flagged'],
                            basis.loc[negative].abs()*risk.PEER_SCALE,
                            risk.MAX_SHORT_MULTIPLE*shorts.mean())*expanded.loc[negative,'runup_factor'])
                    try:
                        weights,turn_rules = turnover.constrain(weights,reference,current,drift,limits,
                            preferred_budget,N_PER_SIDE,
                            beta_limit=MAX_ABS_BETA if BETA_NEUTRAL else None)
                        turn_rules['turnover_reserve_fallback'] = False
                    except ValueError as error:
                        if 'infeasible' not in str(error).lower() and 'no feasible book' not in str(error).lower():
                            raise
                        weights,turn_rules = turnover.constrain(weights,reference,current,drift,limits,
                            hard_budget,N_PER_SIDE,minimize_trades=True,
                            beta_limit=MAX_ABS_BETA if BETA_NEUTRAL else None)
                        turn_rules['turnover_reserve_fallback'] = True
                    squeeze = expanded.reindex(pd.Index(weights)).copy()
                    squeeze['risk_reference_weight'] = basis.reindex(squeeze.index)
                    squeeze['short_safety_cap'] = limits.reindex(squeeze.index).where(basis.reindex(squeeze.index)<0)
                    squeeze['risk_multiplier'] = pd.Series(weights)/basis
                    rules.update(turn_rules)
                else:
                    rules['rebalance_one_way_turnover'] = sum(abs(v) for v in weights.values())/2
                rules['short_weight_in_flagged'] = float(-pd.Series(weights).where(squeeze.peer_flagged,0).sum())
            part2.enforce_book(current, weights)
            if sum(v>0 for v in weights.values()) != N_PER_SIDE or sum(v<0 for v in weights.values()) != N_PER_SIDE:
                raise ValueError(f'{month}: Treasury sizing lacks {N_PER_SIDE} names on each leg')
        final_beta = neutrality.enforce(weights, current,
            MAX_ABS_BETA if BETA_NEUTRAL else None)
        rules.update(final_beta_limit=MAX_ABS_BETA if BETA_NEUTRAL else None,
                     final_beta_source=neutrality.SOURCE,
                     final_beta_daily_prior_names=int(current.set_index('permno').beta_sp500_daily_source.reindex(list(weights)).ne('stock_regression').sum()),
                     final_beta_monthly_prior_names=int(current.set_index('permno').beta_sp500_monthly_source.reindex(list(weights)).ne('stock_regression').sum()),
                     beta_exposure_monthly=float(neutrality.exposures(weights,current)['beta_sp500_monthly']))
        attributes = current.set_index("permno")[_ATTRIBUTES]
        for permno, weight in weights.items():
            row = attributes.loc[permno]
            positions.append(
                {
                    "return_month": month,
                    "permno": permno,
                    "ticker": row.ticker,
                    "company_name": row.company_name,
                    "ticker_name_reference_date": row.ticker_name_reference_date,
                    "ticker_name_source": row.ticker_name_source,
                    "ticker_name_status": row.ticker_name_status,
                    "prc": float(row.prc),
                    "me": float(row.me),
                    "size_grp": row.size_grp,
                    "sector": row.sector,
                    "beta_60m": float(row.beta_60m),
                    **{name:row[name] for name in [*neutrality.COLUMNS,
                       'beta_sp500_daily_observations','beta_sp500_monthly_observations',
                       'beta_sp500_daily_source','beta_sp500_monthly_source','beta_information_cutoff']},
                    "dolvol_126d": float(row.dolvol_126d),
                    "final_score": float(row.final_score),
                    "target_weight": weight,
                    "target_dollars": weight * capital,
                    "side": "long" if weight > 0 else "short",
                    "capital": capital,
                    **squeeze.loc[permno].to_dict(),
                }
            )
        book = pd.Series(weights, dtype=float)
        sectors = attributes.sector.reindex(book.index)
        names = set(weights) | set(previous)
        traded = sum(abs(weights.get(p, 0.0) - previous.get(p, 0.0)) for p in names)
        if 'rebalance_one_way_turnover' in rules:
            traded = 2*rules['rebalance_one_way_turnover']
        diagnostics.append(
            {
                "return_month": month,
                "long_notional": float(book[book > 0].sum()),
                "short_notional": float(-book[book < 0].sum()),
                "net_exposure": float(book.sum()),
                "sector_net_abs": float(book.groupby(sectors).sum().abs().sum()),
                "beta_exposure": final_beta,
                "turnover_nav": float(traded),
                **coverage,
                **rules,
            }
        )
        previous = weights
        if hasattr(nav, 'settle'):
            nav.settle(month, weights)
            if annual_budget is not None:
                initial_deployment = (sum(abs(v) for v in weights.values())/2
                                      if rules.get('turnover_initial_deployment_exempt',False) else 0.0)
                diagnostics[-1].update(annual_budget.settle(month,
                    nav.ledger.loc[month,'traded']/2,initial_deployment))
            if (MAX_ONE_WAY_TURNOVER is not None
                and not rules.get('turnover_initial_deployment_exempt',False)
                and nav.ledger.loc[month,'traded']/2>MAX_ONE_WAY_TURNOVER+1e-9):
                raise ValueError(f'{month}: realized turnover including forced exits exceeds cap')
            print(f'Finalized portfolio {month}: capital ${capital:,.0f}', flush=True)
    return pd.DataFrame(positions), pd.DataFrame(diagnostics)


def write(
    portfolio: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: pathlib.Path,
    nav: float = NAV,
) -> dict:
    """Writes the book, its diagnostics and the construction settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stored = portfolio.copy()
    stored["return_month"] = stored.return_month.dt.to_timestamp("M")
    stored.to_parquet(output_dir / "portfolio.parquet", index=False, compression="zstd")
    stored = diagnostics.copy()
    stored["return_month"] = stored.return_month.astype(str)
    stored.to_csv(output_dir / "optimizer_diagnostics.csv", index=False)
    config = {
        **dollar_limits(nav),
        "dollar_limits_basis": "initial selection screen; final weights obey Treasury cap and gross/net/ADV bounds",
        "names_per_side": N_PER_SIDE,
        "max_treasury_weight": MAX_TREASURY_WEIGHT,
        "max_one_way_turnover": MAX_ONE_WAY_TURNOVER,
        "max_annual_one_way_turnover": MAX_ANNUAL_ONE_WAY_TURNOVER,
        "annual_turnover_policy": "calendar-year sum of monthly one-way turnover, including forced exits; initial funding excluded when configured. Pace remaining budget over remaining calendar months; formation-only minimum-trade fallback can borrow future monthly allocations. No fixed monthly ceiling.",
        "forced_exit_turnover_reserve": FORCED_EXIT_TURNOVER_RESERVE,
        "exempt_initial_deployment": EXEMPT_INITIAL_DEPLOYMENT,
        "turnover_policy": "half absolute signed dollar changes from drifted pretrade holdings / current NAV; involuntary within-month liquidations reported separately",
        "deployment_policy": "cash cap supersedes global ES/edge scaling; individual short protections and final capacity bounds retained",
        "min_price_dollars": MIN_PRICE,
        "execution_days": EXECUTION_DAYS,
        "adv_participation": {
            "long": LONG_ADV_PARTICIPATION,
            "short": SHORT_ADV_PARTICIPATION,
        },
        "require_verified_point_in_time_labels": REQUIRE_VERIFIED_LABELS,
        "sector_neutral": SECTOR_NEUTRAL,
        "rank_buffer": BUFFER,
        "short_interest_cap": SHORT_INTEREST_CAP,
        "short_me_floor_musd": SHORT_ME_FLOOR,
        "micro_sleeves_max_nav": MICRO_SLEEVES_MAX_NAV,
        "micro_sleeve_switch_enabled": ENABLE_MICRO_SLEEVE_SWITCH,
        "parameter_source": "fixed_recipe_with_user_requested_final_formation_beta_band",
        "beta_neutral": BETA_NEUTRAL,
        "max_abs_beta": MAX_ABS_BETA if BETA_NEUTRAL else None,
        "beta_constraint_source": neutrality.SOURCE,
        "beta_constraint_windows": {
            'daily_sessions':neutrality.DAILY_WINDOW,'daily_minimum':neutrality.DAILY_MIN,
            'monthly_observations':neutrality.MONTHLY_WINDOW,'monthly_minimum':neutrality.MONTHLY_MIN},
        "risk": {
            "peer_top_share": risk.PEER_TOP_SHARE,
            "peer_scale": risk.PEER_SCALE,
            "squeeze_move": risk.SQUEEZE_MOVE,
            "max_short_multiple": risk.MAX_SHORT_MULTIPLE,
            "runup_cap": risk.RUNUP_CAP,
            "expected_shortfall_tail": risk.ES_TAIL,
            "expected_shortfall_norm_months": risk.ES_NORM_MONTHS,
            "expected_shortfall_tolerance": risk.ES_TOLERANCE,
            "expected_shortfall_floor": risk.ES_FLOOR,
            "edge_months": risk.EDGE_MONTHS,
            "edge_tail": risk.EDGE_TAIL,
            "edge_floor": risk.EDGE_FLOOR,
            "edge_step": risk.EDGE_STEP,
        },
        "months": int(portfolio.return_month.nunique()),
        "average_turnover_nav": float(diagnostics.turnover_nav.mean()),
        "average_net_exposure": float(diagnostics.net_exposure.mean()),
        "max_abs_net_exposure": float(diagnostics.net_exposure.abs().max()),
    }
    (output_dir / "optimizer_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    return config


def _eligible(frame: pd.DataFrame, nav: float):
    """The same formation-date entry and retention eligibility checks."""
    side = GROSS / 2
    one = nav * side / N_PER_SIDE
    valid = frame.loc[
        (frame.available_count > 0)
        & np.isfinite(frame.final_score)
        & np.isfinite(frame.prc)
        & (frame.dolvol_126d > 0)
    ]
    # Unknown classifications cannot enter a mandate that bounds every GICS
    # sector. Filter before quotas so an "na" group cannot receive holdings.
    unknown_sector_blocked = int((~valid.sector.isin(part2.SECTORS)).sum())
    valid = valid.loc[valid.sector.isin(part2.SECTORS)]
    if REQUIRE_VERIFIED_LABELS:
        valid = valid.loc[
            valid.ticker.notna()
            & valid.company_name.notna()
            & valid.ticker_name_status.ne(_UNVERIFIED_LABEL)
        ]
    long_ok = valid[
        (valid.prc >= MIN_PRICE)
        & (valid.dolvol_126d * LONG_ADV_PARTICIPATION * EXECUTION_DAYS >= one)
    ]
    short_ok = valid[
        (valid.prc >= MIN_PRICE)
        & (valid.me >= SHORT_ME_FLOOR)
        & (valid.dolvol_126d * SHORT_ADV_PARTICIPATION * EXECUTION_DAYS >= one)
    ]
    scarce = short_ok.short_pct >= SHORT_INTEREST_CAP  # NaN compares False
    short_ok = short_ok.loc[~scarce]
    return long_ok,short_ok,{
        "candidates":len(valid), "long_eligible":len(long_ok), "short_eligible":len(short_ok),
        "short_blocked":int(scarce.sum()), "short_floor_blocked":int((valid.me<SHORT_ME_FLOOR).sum()),
        "unknown_sector_blocked":unknown_sector_blocked}


def _book(
    frame: pd.DataFrame, nav: float, prev_long: set, prev_short: set
) -> tuple[dict[int, float], dict]:
    """Equal-weight sector-matched longs and shorts that ADV can absorb."""
    side = GROSS/2
    long_ok,short_ok,coverage = _eligible(frame,nav)
    if SECTOR_NEUTRAL:
        counts = pd.concat([long_ok.sector, short_ok.sector]).value_counts()
        quota = _quota(counts[counts >= MIN_SECTOR_NAMES], N_PER_SIDE)
        groups = [
            (long_ok[long_ok.sector == s], short_ok[short_ok.sector == s], q)
            for s, q in quota.items()
            if q > 0
        ]
    else:
        groups = [(long_ok, short_ok, N_PER_SIDE)]
    longs, shorts = [], []
    for lo, so, k in groups:
        k = min(k, len(lo), len(so))
        short_ids = _pick(so, k, prev_short, ascending=True)
        long_ids = _pick(lo[~lo.permno.isin(short_ids)], k, prev_long, ascending=False)
        k = min(len(short_ids), len(long_ids))
        longs += long_ids[:k]
        shorts += short_ids[:k]
    coverage.update(selected_longs=len(longs),selected_shorts=len(shorts))
    if not longs or not shorts:
        raise ValueError('Part 2 infeasible portfolio: no eligible long/short book')
    long_notional = short_notional = side
    if BETA_NEUTRAL:
        beta = frame.set_index("permno").beta_60m.fillna(1.0).clip(*BETA_BOUNDS)
        long_beta = float(beta.reindex(longs).mean())
        short_beta = float(beta.reindex(shorts).mean())
        long_notional = float(
            np.clip(
                GROSS * short_beta / (long_beta + short_beta),
                side * (1 - BETA_TILT),
                side * (1 + BETA_TILT),
            )
        )
        short_notional = GROSS - long_notional
    weights = {int(p): long_notional / len(longs) for p in longs}
    weights.update({int(p): -short_notional / len(shorts) for p in shorts})
    part2.enforce_book(frame, weights)
    return weights, coverage


def _quota(counts: pd.Series, total: int) -> pd.Series:
    """Largest-remainder split of `total` names in proportion to `counts`."""
    base = counts / counts.sum() * total
    quota = np.floor(base).astype(int)
    remainder = (base - quota).sort_values(ascending=False)
    for group in remainder.index[: total - int(quota.sum())]:
        quota[group] += 1
    return quota


def _pick(candidates: pd.DataFrame, k: int, incumbents: set, ascending: bool) -> list:
    """Best k by score, first keeping incumbents ranked within the buffer."""
    if k <= 0 or candidates.empty:
        return []
    order = candidates.sort_values(
        ["final_score", "permno"], ascending=[ascending, True]
    ).permno.to_numpy()
    zone = set(order[: int(np.ceil(k * (1 + BUFFER)))]) if BUFFER else set()
    kept = [p for p in order if p in incumbents and p in zone][:k]
    taken = set(kept)
    return kept + [p for p in order if p not in taken][: k - len(kept)]
