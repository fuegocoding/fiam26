"""Monthly and daily portfolio evaluation with drifted-weight turnover,
missing-return last-quote liquidation, size-tiered transaction/borrow costs,
and daily mark-to-market analytics on dsf_mini_from_2014.csv.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

NAME_COST_BP: dict[str, float] = {
    "mega": 5.0,
    "large": 10.0,
    "small": 20.0,
    "micro": 40.0,
    "nano": 60.0,
}

BORROW_BPS: dict[str, float] = {
    "mega": 25.0,
    "large": 40.0,
    "mid": 100.0,
    "small": 300.0,
    "micro": 700.0,
    "nano": 1500.0,
}

# Final Guide Section 7 conservative flat alternative: 50 bp large-cap, 5% (500 bp) others
CONSERVATIVE_BORROW_BPS: dict[str, float] = {
    "mega": 50.0,
    "large": 50.0,
    "mid": 500.0,
    "small": 500.0,
    "micro": 500.0,
    "nano": 500.0,
}

HURDLE = 0.04
TRADING_DAYS = 252


def load_evaluation_universe(panel_path: Path) -> pd.DataFrame:
    """Formation fields, realized next-month excess return, and last-quote exit prices."""
    cols = [
        "permno",
        "eom",
        "prc",
        "size_grp",
        "dolvol_126d",
        "me",
        "ret_exc_lead1m",
        "gics",
        "beta_60m",
        "be_me",
        "ret_12_1",
    ]
    u = pd.read_parquet(panel_path, columns=cols)
    u["return_month"] = pd.to_datetime(u["eom"]).dt.to_period("M") + 1
    u["sector"] = u["gics"].astype("string").str[:2].fillna("na")
    u["size_grp"] = u["size_grp"].astype("string")
    quotes = u.loc[u["prc"].notna() & u["prc"].abs().gt(0), ["permno", "eom", "prc"]].copy()
    quotes["quote_date"] = pd.to_datetime(quotes.pop("eom")).astype("datetime64[ns]")
    quotes["exit_price"] = quotes.pop("prc").abs()
    quotes = quotes.drop_duplicates(["permno", "quote_date"])
    u["liquidation_cutoff"] = u["return_month"].dt.to_timestamp("M").astype("datetime64[ns]")
    u = pd.merge_asof(
        u.sort_values("liquidation_cutoff"),
        quotes.sort_values("quote_date"),
        left_on="liquidation_cutoff",
        right_on="quote_date",
        by="permno",
        direction="backward",
    ).rename(columns={"quote_date": "exit_price_date"})
    u = u.drop(columns=["eom", "gics", "liquidation_cutoff"])
    u = u.drop_duplicates(["return_month", "permno"]).reset_index(drop=True)
    for c in ["prc", "dolvol_126d", "me", "ret_exc_lead1m", "beta_60m", "be_me", "ret_12_1"]:
        u[c] = u[c].astype("float64")
    return u


def load_fred(external_path: Path) -> pd.DataFrame:
    ext = pd.read_parquet(external_path, columns=["table", "date", "series", "value"])
    fred = ext[ext["table"] == "fred"].drop(columns=["table"]).copy()
    fred["date"] = pd.to_datetime(fred["date"])
    return fred.reset_index(drop=True)


def load_market_series(external_path: Path) -> pd.DataFrame:
    """Monthly T-bill rate and S&P 500 price return keyed by holding month (Period['M'])."""
    fred = load_fred(external_path)
    tbill = fred[fred["series"] == "TB3MS"].set_index("date")["value"]
    tbill.index = tbill.index.to_period("M")
    sp500 = fred[fred["series"] == "SP500"].dropna(subset=["value"])
    close = sp500.set_index("date")["value"].resample("ME").last()
    spx = close.pct_change()
    spx.index = spx.index.to_period("M")
    return pd.DataFrame({"rf": tbill / 100.0 / 12.0, "spx": spx})


def position_returns(cur: pd.DataFrame, rf: float) -> tuple[pd.Series, pd.Series]:
    """Missing returns trigger a last-quote liquidation, never a zero-label fill."""
    realized = cur["ret_exc_lead1m"].astype(float).copy()
    missing = realized.isna()
    if missing.any():
        needed = {"prc", "exit_price", "exit_price_date"}
        if not needed.issubset(cur.columns):
            raise ValueError("Liquidation requires opening price and a dated last quote")
        entry = cur["prc"].abs()
        exit_price = cur["exit_price"]
        invalid = missing & (
            entry.isna() | entry.le(0) | exit_price.isna() | exit_price.le(0)
        )
        if invalid.any():
            raise ValueError("No usable price for missing-return liquidation")
        realized.loc[missing] = exit_price.loc[missing] / entry.loc[missing] - 1.0 - rf
    return realized, missing


def evaluate_monthly_books(
    books: dict[pd.Period, dict[int, float]],
    universe: pd.DataFrame,
    mkt: pd.DataFrame,
    *,
    cost_bp_map: dict[str, float] = NAME_COST_BP,
    borrow_bps_map: dict[str, float] = BORROW_BPS,
    cost_multiplier: float = 1.0,
    borrow_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Evaluates monthly books with drifted-weight turnover and last-quote liquidation."""
    keyed = universe.set_index(["return_month", "permno"])
    rows, exits = [], []
    prev: dict[int, float] = {}
    prev_r: dict[int, float] = {}
    prev_size: dict[int, str] = {}
    prev_rf, prev_total = 0.0, 0.0

    for m in sorted(books):
        w = {int(k): float(v) for k, v in books[m].items() if v != 0.0}
        if not w:
            continue
        rf = float(mkt["rf"].get(m, np.nan))
        ids = list(w)
        cur = keyed.reindex(pd.MultiIndex.from_arrays([[m] * len(ids), ids]))
        realized, liquidated = position_returns(cur, rf)
        ret = pd.Series(realized.to_numpy(), index=ids)
        liquidated = pd.Series(liquidated.to_numpy(), index=ids)
        size = pd.Series(cur["size_grp"].astype("string").to_numpy(), index=ids)
        sector = pd.Series(cur["sector"].to_numpy(), index=ids)
        beta = pd.Series(cur["beta_60m"].to_numpy(), index=ids)

        drift = {
            p: x * (1.0 + prev_r.get(p, 0.0) + prev_rf) / (1.0 + prev_total)
            for p, x in prev.items()
        }
        cost = traded = 0.0
        for p in set(w) | set(drift):
            d = abs(w.get(p, 0.0) - drift.get(p, 0.0))
            if d:
                tier = size.get(p) if p in w else prev_size.get(p)
                bp = cost_bp_map.get(str(tier), 25.0) * cost_multiplier
                cost += d * bp / 1e4
                traded += d

        for p in liquidated[liquidated].index:
            j = ids.index(p)
            exit_notional = abs(w[p]) * (1.0 + ret[p] + rf)
            bp = cost_bp_map.get(str(size[p]), 25.0) * cost_multiplier
            exit_cost = exit_notional * bp / 1e4
            cost += exit_cost
            traded += exit_notional
            exits.append(
                {
                    "return_month": m,
                    "permno": p,
                    "opening_weight": w[p],
                    "opening_price": float(cur["prc"].iloc[j]),
                    "exit_price": float(cur["exit_price"].iloc[j]),
                    "exit_price_date": cur["exit_price_date"].iloc[j],
                    "exit_notional": exit_notional,
                    "exit_cost": exit_cost,
                    "price_return": ret[p] + rf,
                    "reason": "missing_return_last_available_quote",
                }
            )

        ws = pd.Series(w)
        neg = ws < 0
        borrow_rates = size[neg].map(borrow_bps_map).astype(float).fillna(700.0) * borrow_multiplier
        borrow = float((-ws[neg] * borrow_rates).sum()) / 1e4 / 12.0
        pnl = float((ws * ret).sum())
        net_sector = ws.groupby(sector).sum()
        small = ws[neg & size.isin(["micro", "nano"])].abs().sum()

        rows.append(
            {
                "month": m,
                "rf": rf,
                "pnl": pnl,
                "pnl_long": float((ws[~neg] * ret[~neg]).sum()),
                "pnl_short": float((ws[neg] * ret[neg]).sum()),
                "cost": cost,
                "borrow": borrow,
                "gross_return": rf + pnl,
                "total": rf + pnl - cost - borrow,
                "traded": traded,
                "turnover_oneway": traded / 2.0,
                "names": len(w),
                "longs": int((~neg).sum()),
                "shorts": int(neg.sum()),
                "long_notional": float(ws[~neg].sum()),
                "short_notional": float(-ws[neg].sum()),
                "gross": float(ws.abs().sum()),
                "net": float(ws.sum()),
                "sector_bet": float(net_sector.abs().sum()),
                "max_sector": float(net_sector.abs().max()),
                "beta_exante": float((ws * beta.fillna(1.0)).sum()),
                "short_microcap": float(small / max(-ws[neg].sum(), 1e-12)),
                "missing": int(cur["ret_exc_lead1m"].isna().sum()),
            }
        )
        prev = {p: x for p, x in w.items() if not liquidated[p]}
        prev_r = ret.to_dict()
        prev_rf = rf
        prev_total = rf + pnl - cost - borrow
        prev_size.update(size.to_dict())

    out = pd.DataFrame(rows).set_index("month")
    out["spx"] = mkt["spx"].reindex(out.index)
    out["benchmark"] = out["rf"] + HURDLE / 12.0
    out["active"] = out["total"] - out["benchmark"]
    out.attrs["liquidations"] = pd.DataFrame(exits)
    return out


def ols_hac(y: np.ndarray, x: np.ndarray, lags: int = 3) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with an intercept; Newey-West (HAC) t-statistics and R2."""
    x_mat = np.column_stack([np.ones(len(y)), x])
    b = np.linalg.lstsq(x_mat, y, rcond=None)[0]
    e = y - x_mat @ b
    xtxi = np.linalg.inv(x_mat.T @ x_mat)
    s = (x_mat * e[:, None]).T @ (x_mat * e[:, None])
    for lag in range(1, lags + 1):
        g = (x_mat[lag:] * e[lag:, None]).T @ (x_mat[:-lag] * e[:-lag, None])
        s += (1.0 - lag / (lags + 1.0)) * (g + g.T)
    t = b / np.sqrt(np.diag(xtxi @ s @ xtxi))
    denom = float((y - y.mean()) @ (y - y.mean()))
    r2 = float(1.0 - (e @ e) / denom) if denom > 0 else 0.0
    return b, t, r2


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain OLS with an intercept: coefficients and classical standard errors."""
    x_mat = np.column_stack([np.ones(len(y)), x])
    b = np.linalg.lstsq(x_mat, y, rcond=None)[0]
    e = y - x_mat @ b
    s2 = (e @ e) / max(1, len(y) - x_mat.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(x_mat.T @ x_mat)))
    return b, se


def evaluate_cost_borrow_sensitivity(
    books: dict[pd.Period, dict[int, float]],
    universe: pd.DataFrame,
    mkt: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compares tiered baseline vs Final Guide Section 7 conservative flat borrow and 2x costs."""
    scenarios = {
        "tiered_baseline": (NAME_COST_BP, BORROW_BPS, 1.0, 1.0),
        "section7_conservative_flat_borrow": (NAME_COST_BP, CONSERVATIVE_BORROW_BPS, 1.0, 1.0),
        "double_transaction_cost_2x": (NAME_COST_BP, BORROW_BPS, 2.0, 1.0),
        "double_cost_and_conservative_borrow": (NAME_COST_BP, CONSERVATIVE_BORROW_BPS, 2.0, 1.0),
    }
    results: dict[str, dict[str, float]] = {}
    for label, (c_map, b_map, c_mult, b_mult) in scenarios.items():
        r = evaluate_monthly_books(
            books,
            universe,
            mkt,
            cost_bp_map=c_map,
            borrow_bps_map=b_map,
            cost_multiplier=c_mult,
            borrow_multiplier=b_mult,
        )
        active = r["active"]
        ex = r["total"] - r["rf"]
        cagr_val = float((1.0 + r["total"]).prod() ** (12.0 / len(r)) - 1.0)
        ir_val = float(math.sqrt(12.0) * active.mean() / active.std()) if active.std() > 0 else 0.0
        sh_val = float(math.sqrt(12.0) * ex.mean() / ex.std()) if ex.std() > 0 else 0.0
        results[label] = {
            "cagr": cagr_val,
            "information_ratio": ir_val,
            "sharpe_ratio": sh_val,
            "annual_trading_cost_pct": float(r["cost"].mean() * 12.0),
            "annual_borrow_cost_pct": float(r["borrow"].mean() * 12.0),
            "annual_total_cost_pct": float((r["cost"].mean() + r["borrow"].mean()) * 12.0),
        }
    return results


def load_dsf_daily(
    dsf_path: Path,
    permnos: set[int] | None = None,
    first_date: pd.Timestamp | None = None,
    last_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Memory-lean reader for dsf_mini_from_2014.csv using pyarrow.csv."""
    import pyarrow.csv as pv

    table = pv.read_csv(
        dsf_path,
        convert_options=pv.ConvertOptions(
            column_types={"permno": "int64", "date": "int64"}
        ),
    )
    d = table.to_pandas()
    if first_date is not None:
        first_int = int(first_date.strftime("%Y%m%d"))
        d = d.loc[d["date"] >= first_int]
    if last_date is not None:
        last_int = int(last_date.strftime("%Y%m%d"))
        d = d.loc[d["date"] <= last_int]
    # Calendar comes from every source security in range, before held-name filtering.
    calendar = pd.to_datetime(sorted(d['date'].dropna().unique()), format='%Y%m%d')
    if permnos is not None:
        d = d.loc[d["permno"].isin(permnos)].copy()
    d["day"] = pd.to_datetime(d["date"], format="%Y%m%d")
    d = d.sort_values(["permno", "day"]).reset_index(drop=True)
    d.attrs['trading_calendar'] = list(calendar)
    return d


def _daily_pivot(dsf: pd.DataFrame) -> pd.DataFrame:
    required = {'day', 'permno', 'daily_return'}
    if not required.issubset(dsf):
        raise ValueError('daily input requires day, permno and daily_return')
    if dsf[['day', 'permno']].isna().any().any() or dsf.duplicated(['day', 'permno']).any():
        raise ValueError('daily identifiers/dates must be present and unique')
    values = pd.to_numeric(dsf['daily_return'], errors='raise')
    observed = values.dropna()
    if not np.isfinite(observed).all() or (observed < -1).any():
        raise ValueError('daily total returns must be finite and at least -1; never clipped')
    frame = dsf.assign(daily_return=values, day=pd.to_datetime(dsf['day']))
    px = frame.pivot(index='day', columns='permno', values='daily_return').sort_index()
    calendar = dsf.attrs.get('trading_calendar')
    if calendar is not None:
        dates = pd.DatetimeIndex(calendar).sort_values()
        if dates.has_duplicates or dates.isna().any() or not px.index.isin(dates).all():
            raise ValueError('invalid explicit daily source calendar')
        px = px.reindex(dates)
    return px


def _held_daily(px: pd.DataFrame, month, weights) -> pd.DataFrame:
    days = px.loc[px.index.to_period('M') == month]
    if days.empty:
        raise ValueError(f'no supplied daily observations for held month {month}')
    missing = set(weights) - set(days.columns)
    if missing:
        raise ValueError(f'missing daily held identifiers: {sorted(missing)}')
    selected = days[list(weights)]
    if selected.isna().any().any():
        raise ValueError(f'missing held daily returns in {month}; supply explicit exit/suspense evidence')
    return selected


def evaluate_daily_spine(
    books: dict[pd.Period, dict[int, float]],
    dsf: pd.DataFrame,
    fred: pd.DataFrame,
    monthly_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Daily total-return analytics; monthly cost allocation is a declared approximation.

    Use reporting.daily.reconcile_daily with actual dated fills/cash for an exact
    execution ledger. Monthly target weights do not establish daily entry dates.
    """
    tbill = fred[fred["series"] == "TB3MS"].dropna(subset=["value"]).copy()
    tbill["day"] = pd.to_datetime(tbill["date"])
    rf_daily = (tbill.set_index("day")["value"] / 100.0 / TRADING_DAYS).sort_index()

    spx_rows = fred[fred["series"] == "SP500"].dropna(subset=["value"]).copy()
    spx_rows["day"] = pd.to_datetime(spx_rows["date"])
    spx_close = spx_rows.set_index("day")["value"].sort_index()
    spx_ret = spx_close.pct_change().dropna()

    px = _daily_pivot(dsf)
    total_rows = []
    wealth = 1.0
    for month in sorted(books):
        w = {int(k): float(v) for k, v in books[month].items() if v != 0.0}
        if not w:
            continue
        if not all(np.isfinite(v) for v in w.values()):
            raise ValueError('held daily weights must be finite')
        days = _held_daily(px, month, w)
        weights = pd.Series(w, dtype=float)
        cols = list(weights.index)
        # Amortize monthly trading + borrow drag across trading days if monthly_df supplied
        daily_drag = 0.0
        if monthly_df is not None and month in monthly_df.index:
            m_row = monthly_df.loc[month]
            monthly_drag = float(m_row["cost"] + m_row["borrow"])
            if not np.isfinite(monthly_drag) or monthly_drag < 0:
                raise ValueError('monthly modeled costs must be finite and nonnegative')
            daily_drag = wealth * monthly_drag / len(days)
        for day, row in days[cols].iterrows():
            r = row
            rfd = float(rf_daily.get(day, np.nan))
            if not np.isfinite(rfd):
                rfd = float(rf_daily.asof(day))
            if not np.isfinite(rfd):
                raise ValueError(f'missing dated risk-free input at {day}')
            pnl = float((weights[cols] * r).sum())
            cash_income = (1.0-float(weights.sum())) * rfd
            cost_fraction = daily_drag / wealth
            net_d = cash_income + pnl - cost_fraction
            if not np.isfinite(net_d) or net_d <= -1:
                raise ValueError(f'nonpositive or invalid daily NAV at {day}')
            total_rows.append(
                {
                    "day": day,
                    "month": str(month),
                    "gross_daily": cash_income + pnl,
                    "total": net_d,
                    "rf": rfd,
                    "benchmark_daily": rfd + HURDLE / TRADING_DAYS,
                }
            )
            wealth *= 1.0 + net_d
            weights[cols] = weights[cols] * (1.0 + r) / (1.0 + net_d)

    if not total_rows:
        return pd.DataFrame(), {}

    d = pd.DataFrame(total_rows).set_index("day")
    d = d.join(spx_ret.rename("spx"), how="left")
    d["excess"] = d["total"] - d["rf"]
    d["mkt_excess"] = d["spx"] - d["rf"]
    d["active"] = d["total"] - d["benchmark_daily"]
    d["wealth"] = (1.0 + d["total"]).cumprod()
    d["drawdown"] = d["wealth"] / d["wealth"].cummax().clip(lower=1.0) - 1.0
    cov_252 = d["excess"].rolling(TRADING_DAYS, min_periods=63).cov(d["mkt_excess"])
    var_252 = d["mkt_excess"].rolling(TRADING_DAYS, min_periods=63).var()
    d["rolling_252d_beta"] = cov_252 / var_252.replace(0.0, np.nan)

    worst = d["total"].idxmin()
    best = d["total"].idxmax()
    summary: dict[str, object] = {
        'evidence_class': 'daily_return_analytics_with_monthly_target_and_cost_assumptions',
        'actual_dated_execution_ledger': False,
        'monthly_cost_timing': 'allocated equally in dollars over supplied source trading days; modeled, not actual fills',
        'missing_held_returns': 'fail_closed',
        'missing_market_returns': int(d['spx'].isna().sum()),
        "days": int(len(d)),
        "max_one_day_loss": float(d["total"].min()),
        "max_one_day_loss_date": str(pd.Timestamp(worst).date()),
        "best_one_day_return": float(d["total"].max()),
        "best_one_day_return_date": str(pd.Timestamp(best).date()),
        "ann_volatility_daily": float(d["total"].std() * math.sqrt(TRADING_DAYS)),
        "max_drawdown_daily": float(d["drawdown"].min()),
    }
    valid_reg = d.dropna(subset=["excess", "mkt_excess"])
    if len(valid_reg) > 20:
        b, se = ols(valid_reg["excess"].to_numpy(), valid_reg["mkt_excess"].to_numpy()[:, None])
        _, th, _ = ols_hac(
            valid_reg["excess"].to_numpy(), valid_reg["mkt_excess"].to_numpy()[:, None], lags=5
        )
        summary.update(
            {
                "beta_daily": float(b[1]),
                "beta_daily_se": float(se[1]),
                "beta_daily_t_hac": float(th[1]),
                "alpha_daily_ann": float(b[0] * TRADING_DAYS),
                "alpha_daily_t": float(b[0] / se[0]),
                "alpha_daily_t_hac": float(th[0]),
            }
        )
    return d.reset_index(), summary


def daily_alpha_decay(
    dsf: pd.DataFrame,
    predictions: pd.DataFrame,
    score_col: str = "final_score",
    horizons_days: tuple[int, ...] = (1, 3, 5, 10, 21, 42, 63),
) -> dict[str, object]:
    """Measures daily top-minus-bottom quintile cumulative return spread after formation."""
    px = _daily_pivot(dsf)
    preds = predictions[["return_month", "permno", score_col]].dropna().copy()
    preds["return_month"] = pd.to_datetime(preds["return_month"]).dt.to_period("M")
    spreads_by_h = {h: [] for h in horizons_days}
    missing_by_h = {h: 0 for h in horizons_days}
    for month, grp in preds.groupby("return_month"):
        # Rank the formation cohort before observing future return availability.
        if len(grp) < 50 or grp['permno'].duplicated().any():
            continue
        if not np.isfinite(grp[score_col]).all():
            raise ValueError('decay scores must be finite')
        positions = np.flatnonzero(px.index.to_period('M') == month)
        if not len(positions):
            continue
        # Information at preceding month end, entry at first trading close;
        # first earned total return belongs to the second supplied trading day.
        start_pos = int(positions[0])+1
        sub = grp.copy()
        sub['q'] = pd.qcut(sub[score_col].rank(method='first'), 5, labels=False)
        top = sub.loc[sub['q']==4, 'permno'].astype(int).tolist()
        bottom = sub.loc[sub['q']==0, 'permno'].astype(int).tolist()
        for h in horizons_days:
            if h < 1:
                raise ValueError('decay horizons must be positive trading-day counts')
            if start_pos+h > len(px):
                continue  # Outcome is not yet mature; never invent a zero return.
            if set(top+bottom)-set(px.columns):
                missing_by_h[h] += 1
                continue
            window = px.iloc[start_pos:start_pos+h][top+bottom]
            if window.isna().any().any():
                missing_by_h[h] += 1
                continue
            cumulative = (1.0+window).prod()-1.0  # A total loss (-1) is preserved.
            spreads_by_h[h].append(float(cumulative[top].mean()-cumulative[bottom].mean()))
    out = {'basis': 'gross cohort diagnostic; first earned return after first trading-day close',
           'missing_return_policy': 'exclude entire affected cohort/horizon and disclose count; no clipping or zero fill'}
    for h, vals in spreads_by_h.items():
        arr = np.array(vals, dtype=float)
        out[f'quintile_spread_{h}d_cohorts'] = len(arr)
        out[f'quintile_spread_{h}d_missing_cohorts'] = missing_by_h[h]
        out[f'quintile_spread_{h}d_cum_bps'] = float(arr.mean()*1e4) if len(arr) else None
        out[f'quintile_spread_{h}d_ann'] = float(arr.mean()*TRADING_DAYS/h) if len(arr) else None
        out[f'quintile_spread_{h}d_t'] = (float(arr.mean()/arr.std(ddof=1)*math.sqrt(len(arr)))
            if len(arr)>1 and arr.std(ddof=1)>0 else None)
    return out


def earnings_window_decomposition(
    books: dict[pd.Period, dict[int, float]],
    dsf: pd.DataFrame,
    external_path: Path,
) -> dict[str, object]:
    """Decomposes realized daily portfolio P&L into earnings announcement windows [-1d, +2d] vs ex-earnings days."""
    ext = pd.read_parquet(
        external_path,
        columns=["table", "TICKER", "PERMNO", "sdate", "edate", "SCORE", "anndats", "USFIRM", "MEASURE", "FISCALP"],
    )
    link = ext[ext["table"] == "ibes_link"].dropna(subset=["PERMNO"]).copy()
    link = link[link["SCORE"] <= 2]
    link["permno"] = link["PERMNO"].astype(int)
    link["sdate"] = pd.to_datetime(link["sdate"])
    link["edate"] = pd.to_datetime(link["edate"])

    surp = ext[
        (ext["table"] == "ibes_surprise")
        & (ext["USFIRM"] == 1)
        & (ext["MEASURE"] == "EPS")
        & (ext["FISCALP"] == "QTR")
    ].dropna(subset=["TICKER", "anndats"]).copy()
    surp["anndats"] = pd.to_datetime(surp["anndats"])
    merged = surp[["TICKER", "anndats"]].merge(link[["TICKER", "permno", "sdate", "edate"]], on="TICKER")
    merged = merged[(merged["anndats"] >= merged["sdate"]) & (merged["anndats"] <= merged["edate"])]
    ann_dates = merged[["permno", "anndats"]].drop_duplicates()

    # Ex-post event attribution: four trading observations [-1, +2], not five
    # calendar days. Announcement time-of-day is not a historical signal here.
    px = _daily_pivot(dsf)
    window_pairs: set[tuple[int, pd.Timestamp]] = set()
    for r in ann_dates.itertuples():
        position = px.index.searchsorted(pd.Timestamp(r.anndats).normalize())
        if position >= len(px):
            continue
        for idx in range(max(0, position-1), min(len(px), position+3)):
            window_pairs.add((int(r.permno), px.index[idx]))
    earn_pnl_total = 0.0
    ex_earn_pnl_total = 0.0
    earn_pos_days = 0
    ex_earn_pos_days = 0

    for month in sorted(books):
        w = {int(k): float(v) for k, v in books[month].items() if v != 0.0}
        if not w:
            continue
        days = _held_daily(px, month, w)
        cols = list(w)
        for day, row in days[cols].iterrows():
            r = row
            for p in cols:
                contrib = w[p] * float(r[p])
                if (p, day) in window_pairs:
                    earn_pnl_total += contrib
                    earn_pos_days += 1
                else:
                    ex_earn_pnl_total += contrib
                    ex_earn_pos_days += 1
                w[p] *= 1.0+float(r[p])

    total_pnl = earn_pnl_total + ex_earn_pnl_total
    total_obs = max(1, earn_pos_days + ex_earn_pos_days)
    years = max(len(books) / 12.0, 1e-6)
    return {
        "basis": "ex-post gross security PnL normalized to each month start NAV; excludes cash and execution costs",
        "earnings_window_pnl_ann": float(earn_pnl_total / years),
        "ex_earnings_window_pnl_ann": float(ex_earn_pnl_total / years),
        "earnings_window_share_of_pnl": float(earn_pnl_total / total_pnl) if abs(total_pnl) > 1e-9 else 0.0,
        "ex_earnings_window_share_of_pnl": float(ex_earn_pnl_total / total_pnl) if abs(total_pnl) > 1e-9 else 0.0,
        "earnings_window_stock_day_fraction": float(earn_pos_days / total_obs),
    }
