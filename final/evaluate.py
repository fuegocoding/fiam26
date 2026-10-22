"""Calculates monthly portfolio returns, costs and performance metrics."""

import pathlib

import numpy as np
import pandas as pd

import external

NAME_COST_BP = {"mega": 5.0, "large": 10.0, "small": 20.0, "micro": 40.0, "nano": 60.0}
BORROW_BPS = {
    "mega": 25,
    "large": 40,
    "mid": 100,
    "small": 300,
    "micro": 700,
    "nano": 1500,
}
HURDLE = 0.04


def load_universe(panel_path: pathlib.Path) -> pd.DataFrame:
    """Formation fields and the realized next-month excess return."""
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
    u["return_month"] = pd.to_datetime(u.eom).dt.to_period("M") + 1
    u["sector"] = u.gics.astype("string").str[:2].fillna("na")
    u["size_grp"] = u.size_grp.astype("string")
    quotes = u.loc[u.prc.notna() & u.prc.abs().gt(0), ["permno", "eom", "prc"]].copy()
    quotes["quote_date"] = pd.to_datetime(quotes.pop("eom")).astype("datetime64[ns]")
    quotes["exit_price"] = quotes.pop("prc").abs()
    quotes = quotes.drop_duplicates(["permno", "quote_date"])
    u["liquidation_cutoff"] = u.return_month.dt.to_timestamp("M").astype(
        "datetime64[ns]"
    )
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
    for c in [
        "prc",
        "dolvol_126d",
        "me",
        "ret_exc_lead1m",
        "beta_60m",
        "be_me",
        "ret_12_1",
    ]:
        u[c] = u[c].astype("float64")
    return u


def market() -> pd.DataFrame:
    """Monthly T-bill rate and S&P 500 price return (FRED TB3MS, SP500)."""
    fred = external.read("fred")
    tbill = fred[fred.series == "TB3MS"].set_index("date").value
    tbill.index = tbill.index.to_period("M")
    sp500 = fred[fred.series == "SP500"].dropna(subset=["value"])
    close = sp500.set_index("date").value.resample("ME").last()
    spx = close.pct_change()
    spx.index = spx.index.to_period("M")
    return pd.DataFrame({"rf": tbill / 100 / 12, "spx": spx})


def position_returns(cur: pd.DataFrame, rf: float) -> tuple[pd.Series, pd.Series]:
    """Missing returns trigger a last-quote liquidation, never a zero-label fill."""
    realized = cur.ret_exc_lead1m.astype(float).copy()
    missing = realized.isna()
    if missing.any():
        needed = {"prc", "exit_price", "exit_price_date"}
        if not needed.issubset(cur.columns):
            raise ValueError(
                "Liquidation requires opening price and a dated last quote"
            )
        entry = cur.prc.abs()
        exit_price = cur.exit_price
        invalid = missing & (
            entry.isna() | entry.le(0) | exit_price.isna() | exit_price.le(0)
        )
        if invalid.any():
            raise ValueError("No usable price for missing-return liquidation")
        # The last-quote price change is a total price return, not an excess return.
        realized.loc[missing] = exit_price.loc[missing] / entry.loc[missing] - 1 - rf
    return realized, missing


def evaluate(books: dict, universe: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """One row per month: rf, P&L by leg, costs, total, exposures."""
    keyed = universe.set_index(["return_month", "permno"])
    rows, exits, prev, prev_r, prev_size = [], [], {}, {}, {}
    prev_rf, prev_total = 0.0, 0.0
    for m in sorted(books):
        w = books[m]
        if not w:
            continue
        rf = float(mkt.rf.get(m, np.nan))
        ids = list(w)
        cur = keyed.reindex(pd.MultiIndex.from_arrays([[m] * len(ids), ids]))
        realized, liquidated = position_returns(cur, rf)
        ret = pd.Series(realized.to_numpy(), index=ids)
        liquidated = pd.Series(liquidated.to_numpy(), index=ids)
        size = pd.Series(cur.size_grp.astype("string").to_numpy(), index=ids)
        sector = pd.Series(cur.sector.to_numpy(), index=ids)
        beta = pd.Series(cur.beta_60m.to_numpy(), index=ids)
        drift = {
            p: x * (1 + prev_r.get(p, 0.0) + prev_rf) / (1 + prev_total)
            for p, x in prev.items()
        }
        cost = traded = 0.0
        for p in set(w) | set(drift):
            d = abs(w.get(p, 0.0) - drift.get(p, 0.0))
            if d:
                tier = size.get(p) if p in w else prev_size.get(p)
                cost += d * NAME_COST_BP.get(str(tier), 25.0) / 1e4
                traded += d
        # Close the missing-data holding in this month; it is absent from next
        # month's drift, so the exit cannot be charged again one month later.
        for p in liquidated[liquidated].index:
            j = ids.index(p)
            exit_notional = abs(w[p]) * (1 + ret[p] + rf)
            exit_cost = exit_notional * NAME_COST_BP.get(str(size[p]), 25.0) / 1e4
            cost += exit_cost
            traded += exit_notional
            exits.append(
                {
                    "return_month": m,
                    "permno": p,
                    "opening_weight": w[p],
                    "opening_price": float(cur.prc.iloc[j]),
                    "exit_price": float(cur.exit_price.iloc[j]),
                    "exit_price_date": cur.exit_price_date.iloc[j],
                    "exit_notional": exit_notional,
                    "exit_cost": exit_cost,
                    "price_return": ret[p] + rf,
                    "reason": "missing_return_last_available_quote",
                }
            )
        ws = pd.Series(w)
        neg = ws < 0
        borrow = (
            float(
                (-ws[neg] * size[neg].map(BORROW_BPS).astype(float).fillna(700.0)).sum()
            )
            / 1e4
            / 12
        )
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
                "total": rf + pnl - cost - borrow,
                "traded": traded,
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
                "missing": int(cur.ret_exc_lead1m.isna().sum()),
            }
        )
        prev = {p: x for p, x in w.items() if not liquidated[p]}
        prev_r = ret.to_dict()
        prev_rf = rf
        prev_total = rf + pnl - cost - borrow
        prev_size.update(size.to_dict())
    out = pd.DataFrame(rows).set_index("month")
    out["spx"] = mkt.spx.reindex(out.index)
    out["benchmark"] = out.rf + HURDLE / 12
    out["active"] = out.total - out.benchmark
    out.attrs["liquidations"] = pd.DataFrame(exits)
    return out


def ols_hac(y: np.ndarray, x: np.ndarray, lags: int = 3):
    """OLS with an intercept; Newey-West (HAC) t-statistics and R2."""
    x = np.column_stack([np.ones(len(y)), x])
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ b
    xtxi = np.linalg.inv(x.T @ x)
    s = (x * e[:, None]).T @ (x * e[:, None])
    for lag in range(1, lags + 1):
        g = (x[lag:] * e[lag:, None]).T @ (x[:-lag] * e[:-lag, None])
        s += (1 - lag / (lags + 1)) * (g + g.T)
    t = b / np.sqrt(np.diag(xtxi @ s @ xtxi))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, t, r2


def ols(y: np.ndarray, x: np.ndarray):
    """Plain OLS with an intercept: coefficients and classical standard errors."""
    x = np.column_stack([np.ones(len(y)), x])
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ b
    s2 = (e @ e) / (len(y) - x.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(x.T @ x)))
    return b, se


def factors(universe: pd.DataFrame) -> pd.DataFrame:
    """Size, value and momentum long-short factors built from the panel."""
    p = universe.dropna(subset=["ret_exc_lead1m"])

    def spread(col):
        r = p.groupby("return_month")[col].rank(pct=True)
        hi = p[r >= 0.7].groupby("return_month").ret_exc_lead1m.mean()
        lo = p[r <= 0.3].groupby("return_month").ret_exc_lead1m.mean()
        return hi - lo

    g = p.groupby(["return_month", "size_grp"]).ret_exc_lead1m.mean().unstack()
    return pd.DataFrame(
        {
            "smb": g[["micro", "small"]].mean(axis=1)
            - g[["large", "mega"]].mean(axis=1),
            "hml": spread("be_me"),
            "mom": spread("ret_12_1"),
        }
    )


def metrics(r: pd.DataFrame, factor_returns: pd.DataFrame | None = None) -> dict:
    """Headline statistics of an evaluate() frame."""
    a, ex = r.active, r.total - r.rf
    curve = (1 + r.total).cumprod()
    out = {
        "months": len(r),
        "ann_total": float((1 + r.total).prod() ** (12 / len(r)) - 1),
        "ann_active": float(a.mean() * 12),
        "ir": float(np.sqrt(12) * a.mean() / a.std()),
        "sharpe": float(np.sqrt(12) * ex.mean() / ex.std()),
        "max_dd": float((curve / curve.cummax().clip(lower=1.0) - 1).min()),
        "turnover": float(r.traded.mean()),
        "cost_ann": float(r.cost.mean() * 12),
        "borrow_ann": float(r.borrow.mean() * 12),
    }
    d = pd.DataFrame({"y": ex, "mkt": r.spx - r.rf}).dropna()
    if len(d) > 12:
        b, se = ols(d.y.to_numpy(), d[["mkt"]].to_numpy())
        _, th, _ = ols_hac(d.y.to_numpy(), d[["mkt"]].to_numpy())
        out.update(
            {
                "alpha_ann": float(b[0] * 12),
                "alpha_t": float(b[0] / se[0]),
                "alpha_t_hac": float(th[0]),
                "beta": float(b[1]),
                "beta_se": float(se[1]),
                "beta_t_hac": float(th[1]),
                "beta_se_hac": float(abs(b[1] / th[1])),
                "corr_spx": float(np.corrcoef(d.y, d.mkt)[0, 1]),
            }
        )
        if factor_returns is not None:
            f = d.join(factor_returns).dropna()
            b4, t4, _ = ols_hac(
                f.y.to_numpy(), f[["mkt", "smb", "hml", "mom"]].to_numpy()
            )
            out.update(
                {
                    "alpha4_ann": float(b4[0] * 12),
                    "alpha4_t_hac": float(t4[0]),
                    "load_mkt": float(b4[1]),
                    "load_smb": float(b4[2]),
                    "load_hml": float(b4[3]),
                    "load_mom": float(b4[4]),
                }
            )
    return out


# --- Daily frequency (Instructions 9.1 and 9.4) -----------------------------
#
# The mandate supplies daily returns and asks for daily beta and the maximum
# one-day loss. The book is set monthly, so the daily series is built by
# carrying each month's book across its trading days and letting the weights
# drift exactly as evaluate() does from month to month.

DSF_FILE = "dsf_mini_from_2014.csv"
TRADING_DAYS = 252


def load_dsf(
    data_dir: pathlib.Path, permnos: set[int], first: pd.Timestamp, last: pd.Timestamp
) -> pd.DataFrame:
    """Daily returns for `permnos` between `first` and `last`, memory-lean."""
    import pyarrow.csv as pv

    path = data_dir / DSF_FILE
    table = pv.read_csv(
        path,
        convert_options=pv.ConvertOptions(
            column_types={"permno": "int64", "date": "int64"}
        ),
    )
    d = table.to_pandas()
    d["day"] = pd.to_datetime(d.date, format="%Y%m%d")
    d = d.loc[d.permno.isin(permnos) & d.day.between(first, last)]
    return d.sort_values(["permno", "day"]).reset_index(drop=True)


def daily_spx(fred: pd.DataFrame) -> pd.Series:
    """Daily S&P 500 total-of-price returns from the FRED SP500 series."""
    s = fred[fred.series == "SP500"].dropna(subset=["value"]).copy()
    s["day"] = pd.to_datetime(s.date)
    close = s.set_index("day").value.sort_index()
    return close.pct_change().dropna()


def daily_metrics(
    books: dict,
    dsf: pd.DataFrame,
    fred: pd.DataFrame,
) -> dict:
    """Daily book returns, maximum one-day loss and daily beta vs the S&P 500.

    Each month's book is held at its target weights and drifts with the
    realized daily returns until the next rebalance, which matches the monthly
    convention in evaluate(). Cash earns the daily T-bill.
    """
    tbill = fred[fred.series == "TB3MS"].dropna(subset=["value"]).copy()
    tbill["day"] = pd.to_datetime(tbill.date)
    rf_daily = (tbill.set_index("day").value / 100 / 252).sort_index()
    px = dsf.pivot(index="day", columns="permno", values="daily_return")
    total, book_ids = [], {}
    for month in sorted(books):
        w = books[month]
        if not w:
            continue
        days = px.loc[px.index.to_period("M") == month]
        if days.empty:
            continue
        book_ids[month] = list(w)
        weights = pd.Series(w, dtype=float)
        cols = [c for c in weights.index if c in days.columns]
        for day, row in days[cols].iterrows():
            r = row.fillna(0.0)
            rfd = float(rf_daily.get(day, np.nan))
            if not np.isfinite(rfd):
                rfd = float(rf_daily.asof(day))
            pnl = float((weights[cols] * r).sum())
            total.append({"day": day, "month": month, "total": rfd + pnl, "rf": rfd})
            denom = 1.0 + pnl + rfd
            if denom > 0:
                weights[cols] = weights[cols] * (1.0 + r + rfd) / denom
    if not total:
        return {}
    d = pd.DataFrame(total).set_index("day")
    spx = daily_spx(fred)
    d = d.join(spx.rename("spx"), how="inner").dropna(subset=["spx"])
    ex = d.total - d.rf
    mkt = d.spx - d.rf
    worst = d.total.idxmin()
    out = {
        "days": int(len(d)),
        "max_one_day_loss": float(d.total.min()),
        "max_one_day_loss_date": str(pd.Timestamp(worst).date()),
        "best_one_day_return": float(d.total.max()),
        "ann_volatility_daily": float(d.total.std() * np.sqrt(TRADING_DAYS)),
    }
    if len(d) > 20:
        b, se = ols(ex.to_numpy(), mkt.to_numpy()[:, None])
        _, th, _ = ols_hac(ex.to_numpy(), mkt.to_numpy()[:, None])
        out.update(
            {
                "beta_daily": float(b[1]),
                "beta_daily_se": float(se[1]),
                "beta_daily_t_hac": float(th[1]),
                "alpha_daily_ann": float(b[0] * TRADING_DAYS),
                "alpha_daily_t": float(b[0] / se[0]),
            }
        )
    return out


def capital_at(nav: float | pd.Series, months: pd.Series) -> pd.Series:
    """Capital at each month's formation date.

    A scalar is the endowed amount. A Series indexed by holding month is the
    compounding path the mandate requires (Instructions 3.1, 3.3): month t is
    sized on the endowment grown by every prior month's net return.
    """
    if not isinstance(nav, pd.Series):
        return pd.Series(float(nav), index=months.index)
    return pd.Series(months.map(nav).astype(float).to_numpy(), index=months.index)


def adv_days_exhibit(p: pd.DataFrame, nav: float | pd.Series) -> pd.DataFrame:
    """Position size in days of average daily volume, long and short apart.

    `dolvol_126d` is the 126-day average of daily dollar volume, so a position
    of D dollars is D / dolvol_126d days of volume (Instructions 9.4). Dollars
    come from the capital the month was sized on, which compounds over the
    out-of-sample period.
    """
    q = p[["return_month", "permno", "side", "target_weight", "dolvol_126d"]].copy()
    q["position_dollars"] = q.target_weight.abs() * capital_at(nav, q.return_month)
    q["adv_days"] = q.position_dollars / q.dolvol_126d.replace(0, np.nan)
    return q.dropna(subset=["adv_days"])
