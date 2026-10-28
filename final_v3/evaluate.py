"""Calculates monthly portfolio returns, costs and performance metrics."""

import pathlib

import numpy as np
import pandas as pd

import official

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


class CausalCapital:
    """Finalize each month's net ledger before sizing the next month."""

    def __init__(self, universe, market, endowment):
        self.universe, self.market = universe, market
        self.endowment = float(endowment)
        self.books, self.starts = {}, {}
        self.current = self.endowment
        self.ledger = None

    def pretrade(self, month):
        """Actual carried positions after prior returns, costs and liquidations."""
        if not self.books:
            return {}
        previous = max(self.books)
        if month != previous+1:
            raise ValueError('Pretrade holdings require consecutive monthly books')
        book = self.books[previous]
        ids = list(book)
        cur = self.universe.loc[self.universe.return_month.eq(previous)].set_index('permno').reindex(ids)
        last = self.ledger.loc[previous]
        returns, liquidated = position_returns(cur, float(last.rf))
        return {p:book[p]*(1+float(returns.loc[p])+float(last.rf))/(1+float(last.total))
                for p in ids if not liquidated.loc[p]}

    def capital(self, month):
        if self.books and month != max(self.books) + 1:
            raise ValueError('Causal capital requires consecutive monthly books')
        if month in self.starts:
            raise ValueError('Causal capital month has already been sized')
        self.starts[month] = self.current
        return self.current

    def settle(self, month, weights):
        if month not in self.starts or month in self.books:
            raise ValueError('Causal settlement requires one previously sized month')
        self.books[month] = dict(weights)
        ledger = evaluate(self.books, self.universe, self.market)
        self.ledger = ledger
        self.current = self.endowment * float((1+ledger.total).prod())
        if not np.isfinite(self.current) or self.current <= 0:
            raise ValueError('Causal portfolio capital is non-positive or invalid')

    @property
    def schedule(self):
        return pd.Series(self.starts, dtype=float).sort_index()


def cash_sweep(weights: pd.Series, accounting_rf: float, treasury_rf: float) -> dict:
    """Sweep unused own capital to bills; retain separate short collateral.

    Short proceeds remain collateral earning the original accounting rate;
    they are not swept a second time. Any borrowing above own NAV is financed
    at that rate. The caller converts stock excess returns back to total P&L.
    """
    if not np.isfinite(accounting_rf) or not np.isfinite(treasury_rf):
        raise ValueError('Missing accounting or point-in-time Treasury rate')
    long = float(weights.clip(lower=0).sum())
    short = float(-weights.clip(upper=0).sum())
    cash = max(1.0 - long, 0.0)
    financing = max(long - 1.0, 0.0)
    return {'treasury_weight': cash, 'treasury_rf': treasury_rf,
            'treasury_income': cash * treasury_rf,
            'short_collateral_weight': short,
            'short_collateral_income': short * accounting_rf,
            'financing_weight': financing,
            'financing_cost': financing * accounting_rf}


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
    """Panel risk-free rate and prior-weighted large-cap market proxy."""
    return official.monthly_market()


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
    rows, exits, cap_rows, prev, prev_r, prev_size = [], [], [], {}, {}, {}
    prev_rf, prev_total = 0.0, 0.0
    for m in sorted(books):
        w = books[m]
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
        cap_cost, cap_traded = {}, {}
        for p in set(w) | set(drift):
            d = abs(w.get(p, 0.0) - drift.get(p, 0.0))
            if d:
                tier = size.get(p) if p in w else prev_size.get(p)
                cost += d * NAME_COST_BP.get(str(tier), 25.0) / 1e4
                traded += d
                tier = str(tier)
                cap_cost[tier] = cap_cost.get(tier, 0.0) + d * NAME_COST_BP.get(tier, 25.0) / 1e4
                cap_traded[tier] = cap_traded.get(tier, 0.0) + d
        # Close the missing-data holding in this month; it is absent from next
        # month's drift, so the exit cannot be charged again one month later.
        for p in liquidated[liquidated].index:
            j = ids.index(p)
            exit_notional = abs(w[p]) * (1 + ret[p] + rf)
            exit_cost = exit_notional * NAME_COST_BP.get(str(size[p]), 25.0) / 1e4
            cost += exit_cost
            traded += exit_notional
            tier = str(size[p])
            cap_cost[tier] = cap_cost.get(tier, 0.0) + exit_cost
            cap_traded[tier] = cap_traded.get(tier, 0.0) + exit_notional
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
        sweep = cash_sweep(ws, rf, float(mkt.treasury_rf.get(m, np.nan)))
        equity_pnl = pnl + float(ws.sum()) * rf
        total = (equity_pnl + sweep['treasury_income']
                 + sweep['short_collateral_income'] - sweep['financing_cost']
                 - cost - borrow)
        month_caps = []
        for tier in sorted(set(NAME_COST_BP) | set(size.astype(str)) | set(cap_cost)):
            selected = size.eq(tier)
            short_weights = ws.where(selected & neg, 0.0)
            tier_borrow = float(-short_weights.sum()) * BORROW_BPS.get(tier, 700.0) / 1e4 / 12
            tier_pnl = float((ws.where(selected, 0.0) * (ret + rf)).sum())
            month_caps.append({'month': m, 'cap_group': tier, 'kind': 'equity',
                'names': int(selected.sum()),
                'long_notional': float(ws.where(selected & ~neg, 0.0).sum()),
                'short_notional': float(-short_weights.sum()),
                'gross': float(ws.where(selected, 0.0).abs().sum()),
                'price_pnl': tier_pnl, 'trading_cost': cap_cost.get(tier, 0.0),
                'borrow_cost': tier_borrow, 'traded': cap_traded.get(tier, 0.0),
                'net_contribution': tier_pnl - cap_cost.get(tier, 0.0) - tier_borrow})
        for label, weight, income in (
            ('Treasury', sweep['treasury_weight'], sweep['treasury_income']),
            ('Short collateral', sweep['short_collateral_weight'], sweep['short_collateral_income']),
            ('Financing', -sweep['financing_weight'], -sweep['financing_cost'])):
            month_caps.append({'month': m, 'cap_group': label, 'kind': 'cash',
                'names': 0, 'long_notional': 0.0, 'short_notional': 0.0,
                'gross': 0.0, 'cash_weight': weight, 'price_pnl': income,
                'trading_cost': 0.0, 'borrow_cost': 0.0, 'traded': 0.0,
                'net_contribution': income})
        if not np.isclose(sum(row['net_contribution'] for row in month_caps), total, atol=1e-12, rtol=0):
            raise ValueError('Market-cap/cash attribution does not reconcile to monthly NAV return')
        cap_rows.extend(month_caps)
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
                "equity_pnl": equity_pnl,
                **sweep,
                "total": total,
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
        prev_total = total
        prev_size.update(size.to_dict())
    out = pd.DataFrame(rows).set_index("month")
    out["spx"] = mkt.spx.reindex(out.index)
    out["benchmark"] = mkt.benchmark_rf.reindex(out.index) + HURDLE / 12
    if out.benchmark.isna().any() or out.spx.isna().any():
        raise ValueError('Incomplete official Treasury or S&P 500 reporting history')
    out["active"] = out.total - out.benchmark
    out.attrs["liquidations"] = pd.DataFrame(exits)
    out.attrs['cap_attribution'] = pd.DataFrame(cap_rows)
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
                "alpha_ann_se": float(se[0] * 12),
                "alpha_t": float(b[0] / se[0]),
                "alpha_t_hac": float(th[0]),
                "beta": float(b[1]),
                "beta_se": float(se[1]),
                "beta_t_hac": float(th[1]),
                "beta_se_hac": float(abs(b[1] / th[1])),
                "corr_large_cap_market_proxy": float(np.corrcoef(d.y, d.mkt)[0, 1]),
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
    parts = []
    for chunk in pd.read_csv(data_dir / DSF_FILE,
                             usecols=['permno', 'date', 'daily_return'],
                             chunksize=250000):
        chunk = chunk.loc[chunk.permno.isin(permnos)].copy()
        chunk['day'] = pd.to_datetime(chunk.date, format='%Y%m%d')
        parts.append(chunk.loc[chunk.day.between(first, last)])
    return pd.concat(parts, ignore_index=True).sort_values(['permno', 'day'])


def daily_metrics(
    books: dict,
    dsf: pd.DataFrame,
    market_daily: pd.DataFrame,
    monthly=None,
    ledger=None,
) -> dict:
    """Daily book returns, maximum one-day loss and daily beta vs the large-cap market proxy.

    Each month's book is held at its target weights and drifts with the
    realized daily returns until the next rebalance, which matches the monthly
    convention in evaluate(). Unused own capital earns the Treasury proxy;
    short collateral and financing retain the panel-implied rate.
    """
    rf_daily = market_daily.rf
    px = dsf.pivot(index="day", columns="permno", values="daily_return")
    total, book_ids = [], {}
    for month in sorted(books):
        w = books[month]
        month_days = market_daily.index[market_daily.index.to_period('M') == month]
        days = px.reindex(month_days)
        if not len(days.index):
            continue
        book_ids[month] = list(w)
        weights = pd.Series(w, dtype=float)
        cols = [c for c in weights.index if c in days.columns]
        for day, row in days[cols].iterrows():
            r = row.fillna(0.0)
            rfd = float(rf_daily.get(day, np.nan))
            if not np.isfinite(rfd):
                raise ValueError(f"Missing official risk-free rate for {day}")
            missing = int((row.isna() & weights.reindex(row.index).abs().gt(1e-10)).sum()) + sum(abs(weights[k])>1e-10 for k in weights.index if k not in cols)
            pnl = float((weights[cols] * (r-rfd)).sum())
            sweep = cash_sweep(weights, rfd, float(market_daily.treasury_rf.get(day, np.nan)))
            equity_pnl = float((weights[cols] * r).sum())
            fee = 0.0 if monthly is None else (float(monthly.loc[month,'cost']) if day==days.index[0] else 0.0)
            borrow = 0.0 if monthly is None else float(monthly.loc[month,'borrow'])/len(days)
            day_return = (equity_pnl + sweep['treasury_income']
                          + sweep['short_collateral_income'] - sweep['financing_cost'] - fee - borrow)
            total.append({"day": day, "month": month, "total": day_return, "rf": rfd,
                          **sweep,
                          'trading_cost':fee,'borrow_cost':borrow,'missing_held_returns':missing})
            denom = 1.0 + day_return
            if denom > 0:
                weights[cols] = weights[cols] * (1.0 + r) / denom
    if not total:
        return {}
    d = pd.DataFrame(total).set_index("day")
    spx = market_daily.spx
    d = d.join(spx.rename("spx"), how="left")
    d['accounting_rf'] = d.rf
    if 'reporting_rf' in market_daily:
        d['rf'] = market_daily.reporting_rf.reindex(d.index)
    d.attrs['expected_sessions'] = market_daily.index
    if ledger is not None:
        ledger.append(d.copy())
    if d.empty:
        return {}
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
    observed = ex.notna() & mkt.notna()
    if observed.sum() > 20:
        b, se = ols(ex[observed].to_numpy(), mkt[observed].to_numpy()[:, None])
        _, th, _ = ols_hac(ex[observed].to_numpy(), mkt[observed].to_numpy()[:, None])
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
