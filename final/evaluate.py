"""Monthly evaluation of a book under the competition's rules.

Returns are on NAV. Each month the book is set at the start of the month at its
target weights; the previous book drifts with its realized returns first, so
turnover is measured against what is actually held. Costs and conventions:

* trading cost per unit traded, by size tier (5 to 60 bp; 25 bp if unknown);
* borrow fee on short notional, by size tier (25 to 1,500 bp a year);
* capital earns the 3-month T-bill (TB3MS / 100 / 12);
* a holding with no next-month return in the panel exits at its last price
  (zero return for that month);
* benchmark = T-bill + 4% a year; active = total - benchmark;
  information ratio = sqrt(12) * mean(active) / stdev(active).

This is the evaluator behind every number in README.md.
"""

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
    u = u.drop(columns=["eom", "gics"])
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


def evaluate(books: dict, universe: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """One row per month: rf, P&L by leg, costs, total, exposures."""
    keyed = universe.set_index(["return_month", "permno"])
    rows, prev, prev_r, prev_size = [], {}, {}, {}
    for m in sorted(books):
        w = books[m]
        if not w:
            continue
        rf = float(mkt.rf.get(m, np.nan))
        ids = list(w)
        cur = keyed.reindex(pd.MultiIndex.from_arrays([[m] * len(ids), ids]))
        ret = pd.Series(cur.ret_exc_lead1m.to_numpy(), index=ids).fillna(0.0)
        size = pd.Series(cur.size_grp.astype("string").to_numpy(), index=ids)
        sector = pd.Series(cur.sector.to_numpy(), index=ids)
        beta = pd.Series(cur.beta_60m.to_numpy(), index=ids)
        prev_total = sum(x * prev_r.get(p, 0.0) for p, x in prev.items())
        drift = {
            p: x * (1 + prev_r.get(p, 0.0) + rf) / (1 + prev_total + rf)
            for p, x in prev.items()
        }
        cost = traded = 0.0
        for p in set(w) | set(drift):
            d = abs(w.get(p, 0.0) - drift.get(p, 0.0))
            if d:
                tier = size.get(p) if p in w else prev_size.get(p)
                cost += d * NAME_COST_BP.get(str(tier), 25.0) / 1e4
                traded += d
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
        prev, prev_r = w, ret.to_dict()
        prev_size.update(size.to_dict())
    out = pd.DataFrame(rows).set_index("month")
    out["spx"] = mkt.spx.reindex(out.index)
    out["benchmark"] = out.rf + HURDLE / 12
    out["active"] = out.total - out.benchmark
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
        "max_dd": float((curve / curve.cummax() - 1).min()),
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
