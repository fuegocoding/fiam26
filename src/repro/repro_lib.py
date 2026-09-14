"""Shared utilities for reproducing docs/timesfm_strategy_memo.md.

All scripts in src/repro/ import from here so every memo stage is rebuilt from the
same primitives:

- Forecasts : output/preds_timesfm_all.parquet (TimesFM 3.0 zero-shot, target
              months 2016-02 .. 2026-08; committed artifact, no model run needed)
- Market    : data/external/SP500.csv + TB3MS.csv (FRED, cached via xgb_baseline)
- Book rules: top-100/bottom-100 by forecast, equal weight, monthly rebalance,
              with the rank buffer (hysteresis) described in memo section 3
- Sizing    : drawdown-responsive ratchet described in memo section 5.4

The ratchet de-gross mapping is "exposure = max(0.3, 1 - 2x drawdown)" while
in >10% drawdown; that is the reading that matches every memo figure exactly.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
EXT = ROOT / "data" / "external"
N_SIDE = 100

# ---------------------------------------------------------------------------
# Metrics (same definitions as src/xgb_baseline.py)
# ---------------------------------------------------------------------------

def max_drawdown(ret_series):
    cum = (1.0 + ret_series).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def perf_stats(ret_series):
    ret = pd.Series(ret_series).dropna()
    return {
        "months": len(ret),
        "ann_ret": ret.mean() * 12,
        "sharpe": ret.mean() / ret.std(ddof=1) * np.sqrt(12) if ret.std(ddof=1) > 0 else np.nan,
        "maxdd": max_drawdown(ret),
    }


def yearly_table(ret_series, months_index):
    """Per-calendar-year ann. return and Sharpe from a monthly return series."""
    df = pd.DataFrame({"ret": np.asarray(ret_series)}, index=pd.PeriodIndex(months_index, freq="M"))
    rows = []
    for yr, g in df.groupby(df.index.year):
        rows.append({
            "year": yr,
            "ann_ret": g["ret"].mean() * 12,
            "sharpe": g["ret"].mean() / g["ret"].std(ddof=1) * np.sqrt(12)
            if len(g) > 1 and g["ret"].std(ddof=1) > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def monthly_market():
    """Monthly SPX return, TB3MS RF, SPX excess, benchmark (TB3MS+4%).

    Returns a DataFrame indexed by month Period with columns:
    spx_ret, rf_dec, mkt_rf, benchmark. First SPX return month = 2016-10
    (FRED SP500 series starts 2016-09-12).
    """
    sp = pd.read_csv(EXT / "SP500.csv", parse_dates=["observation_date"])
    sp = sp.rename(columns={"observation_date": "date", "SP500": "sp500"}).dropna()
    sp = sp.set_index("date").resample("ME").last()
    spx = sp.pct_change().dropna()
    spx.index = spx.index.to_period("M")
    spx = spx.rename(columns={"sp500": "spx_ret"})

    tb = pd.read_csv(EXT / "TB3MS.csv", parse_dates=["observation_date"])
    tb = tb.rename(columns={"observation_date": "date", "TB3MS": "tb3ms"}).dropna()
    tb["month"] = tb["date"].dt.to_period("M")
    tb = tb.groupby("month")["tb3ms"].last().to_frame()
    tb["rf_dec"] = tb["tb3ms"] / 100.0 / 12.0
    tb["benchmark"] = tb["rf_dec"] + 0.04 / 12.0

    mkt = spx.join(tb, how="inner")
    mkt["mkt_rf"] = mkt["spx_ret"] - mkt["rf_dec"]
    return mkt


# ---------------------------------------------------------------------------
# Book construction
# ---------------------------------------------------------------------------

def load_preds():
    preds = pd.read_parquet(OUT / "preds_timesfm_all.parquet")
    preds["eom"] = pd.to_datetime(preds["eom"])
    preds["target_month"] = pd.to_datetime(preds["target_month"])
    return preds.dropna(subset=["tfm_forecast"]).reset_index(drop=True)


def monthly_sets(preds, buffer_k=None):
    """Long/short name sets per target month.

    buffer_k=None : plain top-100 / bottom-100 each month (memo section 2).
    buffer_k=k    : rank buffer per memo section 3 - a name enters a leg only
                    when its forecast rank reaches the top/bottom-100, is
                    retained until its rank falls beyond the k-wide band, and
                    vacant slots are filled with the best-ranked non-holders.
    """
    long_sets, short_sets = {}, {}
    prev_long, prev_short = [], []
    for month, g in preds.groupby("target_month"):
        order = g.sort_values("tfm_forecast", ascending=False)
        names = order["permno"].tolist()
        n = len(names)
        pos = {p: i for i, p in enumerate(names)}  # 0-based, 0 = best forecast

        if buffer_k is None:
            long_sets[month] = set(names[:N_SIDE])
            short_sets[month] = set(names[n - N_SIDE:])
        else:
            long_keep = [p for p in prev_long if p in pos and pos[p] < buffer_k]
            short_keep = [p for p in prev_short if p in pos and n - 1 - pos[p] < buffer_k]
            long_keep = sorted(long_keep, key=lambda p: pos[p])[:N_SIDE]
            short_keep = sorted(short_keep, key=lambda p: pos[p])[-N_SIDE:]
            fills_long = [p for p in names if p not in long_keep]
            fills_short = [p for p in reversed(names) if p not in short_keep]
            long_sets[month] = set(long_keep + fills_long[: N_SIDE - len(long_keep)])
            short_sets[month] = set(short_keep + fills_short[: N_SIDE - len(short_keep)])
        prev_long = sorted(long_sets[month], key=lambda p: pos[p])
        prev_short = sorted(short_sets[month], key=lambda p: pos[p])

    return long_sets, short_sets


def book_returns(preds, long_sets, short_sets):
    """Equal-weight leg returns per month from the chosen sets."""
    ret_by_month = {}
    for month, g in preds.groupby("target_month"):
        ret_by_month[month] = g.set_index("permno")["ret_exc_lead1m"]
    rows = []
    for month in sorted(long_sets):
        r = ret_by_month[month]
        long_leg = r.reindex(sorted(long_sets[month]))
        short_leg = r.reindex(sorted(short_sets[month]))
        rows.append({
            "target_month": month,
            "n_long": long_leg.notna().sum(),
            "n_short": short_leg.notna().sum(),
            "ret_long": float(long_leg.mean()),
            "ret_short": float(short_leg.mean()),
        })
    port = pd.DataFrame(rows)
    port["ret_ls"] = port["ret_long"] - port["ret_short"]
    return port


def staggered_sets(preds, T):
    """Staggered-tranche book: 100 names per leg total, split into T tranches;
    tranche c refreshes from raw ranks every T-th month, so each name is held
    ~T months and only ~1/T of the book trades per month. Position count and
    the 100-name-per-leg structure are unchanged."""
    months = sorted(preds["target_month"].unique())
    state = {c: (set(), set()) for c in range(T)}
    long_sets, short_sets = {}, {}
    per = N_SIDE // T
    extra = N_SIDE - per * T
    for i, month in enumerate(months):
        g = preds[preds["target_month"] == month].sort_values("tfm_forecast", ascending=False)
        names = list(g["permno"])
        n = len(names)
        c = i % T
        k_leg = per + (1 if c < extra else 0)
        state[c] = (set(names[:k_leg]), set(names[n - k_leg:]))
        long_sets[month] = set().union(*[state[cc][0] for cc in range(T)])
        short_sets[month] = set().union(*[state[cc][1] for cc in range(T)])
    return long_sets, short_sets


def turnover_per_leg(long_sets, short_sets):
    def rate(sets):
        months = sorted(sets)
        vals = []
        for prev, cur in zip(months, months[1:]):
            if len(sets[cur]):
                vals.append(len(sets[cur] - sets[prev]) / len(sets[cur]))
        return float(np.mean(vals)) if vals else np.nan
    return rate(long_sets), rate(short_sets)


# ---------------------------------------------------------------------------
# Ratchet sizing
# ---------------------------------------------------------------------------

def ratchet_scales(underlying_ret, start=0.5, step=0.1, floor=0.3, dd_gate=0.10,
                   depth_mult=2.0):
    """Exposure schedule from the memo section 5.4 rule.

    Scale for month t uses equity through t-1 only. underlying_ret is the
    UNSCALED strategy return series (indexed by month Period).

    depth_mult controls how hard drawdowns de-gross: while drawdown > dd_gate,
    exposure = max(floor, 1 - depth_mult * drawdown). depth_mult=2.0 is the
    reading that reproduces every memo SS5.4/SS8 number exactly ("scale down
    proportional to drawdown depth" = exposure 1 - 2x drawdown, floor 0.3x).
    """
    ret = pd.Series(underlying_ret).dropna()
    scales, equity, peak = [], 1.0, 1.0
    scale = start - step  # the first month runs at `start`; step-ups begin after
    for m, r in ret.items():
        if equity >= peak:  # new high as of t-1
            scale = min(1.0, scale + step)
        else:
            dd = 1.0 - equity / peak
            if dd > dd_gate:
                scale = max(floor, 1.0 - depth_mult * dd)
        scales.append(scale)
        equity *= 1.0 + scale * r
        peak = max(peak, equity)
    return pd.Series(scales, index=ret.index, name="scale")
