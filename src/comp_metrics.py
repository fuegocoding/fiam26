"""Competition-brief metric accounting (McGill FIAM 2026 brief, pp. 8-9).

Stated conventions, as the brief requires in the deliverables:

- Capital denominator: $1 of capital at each month-start rebalance.
- Signed weights: long leg +100%, short leg -100% of capital
  (gross 200%, net 0%, both inside the -50%/+50% net band).
- The short proceeds finance the longs; collateral earns the 3-month
  Treasury bill rate (TB3MS).
- ``r_spread`` columns in the port CSVs are built from ret_exc excess
  returns, so they are portfolio EXCESS spreads, not total returns.
- Total return on capital: R_total,t = r_spread,t + TB3MS_t/100/12.
- Benchmark: benchmark_t = TB3MS_t/100/12 + 0.04/12.
- Active return: active_t = R_total,t - benchmark_t
                          = r_spread,t - 0.04/12  (cash rate cancels).
- Information Ratio = sqrt(12) * mean(active_t) / stdev(active_t),
  applied to active returns, never to a regression intercept / residual.
- Sharpe: computed on portfolio returns in excess of TB3MS, which under
  the convention above equals the spread itself (no credit for cash).
- Maximum drawdown compounds R_total on capital.
- Beta/alpha: Newey-West HAC regression (maxlags=3) of the portfolio
  excess return on S&P 500 excess returns (beta-near-zero check).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"

HURDLE_MONTHLY = 0.04 / 12.0


def tb3ms_monthly() -> pd.Series:
    """Monthly TB3MS risk-free rate in decimal units, indexed by month Period."""
    tb = pd.read_csv(EXT / "TB3MS.csv", parse_dates=["observation_date"]).dropna()
    tb["month"] = tb["observation_date"].dt.to_period("M")
    return tb.groupby("month")["TB3MS"].last() / 100.0 / 12.0


def spx_monthly() -> pd.Series:
    """Monthly S&P 500 return in decimal units, indexed by month Period."""
    sp = pd.read_csv(EXT / "SP500.csv", parse_dates=["observation_date"]).dropna()
    sp = sp.rename(columns={"observation_date": "date", "SP500": "sp500"})
    sp = sp.set_index("date").resample("ME").last()
    spx = sp.pct_change().dropna()
    spx.index = spx.index.to_period("M")
    return spx["sp500"]


def load_spread(csv_path: Path, spread_col: str = "ret_ls") -> pd.Series:
    """Load a port CSV as a monthly Period-indexed spread series."""
    port = pd.read_csv(csv_path)
    month_col = "target_month" if "target_month" in port.columns else "month"
    port["month"] = pd.to_datetime(port[month_col]).dt.to_period("M")
    port = port.dropna(subset=[spread_col]).set_index("month")
    return port[spread_col].astype(float)


def comp_metrics(
    spread: pd.Series,
    rf: pd.Series | None = None,
    spx: pd.Series | None = None,
    label: str = "",
) -> dict:
    """Compute brief-compliant metrics from a monthly excess-return spread.

    spread  : monthly excess (ret_exc-based) long-short spread
    rf      : TB3MS monthly decimal series (collateral + benchmark cash leg)
    spx     : S&P 500 monthly return series for the beta check
    """
    s = pd.Series(spread).dropna()
    rf = rf.reindex(s.index) if rf is not None else pd.Series(0.0, index=s.index)
    total = s + rf  # R_total on capital: spread plus collateral at the bill rate
    active = total - (rf + HURDLE_MONTHLY)  # == spread - 0.04/12
    bench_hurdle = rf + HURDLE_MONTHLY

    res = {
        "label": label,
        "months": len(s),
        "first_month": str(s.index.min()),
        "last_month": str(s.index.max()),
        "ann_spread": s.mean() * 12,
        "ann_total_return": total.mean() * 12,
        "ann_active_return": active.mean() * 12,
        "ir": np.sqrt(12) * active.mean() / active.std(ddof=1) if active.std(ddof=1) > 0 else np.nan,
        "sharpe_excess": s.mean() / s.std(ddof=1) * np.sqrt(12) if s.std(ddof=1) > 0 else np.nan,
        "max_drawdown": _max_drawdown(total),
        "worst_month": s.min(),
        "hit_rate": (active > 0).mean(),
    }

    if spx is not None:
        mkt = spx.reindex(s.index) - rf  # market excess return
        reg = pd.DataFrame({"port_rf": s, "mkt_rf": mkt}).dropna()
        if len(reg) > 24:
            x = reg["mkt_rf"].to_numpy()
            y = reg["port_rf"].to_numpy()
            X = np.column_stack([np.ones_like(x), x])
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha, beta = float(coef[0]), float(coef[1])
            resid = y - X @ np.array([alpha, beta])
            lags = 3
            n = len(y)
            XtX_inv = np.linalg.inv(X.T @ X / n)
            S = (X * resid[:, None]).T @ (X * resid[:, None]) / n
            for l in range(1, lags + 1):
                w = 1.0 - l / (lags + 1.0)
                G = (X[l:] * resid[l:, None]).T @ (X[:-l] * resid[:-l, None]) / n
                S += w * (G + G.T)
            cov = XtX_inv @ S @ XtX_inv / n
            res["alpha_monthly"] = alpha
            res["alpha_ann"] = alpha * 12
            res["alpha_tstat"] = alpha / np.sqrt(cov[0, 0])
            res["beta"] = beta
            res["beta_se"] = np.sqrt(cov[1, 1])
            res["corr_spx"] = float(reg["port_rf"].corr(reg["mkt_rf"]))
    return res


def _max_drawdown(ret_series: pd.Series) -> float:
    cum = (1.0 + pd.Series(ret_series).dropna()).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def evaluate_csv(csv_path: Path, rf: pd.Series, spx: pd.Series | None) -> dict:
    spread = load_spread(csv_path)
    return comp_metrics(spread, rf=rf, spx=spx, label=csv_path.stem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Competition-brief metrics from port CSVs")
    ap.add_argument("csvs", nargs="*", help="port_monthly_*.csv files (default: all in output/)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.csvs] or sorted(ROOT.glob("output/port_monthly_*.csv"))
    rf = tb3ms_monthly()
    try:
        spx = spx_monthly()
    except Exception:
        spx = None

    rows = [evaluate_csv(p, rf, spx) for p in paths]
    table = pd.DataFrame(rows)
    cols = [
        "label", "months", "ann_total_return", "ann_active_return", "ir",
        "sharpe_excess", "max_drawdown", "worst_month", "hit_rate",
        "alpha_ann", "alpha_tstat", "beta", "corr_spx",
    ]
    cols = [c for c in cols if c in table.columns]
    print(table[cols].round(4).to_string(index=False))

    out = ROOT / "output" / "comp_metrics_summary.csv"
    table.round(6).to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
