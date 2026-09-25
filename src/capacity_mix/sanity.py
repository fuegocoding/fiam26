"""Sanity checks on the eight-expert Sharpe: stability, where the P&L comes from, signal lag, selection noise."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

from src.capacity_mix import common
from src.large_mega_hedge import RAW

EULER_GAMMA = 0.5772
MICRO = ("nano", "micro")


def sharpe(returns: pd.Series, market: pd.DataFrame) -> float:
    excess = returns - common.risk_free(market).reindex(returns.index)
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(12))


def yearly(eq: pd.Series, market: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"year": year, "ann_return": g.mean() * 12, "sharpe": sharpe(g, market), "worst_month": g.min(), "months": len(g)}
        for year, g in eq.groupby(eq.index.year)
    ]
    return pd.DataFrame(rows)


def pnl_breakdown(weights: pd.DataFrame, n_experts: int, n_months: int) -> pd.DataFrame:
    """Annualised gross contribution of the equal-weight book by side, price bucket and size bucket."""
    x = weights.copy()
    x["contribution"] = x["weight"] * x["ret_exc_lead1m"].fillna(0) / n_experts
    x["side"] = np.where(x["weight"] > 0, "long", "short")
    x["price"] = np.where(x["prc"] < 5, "under $5", "$5 and above")
    x["size"] = np.where(x["size_grp"].isin(MICRO), "nano+micro", "small and larger")
    rows = [{"cut": "total", "group": "all", "ann_pct": x["contribution"].sum() / n_months * 1200}]
    for cut in ("side", "price", "size"):
        for group, value in x.groupby(cut)["contribution"].sum().items():
            rows.append({"cut": cut, "group": group, "ann_pct": value / n_months * 1200})
    for (side, size), value in x.groupby(["side", "size"])["contribution"].sum().items():
        rows.append({"cut": "side x size", "group": f"{side} / {size}", "ann_pct": value / n_months * 1200})
    return pd.DataFrame(rows)


def lag_placebo(weights: pd.DataFrame, months: pd.PeriodIndex, market: pd.DataFrame, lags=(0, 1, 2, 3)) -> pd.DataFrame:
    """Equal-weight gross return when every expert's holdings are delayed by ``lag`` extra months."""
    panel = pq.read_table(RAW, columns=["permno", "eom", "ret_exc_lead1m"]).to_pandas()
    panel["m"] = pd.to_datetime(panel["eom"]).dt.to_period("M") + 1
    returns = panel.drop_duplicates(["permno", "m"]).set_index(["permno", "m"])["ret_exc_lead1m"]
    rows = []
    for lag in lags:
        per_expert = []
        for expert in common.EXPERTS:
            held = weights[weights["sleeve"] == expert]
            series = []
            for i, month in enumerate(months):
                if i < lag:
                    series.append(np.nan)
                    continue
                book = held[held["m"] == months[i - lag]]
                index = pd.MultiIndex.from_arrays([book["permno"].to_numpy(), [month] * len(book)])
                series.append(float((book["weight"].to_numpy() * returns.reindex(index).fillna(0).to_numpy()).sum()))
            per_expert.append(pd.Series(series, index=months))
        mix = pd.concat(per_expert, axis=1).mean(axis=1).dropna()
        rows.append({"lag_months": lag, "gross_ann_return": mix.mean() * 12, "sharpe": sharpe(mix, market)})
    return pd.DataFrame(rows)


def selection_noise(months: int, trials=(70, 190, 465)) -> pd.DataFrame:
    """Standard error of a Sharpe and the best Sharpe expected from pure noise among ``trials`` tries."""
    se = 1 / np.sqrt(months / 12)
    rows = []
    for n in trials:
        expected_max = se * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e)))
        rows.append({"trials": n, "sharpe_std_error": se, "expected_best_sharpe_from_noise": expected_max})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=common.DEFAULT_DIR)
    parser.add_argument("--state", type=Path, default=common.STATE_PATH)
    args = parser.parse_args()
    with args.state.open("rb") as handle:  # written by routing.py
        state = pickle.load(handle)
    weights, months, net = state["weights"], state["months"], state["net"]
    market = common.market_frame(months)
    equal = net.mean(axis=1)

    corr = net.corr().to_numpy()
    n = len(corr)
    mean_corr = (corr.sum() - n) / (n * (n - 1))
    mean_sharpe = float(np.mean([sharpe(net[k], market) for k in net.columns]))
    diversification = {
        "mean_expert_sharpe": mean_sharpe, "mean_pairwise_correlation": mean_corr,
        "implied_equal_mix_sharpe": mean_sharpe * np.sqrt(n / (1 + (n - 1) * mean_corr)),
        "actual_equal_mix_sharpe": sharpe(equal, market),
    }
    sections = {
        "yearly": yearly(equal, market),
        "pnl_breakdown": pnl_breakdown(weights, n, len(months)),
        "lag_placebo": lag_placebo(weights, months, market),
        "selection_noise": selection_noise(len(months)),
        "diversification": pd.DataFrame([diversification]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.float_format", "{:.3f}".format)
    for name, frame in sections.items():
        frame.to_csv(args.out / f"sanity_{name}.csv", index=False)
        print(f"== {name}\n{frame.to_string(index=False)}\n")


if __name__ == "__main__":
    main()
