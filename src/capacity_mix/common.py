"""Shared loaders and metrics for the eight-expert capacity study.

Experts are unit books (about 100 long / 100 short, gross about 2.0): seven
event/accounting sleeves whose stock weights were produced walk-forward by
``src.edges.combined_book`` (saved in ``sleeve_weights.parquet``), plus the
submitted TimesFM + XGBoost core, whose 1.0x-gross weights are doubled so it is
comparable with the sleeves.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.capacity_backtest import _load_core_targets
from src.large_mega_hedge import NAME_COST_BP, RAW, _market

SLEEVES = (
    "e7_lm",
    "e7_small_gated",
    "gap_micro_best",
    "runway_micro",
    "conf_lm_gated",
    "inventory_retail",
    "oplev_software",
)
EXPERTS = SLEEVES + ("core",)
NOMICRO = ("e7_lm", "e7_small_gated", "conf_lm_gated", "inventory_retail", "oplev_software")
TEST_START = pd.Period("2021-01", freq="M")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "docs" / "research" / "capacity_mix_20260925"
STATE_PATH = ROOT / "output" / "time_series_capacity_mix" / "_state.pkl"  # matches the git-ignored output/time_series_*/


def load_expert_weights(sleeve_weights: Path) -> pd.DataFrame:
    """Stack all experts' stock weights with the panel fields the engine needs."""
    sleeves = pd.read_parquet(sleeve_weights)
    sleeves["m"] = pd.to_datetime(sleeves["return_month"]).dt.to_period("M")
    core, _ = _load_core_targets()
    core = core[core["target_month"] >= TEST_START].copy()
    core["weight"] = 2.0 * core["target_weight"]
    core = core.rename(columns={"target_month": "m"})[["m", "permno", "weight"]]
    core["sleeve"] = "core"
    weights = pd.concat(
        [sleeves[["m", "sleeve", "permno", "weight"]], core[["m", "sleeve", "permno", "weight"]]],
        ignore_index=True,
    )
    weights["permno"] = weights["permno"].astype(int)

    panel = pq.read_table(
        RAW,
        columns=["permno", "eom", "prc", "me", "size_grp", "ret_exc_lead1m", "dolvol_126d"],
    ).to_pandas()
    panel["eom"] = pd.to_datetime(panel["eom"])
    panel["m"] = panel["eom"].dt.to_period("M") + 1
    panel = panel.drop_duplicates(["permno", "m"])
    panel["size_grp"] = panel["size_grp"].astype(str).str.lower()
    fields = ["permno", "m", "prc", "me", "size_grp", "ret_exc_lead1m", "dolvol_126d"]
    return weights.merge(panel[fields], on=["permno", "m"], how="left")


def expert_returns(weights: pd.DataFrame, months: pd.PeriodIndex) -> dict[str, pd.DataFrame]:
    """Monthly gross, trading-cost and net return of each expert (turnover x bp)."""
    gross, cost = {}, {}
    for expert in EXPERTS:
        rows = weights[weights["sleeve"] == expert]
        previous = pd.Series(dtype=float)
        g, c = [], []
        for month in months:
            x = rows[rows["m"] == month]
            current = x.set_index("permno")["weight"]
            bp = x.set_index("permno")["size_grp"].map(NAME_COST_BP).fillna(25.0)
            g.append(float((x["weight"] * x["ret_exc_lead1m"].fillna(0)).sum()))
            delta = current.sub(previous, fill_value=0).abs()
            c.append(float((delta * bp.reindex(delta.index).fillna(25.0)).sum() / 1e4))
            previous = current
        gross[expert] = pd.Series(g, index=months)
        cost[expert] = pd.Series(c, index=months)
    g_frame, c_frame = pd.DataFrame(gross), pd.DataFrame(cost)
    return {"gross": g_frame, "cost": c_frame, "net": g_frame - c_frame}


def market_frame(months: pd.PeriodIndex) -> pd.DataFrame:
    """S&P 500 return, market excess return and the TB3MS+4% benchmark by month."""
    return _market().reindex(months)


def risk_free(market: pd.DataFrame) -> pd.Series:
    """TB3MS recovered from the cash-plus-4% benchmark."""
    return market["benchmark"] - 0.04 / 12


def perf_stats(returns: pd.Series, market: pd.DataFrame) -> dict[str, float]:
    """Return-level summary: Sharpe is excess of TB3MS, IR is versus cash + 4%."""
    market = market.reindex(returns.index)
    excess = returns - risk_free(market)
    active = returns - market["benchmark"]
    curve = (1 + returns).cumprod()
    return {
        "ann": float(returns.mean() * 12),
        "vol": float(returns.std(ddof=1) * np.sqrt(12)),
        "sharpe": float(excess.mean() / excess.std(ddof=1) * np.sqrt(12)),
        "ir": float(active.mean() / active.std(ddof=1) * np.sqrt(12)),
        "max_dd": float((curve / curve.cummax() - 1).min()),
        "beta": float(np.polyfit(market["mkt_rf"].to_numpy(), returns.to_numpy(), 1)[0]),
    }
