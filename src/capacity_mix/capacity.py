"""Stock-level netting of the expert mix and the A-G capacity scenario grid.

Expert stock weights are combined (overlaps net), trimmed to the competition's
500-name limit, and pushed through ``src.capacity_backtest.simulate_capacity_path``.
Borrow fees are stylised tiers by size group, not measured data.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.capacity_backtest import simulate_capacity_path
from src.capacity_constraints import CapacityConfig
from src.capacity_mix import common, routing
from src.large_mega_hedge import NAME_COST_BP, _metrics

FEE_TIER_BP = {"mega": 25, "large": 40, "mid": 100, "small": 300, "micro": 700, "nano": 1500}
UNLIMITED = dict(
    execution_days=100, liquidation_days=100, long_participation=1.0, short_participation=1.0,
    min_short_price=0.0, min_long_price=0.0,
)


def limits(long_part, short_part, days, short_floor) -> dict:
    return dict(
        long_participation=long_part, short_participation=short_part,
        execution_days=days, liquidation_days=days, min_short_price=short_floor,
    )


# scenario -> (CapacityConfig kwargs, borrow fees on?)
SCENARIOS = {
    "U": (UNLIMITED, False),
    "A": (limits(0.05, 0.02, 5, 5.0), False),
    "B": (limits(0.05, 0.02, 5, 1.0), False),
    "C": (limits(0.10, 0.05, 10, 1.0), False),
    "D": (limits(0.20, 0.10, 10, 1.0), False),
    "E": (limits(0.20, 0.10, 10, 1.0), True),
    "F": (limits(0.10, 0.05, 10, 1.0), True),
    "G": (limits(0.05, 0.02, 5, 5.0), True),
}


def cap_names(targets: pd.DataFrame, max_names: int = 500) -> pd.DataFrame:
    """Keep the ``max_names`` largest absolute weights in every month."""
    ranked = targets.assign(_abs=targets["target_weight"].abs())
    ranked = ranked.sort_values(["target_month", "_abs"], ascending=[True, False])
    return ranked.groupby("target_month").head(max_names).drop(columns="_abs")


def combined_targets(weights: pd.DataFrame, expert_weights: pd.DataFrame, max_names: int = 500) -> pd.DataFrame:
    """Net the experts' stock weights under monthly expert weights, then cap names."""
    parts = []
    for expert in common.EXPERTS:
        x = weights.loc[weights["sleeve"] == expert, ["m", "permno", "weight"]].copy()
        x["weight"] = x["weight"] * x["m"].map(expert_weights[expert]).to_numpy()
        parts.append(x)
    netted = pd.concat(parts).groupby(["m", "permno"], as_index=False)["weight"].sum()
    netted = netted[netted["weight"].abs() > 1e-12]
    fields = ["m", "permno", "prc", "me", "size_grp", "ret_exc_lead1m", "dolvol_126d"]
    attributes = weights.drop_duplicates(["m", "permno"])[fields]
    targets = netted.merge(attributes, on=["m", "permno"], how="left")
    targets = targets.rename(columns={"m": "target_month", "weight": "target_weight"})
    targets["size_grp"] = targets["size_grp"].fillna("small")
    targets["borrow_fee_bps"] = targets["size_grp"].map(FEE_TIER_BP).fillna(700)
    return cap_names(targets, max_names)


def run_scenario(targets, nav, scenario, months, market) -> dict:
    kwargs, with_fees = SCENARIOS[scenario]
    frame = targets if with_fees else targets.drop(columns="borrow_fee_bps")
    result = simulate_capacity_path(frame, CapacityConfig(nav=nav, **kwargs), NAME_COST_BP, 0.0)
    monthly = result["monthly"].reindex(months)
    holdings = result["holdings"]
    net = (
        monthly["gross_return"]
        - monthly["transaction_cost_dollars"] / monthly["starting_nav"]
        - monthly["borrow_cost_dollars"] / monthly["starting_nav"]
    )
    summary = _metrics(net, market)
    excess = net - common.risk_free(market)
    curve = (1 + net).cumprod()
    shorts = holdings[holdings["final_dollars"] < 0]
    short_total = -shorts["final_dollars"].sum()
    pct_of_cap = (-shorts["final_dollars"]) / (shorts["me"] * 1e6)
    traded = (
        holdings.groupby("target_month")["executed_trade_dollars"].apply(lambda x: x.abs().sum())
        .reindex(months) / monthly["starting_nav"]
    )

    def side_count(sign):
        mask = holdings["final_dollars"] * sign > 1e-9
        return holdings[mask].groupby("target_month").size().reindex(months).mean()

    return {
        "pnl_m": nav * (curve.iloc[-1] - 1) / 1e6,
        "ann": summary["ann_return"], "cagr": summary["cagr"],
        "vol": net.std(ddof=1) * np.sqrt(12),
        "sharpe": excess.mean() / excess.std(ddof=1) * np.sqrt(12), "ir": summary["ir"],
        "maxdd": summary["max_drawdown"], "worst": net.min(), "hit": (net > 0).mean(),
        "beta": summary["beta"], "beta_t": summary["beta_t"],
        "alpha": summary["alpha_ann"], "alpha_t": summary["alpha_t"],
        "turnover_x": 12 * traded.mean(),
        "tc_pct": 12 * (monthly["transaction_cost_dollars"] / monthly["starting_nav"]).mean() * 100,
        "borrow_pct": 12 * (monthly["borrow_cost_dollars"] / monthly["starting_nav"]).mean() * 100,
        "fill": monthly["dollar_fill_ratio"].mean(),
        "long_pct": (monthly["long_notional"] / monthly["starting_nav"]).mean(),
        "short_pct": (monthly["short_notional"] / monthly["starting_nav"]).mean(),
        "n_long": side_count(1), "n_short": side_count(-1),
        "sub5": (-shorts.loc[shorts["prc"] < 5, "final_dollars"].sum()) / short_total,
        "nano_micro": (-shorts.loc[shorts["size_grp"].isin(["nano", "micro"]), "final_dollars"].sum()) / short_total,
        "mc_med": pct_of_cap.median(), "mc_p90": pct_of_cap.quantile(0.9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=common.DEFAULT_DIR)
    parser.add_argument("--state", type=Path, default=common.STATE_PATH)
    parser.add_argument("--nav", type=float, default=100e6)
    args = parser.parse_args()
    with args.state.open("rb") as handle:  # written by routing.py
        state = pickle.load(handle)
    weights, months, routers = state["weights"], state["months"], state["routers"]
    experts = list(common.EXPERTS)
    market = common.market_frame(months)

    def constant(selected):
        row = routing.equal_weights(experts, selected)
        return pd.DataFrame([row] * len(months), index=months)

    books = {
        "8 equal": constant(None),
        "8 routed": routers[routing.PRIMARY_ROUTER],
        "core + 5 no-micro": constant(list(common.NOMICRO) + ["core"]),
        "7 sleeves": constant([k for k in experts if k != "core"]),
    }
    rows = []
    for name, expert_weights in books.items():
        targets = combined_targets(weights, expert_weights)
        scenarios = list(SCENARIOS) if name == "8 equal" else ["U", "A", "C", "E"]
        for scenario in scenarios:
            rows.append({"book": name, "nav_m": args.nav / 1e6, "sc": scenario,
                         **run_scenario(targets, args.nav, scenario, months, market)})
            print(name, scenario, round(rows[-1]["ann"], 3), flush=True)
        if name == "8 equal":
            for nav in (25e6, 250e6, 500e6):
                for scenario in ("C", "E"):
                    rows.append({"book": name, "nav_m": nav / 1e6, "sc": scenario,
                                 **run_scenario(targets, nav, scenario, months, market)})
    pd.DataFrame(rows).to_csv(args.out / "capacity_grid.csv", index=False)


if __name__ == "__main__":
    main()
