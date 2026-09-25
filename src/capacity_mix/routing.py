"""Fixed mixes, leave-one-out and trailing-performance routers over the experts.

The router is a small soft gate, not a learned neural mixture. Weights for month
``t`` are computed from expert net returns through month ``t-1`` only, with equal
weights until ``lookback`` months of history exist.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.capacity_mix import common

ROUTER_GRID = (
    ("invvol", 12, 1.0), ("invvol", 24, 1.0),
    ("perf", 12, 0.25), ("perf", 12, 0.5), ("perf", 12, 1.0),
    ("perf", 24, 0.25), ("perf", 24, 0.5), ("perf", 24, 1.0),
    ("posgate", 12, 1.0), ("posgate", 24, 1.0),
)
PRIMARY_ROUTER = ("perf", 24, 0.5)  # declared before the grid was run


def equal_weights(experts, selected=None) -> pd.Series:
    """Equal weight over ``selected`` (default all experts)."""
    chosen = list(selected) if selected is not None else list(experts)
    return pd.Series({k: (1 / len(chosen) if k in chosen else 0.0) for k in experts})


def route(net: pd.DataFrame, kind: str, lookback: int, lam: float, tau: float = 1.0) -> pd.DataFrame:
    """Month-by-month expert weights using only returns strictly before each month."""
    experts = list(net.columns)
    base = equal_weights(experts)
    rows = []
    for i in range(len(net)):
        history = net.iloc[max(0, i - lookback):i]
        if len(history) < lookback:
            rows.append(base)
            continue
        mean = history.mean() * 12
        vol = history.std(ddof=1) * np.sqrt(12)
        if kind == "invvol":
            tilt = (1 / vol) / (1 / vol).sum()
            rows.append((1 - lam) * base + lam * tilt)
        elif kind == "perf":
            score = (mean / vol).clip(-3, 3) / tau
            tilt = np.exp(score - score.max())
            rows.append((1 - lam) * base + lam * tilt / tilt.sum())
        elif kind == "posgate":
            gate = (mean > 0).astype(float)
            rows.append(gate / gate.sum() if gate.sum() > 0 else base * 0.0)
        else:
            raise ValueError(f"unknown router kind: {kind}")
    return pd.DataFrame(rows, index=net.index)


def fixed_mixes(experts) -> dict[str, pd.Series]:
    """Named fixed-weight books, including every eight-way leave-one-out."""
    sleeves = [k for k in experts if k != "core"]
    mixes = {
        "core alone": equal_weights(experts, ["core"]),
        "7 sleeves equal": equal_weights(experts, sleeves),
        "8 equal (7 + core)": equal_weights(experts),
        "core + 5 no-micro equal": equal_weights(experts, list(common.NOMICRO) + ["core"]),
        "5 no-micro equal": equal_weights(experts, list(common.NOMICRO)),
    }
    for dropped in experts:
        mixes[f"8 equal minus {dropped}"] = equal_weights(experts, [k for k in experts if k != dropped])
    return mixes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleeve-weights", type=Path, default=common.DEFAULT_DIR / "sleeve_weights.parquet")
    parser.add_argument("--out", type=Path, default=common.DEFAULT_DIR)
    parser.add_argument("--state", type=Path, default=common.STATE_PATH)
    args = parser.parse_args()

    weights = common.load_expert_weights(args.sleeve_weights)
    months = pd.PeriodIndex(sorted(weights["m"].unique()), freq="M")
    experts = list(common.EXPERTS)
    net = common.expert_returns(weights, months)["net"]
    market = common.market_frame(months)
    late = months >= pd.Period("2023-01", freq="M")

    rows = []

    def record(label, series):
        rows.append({"book": label, "window": "2021-01..2026-08", **common.perf_stats(series, market)})
        rows.append({"book": label, "window": "2023-01..2026-08", **common.perf_stats(series[late], market[late])})

    for name, w in fixed_mixes(experts).items():
        record(name, (net * w).sum(axis=1))
    routers = {}
    for kind, lookback, lam in ROUTER_GRID:
        routers[(kind, lookback, lam)] = route(net, kind, lookback, lam)
        record(f"router {kind} L={lookback} lam={lam}", (net * routers[(kind, lookback, lam)]).sum(axis=1))

    args.out.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(args.out / "routing_returnlevel.csv", index=False)
    stats = pd.DataFrame({k: common.perf_stats(net[k], market) for k in experts}).T
    stats.to_csv(args.out / "expert_stats.csv")
    net.corr().to_csv(args.out / "expert_correlations.csv")
    routers[PRIMARY_ROUTER][late].mean().rename("mean_weight_2023on").to_csv(
        args.out / "router_primary_mean_weights.csv"
    )
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.open("wb") as handle:  # consumed by capacity.py; git-ignored output/
        pickle.dump({"weights": weights, "months": months, "net": net, "routers": routers}, handle)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(stats)
    print(report.pivot(index="book", columns="window", values="sharpe"))


if __name__ == "__main__":
    main()
