"""Risk caps on top of the capacity limits: per-name NAV, volatility scaling, nano/micro budget, market-cap share.

The capacity engine only limits each name by its own liquidity. These caps bound how much of the book can sit
in volatile, thinly traded names, and show how much of the reported return survives them.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.capacity_mix import capacity, common, routing
from src.large_mega_hedge import RAW

NAV = 100e6
VOL_FLOOR = 0.15  # annualised; avoids dividing by near-zero volatility
NAME_CAP = (0.02, 0.015)  # max long / short weight per name, as a share of NAV
MCAP_SHARE = (0.02, 0.01)  # max long / short position as a share of the company's market cap
MICRO = ("nano", "micro")

# variant -> keyword arguments for ``apply_caps``
VARIANTS = {
    "V0 as reported": {},
    "V1 name cap 2% long/1.5% short": dict(name=NAME_CAP),
    "V2 +vol scaling": dict(name=NAME_CAP, vol=True),
    "V3 + nano/micro <=25% NAV": dict(name=NAME_CAP, vol=True, bucket=0.25),
    "V4 + shorts<=1%/longs<=2% of mcap": dict(name=NAME_CAP, vol=True, bucket=0.25, mcap=MCAP_SHARE),
    "V5 nano/micro <=10% NAV only": dict(bucket=0.10),
}


def attach_volatility(targets: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Add ``sig``, the stock's annualised volatility from ``rvol_21d``, and return its median."""
    panel = pq.read_table(RAW, columns=["permno", "eom", "rvol_21d"]).to_pandas()
    panel["target_month"] = pd.to_datetime(panel["eom"]).dt.to_period("M") + 1
    panel = panel.drop_duplicates(["permno", "target_month"])[["permno", "target_month", "rvol_21d"]]
    frame = targets.merge(panel, on=["permno", "target_month"], how="left")
    annual = frame["rvol_21d"] * np.sqrt(252)
    frame["sig"] = annual.clip(lower=VOL_FLOOR).fillna(annual.median())
    return frame, float(frame["sig"].median())


def apply_caps(
    targets: pd.DataFrame,
    nav: float,
    vol_ref: float = 0.5,
    name: tuple[float, float] | None = None,
    vol: bool = False,
    bucket: float | None = None,
    mcap: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Return ``targets`` with ``target_weight`` reduced by the requested caps (excess stays in cash).

    name:   (long, short) maximum weight per name as a share of NAV.
    vol:    multiply each weight by ``min(1, vol_ref / stock volatility)``; needs a ``sig`` column.
    mcap:   (long, short) maximum position as a share of the company's market cap; needs ``me`` in $ millions.
    bucket: maximum gross nano+micro exposure as a share of NAV, applied per month to that group only.
    """
    frame = targets.copy()
    weight = frame["target_weight"].to_numpy(dtype=float).copy()
    if name is not None:
        weight = np.where(weight > 0, np.minimum(weight, name[0]), np.maximum(weight, -name[1]))
    if vol:
        weight = weight * np.minimum(1.0, vol_ref / frame["sig"].to_numpy(dtype=float))
    if mcap is not None:
        limit = np.where(weight < 0, mcap[1], mcap[0]) * frame["me"].to_numpy(dtype=float) * 1e6 / nav
        limit = np.nan_to_num(limit, nan=np.inf)
        weight = np.sign(weight) * np.minimum(np.abs(weight), limit)
    frame["target_weight"] = weight
    if bucket is not None:
        micro = frame["size_grp"].isin(MICRO).to_numpy()
        gross = frame.assign(_abs=np.abs(weight)).loc[micro].groupby("target_month")["_abs"].sum()
        scale = frame["target_month"].map((bucket / gross).clip(upper=1.0)).fillna(1.0).to_numpy()
        frame["target_weight"] = np.where(micro, frame["target_weight"] * scale, frame["target_weight"])
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=common.DEFAULT_DIR)
    parser.add_argument("--state", type=Path, default=common.STATE_PATH)
    args = parser.parse_args()
    with args.state.open("rb") as handle:  # written by routing.py
        state = pickle.load(handle)
    weights, months = state["weights"], state["months"]
    market = common.market_frame(months)
    experts = list(common.EXPERTS)
    equal = pd.DataFrame([routing.equal_weights(experts)] * len(months), index=months)
    base, vol_ref = attach_volatility(capacity.combined_targets(weights, equal))

    rows = []
    for variant, options in VARIANTS.items():
        capped = apply_caps(base, NAV, vol_ref, **options).drop(columns=["sig", "rvol_21d"])
        by_month = capped.groupby("target_month")["target_weight"]
        micro = capped[capped["size_grp"].isin(MICRO)].groupby("target_month")["target_weight"]
        for scenario in ("U", "A", "C", "E"):
            result = capacity.run_scenario(capped, NAV, scenario, months, market)
            rows.append({
                "variant": variant, "sc": scenario,
                "gross_target": by_month.apply(lambda x: x.abs().sum()).mean(),
                "nm_gross": micro.apply(lambda x: x.abs().sum()).mean(),
                **{k: result[k] for k in (
                    "ann", "sharpe", "ir", "maxdd", "beta", "alpha_t", "fill",
                    "short_pct", "sub5", "nano_micro", "mc_p90", "turnover_x")},
            })
            print(variant, scenario, round(result["ann"], 3), round(result["sharpe"], 2), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "risk_caps.csv", index=False)


if __name__ == "__main__":
    main()
