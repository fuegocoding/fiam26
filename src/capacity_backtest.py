"""Backtest the frozen core after stock-specific dollar capacity constraints."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow.parquet as pq

from src.capacity_constraints import CapacityConfig, apply_capacity_constraints
from src.large_mega_hedge import NAME_COST_BP, OUT, RAW, ROOT, _core, _market, _metrics


DEFAULT_OUTPUT = OUT / "stock_capacity_backtest_20260924"
BACKTEST_COLUMNS = {
    "target_month",
    "permno",
    "target_weight",
    "prc",
    "dolvol_126d",
    "ret_exc_lead1m",
    "size_grp",
}


def _exit_row(
    permno: int,
    month: pd.Period,
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    row = dict(snapshot)
    row.update(
        {
            "target_month": month,
            "permno": permno,
            "target_weight": 0.0,
            # The name is sold at the rebalance, so it earns no next-month P&L.
            "ret_exc_lead1m": 0.0,
        }
    )
    return row


def simulate_capacity_path(
    targets: pd.DataFrame,
    config: CapacityConfig,
    transaction_cost_bp: Mapping[str, float],
    fallback_borrow_fee_bps: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Carry executable holdings through a chronological monthly backtest.

    ``ret_exc_lead1m`` is applied to the post-trade dollar position. Positions
    drift by that same return into the following rebalance, so only the required
    dollar change consumes the next month's trading capacity.
    """
    missing = BACKTEST_COLUMNS.difference(targets.columns)
    if missing:
        raise ValueError(f"targets is missing required columns: {sorted(missing)}")
    if fallback_borrow_fee_bps < 0:
        raise ValueError("fallback_borrow_fee_bps cannot be negative")

    frame = targets.copy()
    frame["target_month"] = pd.PeriodIndex(frame["target_month"], freq="M")
    frame = frame.sort_values(["target_month", "permno"])
    if frame.duplicated(["target_month", "permno"]).any():
        raise ValueError("targets must have at most one row per month and permno")

    nav = float(config.nav)
    previous_dollars: dict[int, float] = {}
    last_snapshot: dict[int, dict[str, object]] = {}
    monthly_rows: list[dict[str, object]] = []
    holding_frames: list[pd.DataFrame] = []

    for month, current in frame.groupby("target_month", sort=True):
        month = pd.Period(month, freq="M")
        current = current.copy()
        current_ids = set(current["permno"].astype(int))
        exits = [
            _exit_row(permno, month, last_snapshot[permno])
            for permno in previous_dollars
            if permno not in current_ids
        ]
        if exits:
            current = pd.concat([current, pd.DataFrame(exits)], ignore_index=True)

        month_config = replace(config, nav=nav)
        constrained = apply_capacity_constraints(
            current, previous_dollars, month_config
        )
        constrained["target_month"] = month

        tier = constrained["size_grp"].astype(str).str.lower()
        cost_bp = tier.map(transaction_cost_bp).fillna(
            float(transaction_cost_bp.get("unknown", 0.0))
        )
        transaction_cost = float(
            (constrained["executed_trade_dollars"].abs() * cost_bp / 1e4).sum()
        )

        if config.borrow_fee_col in constrained:
            fee_bp = pd.to_numeric(
                constrained[config.borrow_fee_col], errors="coerce"
            ).fillna(fallback_borrow_fee_bps)
        else:
            fee_bp = pd.Series(
                fallback_borrow_fee_bps, index=constrained.index, dtype=float
            )
        short_dollars = constrained["final_dollars"].clip(upper=0).abs()
        borrow_cost = float((short_dollars * fee_bp / 1e4 / 12.0).sum())

        realized = pd.to_numeric(
            constrained["ret_exc_lead1m"], errors="coerce"
        ).fillna(0.0)
        pnl = float((constrained["final_dollars"] * realized).sum())
        ending_nav = nav + pnl - transaction_cost - borrow_cost
        if ending_nav <= 0:
            raise ValueError(f"portfolio NAV became non-positive in {month}")

        requested_gross = float(constrained["requested_dollars"].abs().sum())
        final_gross = float(constrained["final_dollars"].abs().sum())
        monthly_rows.append(
            {
                "target_month": month,
                "starting_nav": nav,
                "ending_nav": ending_nav,
                "pnl_dollars": pnl,
                "transaction_cost_dollars": transaction_cost,
                "borrow_cost_dollars": borrow_cost,
                "gross_return": pnl / nav,
                "net_return": (pnl - transaction_cost - borrow_cost) / nav,
                "requested_gross": requested_gross / nav,
                "executed_gross": final_gross / nav,
                "long_notional": float(
                    constrained["final_dollars"].clip(lower=0).sum()
                ),
                "short_notional": float(
                    constrained["final_dollars"].clip(upper=0).abs().sum()
                ),
                "dollar_fill_ratio": (
                    final_gross / requested_gross if requested_gross > 0 else 1.0
                ),
                "names": int((constrained["final_dollars"].abs() > 1e-9).sum()),
                "proxy_short_names": int(
                    (
                        (constrained["final_dollars"] < 0)
                        & (constrained["borrow_status"] == "proxy_unverified")
                    ).sum()
                ),
            }
        )
        holding_frames.append(constrained)

        previous_dollars = {
            int(permno): float(dollars * (1.0 + stock_return))
            for permno, dollars, stock_return in zip(
                constrained["permno"], constrained["final_dollars"], realized
            )
            if abs(dollars * (1.0 + stock_return)) > 1e-9
        }
        for _, row in current.iterrows():
            last_snapshot[int(row["permno"])] = row.to_dict()
        nav = ending_nav

    monthly = pd.DataFrame(monthly_rows).set_index("target_month")
    holdings = pd.concat(holding_frames, ignore_index=True)
    return {"monthly": monthly, "holdings": holdings}


def _load_core_targets() -> tuple[pd.DataFrame, pd.Series]:
    columns = [
        "permno",
        "eom",
        "prc",
        "me",
        "size_grp",
        "gics",
        "ret_exc_lead1m",
        "o_score",
        "chcsho_12m",
        "eqnpo_12m",
        "dolvol_126d",
    ]
    panel = pq.read_table(RAW, columns=columns).to_pandas()
    panel["eom"] = pd.to_datetime(panel["eom"])
    panel["target_month"] = panel["eom"].dt.to_period("M") + 1
    panel["size_grp"] = panel["size_grp"].astype(str).str.lower()
    panel["sector"] = panel["gics"].astype("string").str[:2].fillna("UNK")
    months, _, _, long_weights, short_weights = _core(panel)

    rows: list[pd.DataFrame] = []
    reconstructed: dict[pd.Period, float] = {}
    panel_by_month = {
        month: group.drop_duplicates("permno")
        for month, group in panel[panel["target_month"].isin(months)].groupby(
            "target_month", sort=True
        )
    }
    for month in months:
        signed: dict[int, float] = {}
        for permno, weight in long_weights[month].items():
            signed[int(permno)] = signed.get(int(permno), 0.0) + float(weight)
        for permno, weight in short_weights[month].items():
            signed[int(permno)] = signed.get(int(permno), 0.0) - float(weight)
        selected = panel_by_month[month]
        selected = selected[selected["permno"].isin(signed)].copy()
        selected["target_weight"] = selected["permno"].map(signed)
        rows.append(selected)
        reconstructed[month] = float(
            (
                selected["target_weight"]
                * selected["ret_exc_lead1m"].fillna(0.0)
            ).sum()
        )
    return pd.concat(rows, ignore_index=True), pd.Series(reconstructed).sort_index()


def run(
    output_dir: Path = DEFAULT_OUTPUT,
    navs: tuple[float, ...] = (10_000_000.0, 50_000_000.0, 100_000_000.0),
    borrow_fee_scenarios_bps: tuple[float, ...] = (0.0, 1_000.0),
) -> dict[str, pd.DataFrame]:
    """Run NAV and borrow-cost scenarios and write auditable artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    targets, reconstructed = _load_core_targets()
    months = reconstructed.index
    market = _market().reindex(months)
    frozen = pd.read_csv(OUT / "repro" / "final_config_monthly.csv", index_col=0)
    frozen.index = pd.PeriodIndex(frozen.index, freq="M")
    frozen_gross = 0.25 * (frozen["ls_tfm_buffered"] + frozen["ls_xgb_raw"])
    frozen_gross = frozen_gross.reindex(months)

    summaries: list[dict[str, object]] = []
    monthly_frames: list[pd.DataFrame] = []
    holding_frames: list[pd.DataFrame] = []
    for starting_nav in navs:
        for borrow_fee_bps in borrow_fee_scenarios_bps:
            label = f"nav_{starting_nav / 1e6:g}m_borrow_{borrow_fee_bps / 100:g}pct"
            config = CapacityConfig(nav=starting_nav)
            result = simulate_capacity_path(
                targets,
                config,
                transaction_cost_bp=NAME_COST_BP,
                fallback_borrow_fee_bps=borrow_fee_bps,
            )
            monthly = result["monthly"].reindex(months).copy()
            monthly["anchored_gross"] = frozen_gross + (
                monthly["gross_return"] - reconstructed
            )
            monthly["transaction_cost_return"] = (
                monthly["transaction_cost_dollars"] / monthly["starting_nav"]
            )
            monthly["borrow_cost_return"] = (
                monthly["borrow_cost_dollars"] / monthly["starting_nav"]
            )
            monthly["anchored_net"] = (
                monthly["anchored_gross"]
                - monthly["transaction_cost_return"]
                - monthly["borrow_cost_return"]
            )
            monthly.index.name = "target_month"
            monthly.insert(0, "scenario", label)
            monthly_frames.append(monthly.reset_index())

            holdings = result["holdings"].copy()
            holdings.insert(0, "scenario", label)
            holding_frames.append(holdings)
            for window, mask in (
                ("full", monthly.index >= pd.Period("2018-01")),
                ("oos", monthly.index >= pd.Period("2021-01")),
            ):
                values = monthly.loc[mask, "anchored_net"]
                summaries.append(
                    {
                        "scenario": label,
                        "starting_nav": starting_nav,
                        "borrow_fee_bps": borrow_fee_bps,
                        "window": window,
                        **_metrics(values, market),
                        "ending_nav_path": float(
                            starting_nav * (1.0 + values).prod()
                        ),
                        "total_pnl_path": float(
                            starting_nav * ((1.0 + values).prod() - 1.0)
                        ),
                        "avg_executed_gross": float(
                            monthly.loc[mask, "executed_gross"].mean()
                        ),
                        "avg_fill_ratio": float(
                            monthly.loc[mask, "dollar_fill_ratio"].mean()
                        ),
                        "avg_monthly_transaction_cost": float(
                            monthly.loc[mask, "transaction_cost_return"].mean()
                        ),
                        "avg_monthly_borrow_cost": float(
                            monthly.loc[mask, "borrow_cost_return"].mean()
                        ),
                    }
                )

    summary = pd.DataFrame(summaries)
    monthly_output = pd.concat(monthly_frames, ignore_index=True)
    holdings_output = pd.concat(holding_frames, ignore_index=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    monthly_output.to_csv(output_dir / "monthly.csv", index=False)
    holdings_output.to_parquet(output_dir / "holdings.parquet", index=False)
    checks = {
        "months": len(months),
        "first_month": str(months[0]),
        "last_month": str(months[-1]),
        "shorts_are_proxy_only": True,
        "anchor": "frozen core plus name-level capacity return delta",
    }
    (output_dir / "run_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    return {"summary": summary, "monthly": monthly_output, "holdings": holdings_output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output_dir)
    summary = result["summary"]
    print(summary[summary["window"] == "oos"].to_string(index=False))


if __name__ == "__main__":
    main()
