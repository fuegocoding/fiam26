"""Independently replay saved training orders and export actual daily holdings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pod_platform.application.common import atomic_write_json, file_sha256
from pod_platform.application.configuration import config_path, load_config
from pod_platform.contracts.base import canonical_digest
from pod_platform.research.training_data import prepare_panel
from pod_platform.research.training_tactical import daily_market, replay_daily


def replay_saved_schedule(output, prefix, panel, market, settings):
    """Replay exported targets and preserve explicit flat decisions."""
    decisions = json.loads(
        (output / f"{prefix}_execution_dates.json").read_text()
    )
    schedule = {
        int(market.dates.get_loc(pd.Timestamp(day))): np.zeros(
            len(panel.securities), dtype=np.float32
        )
        for day in decisions
    }
    targets = pd.read_parquet(output / f"{prefix}_execution_targets.parquet")
    security_index = pd.Index(panel.securities)
    for day, group in targets.groupby("execution_date", sort=True):
        index = int(market.dates.get_loc(day))
        locations = security_index.get_indexer(group.permno)
        if (locations < 0).any() or index not in schedule:
            raise ValueError(
                "Saved target contains an unknown security or date"
            )
        schedule[index][locations] = group.target_weight.to_numpy()
    return replay_daily(
        market,
        schedule,
        {**settings, "execution_start_nav": settings["fund_capital_usd"]},
        keep_holdings=True,
    )


def mandate_passed(execution, settings):
    return execution["maximum_positions"] >= settings["min_positions"] and all(
        execution[key] == 0
        for key in (
            "gross_limit_breach_sessions",
            "net_limit_breach_sessions",
            "single_name_limit_breach_sessions",
            "position_count_breach_sessions",
        )
    )


def verify_training_run():
    """Reconstruct fills from saved decisions, including transitions to cash."""
    config = load_config()
    state = config_path(config["application"]["state_dir"])
    pointer = json.loads((state / "latest.json").read_text())
    report_path = Path(pointer["report"])
    output = report_path.parent
    report = json.loads(report_path.read_text())
    experiment = json.loads((output / "experiment.json").read_text())
    settings = experiment["settings"]
    code = hashlib.sha256()
    root = Path(__file__).parent / "pod_platform"
    for source in sorted(root.rglob("*.py")):
        code.update(str(source.relative_to(root)).encode())
        code.update(source.read_bytes())
    if code.hexdigest() != experiment["code_sha256"]:
        raise ValueError("Saved results require their exact model code version")
    panel_path = config_path(settings["panel_path"])
    daily_path = config_path(settings["daily_path"])
    if (
        file_sha256(panel_path) != experiment["panel_sha256"]
        or file_sha256(daily_path) != experiment["daily_sha256"]
    ):
        raise ValueError("Saved results require their exact input snapshots")
    cache_key = canonical_digest(
        {
            key: experiment[key]
            for key in ("panel_sha256", "daily_sha256", "settings", "runtime")
            if key in experiment
        }
    )[:16]
    panel = prepare_panel(
        panel_path, daily_path, settings, state / (cache_key + "_daily.parquet")
    )
    market = daily_market(panel, settings)
    book = replay_saved_schedule(output, "fund", panel, market, settings)
    expected = pd.read_csv(output / "daily_returns.csv", parse_dates=["date"])
    indices = market.dates.get_indexer(expected.date)
    if (indices < 0).any():
        raise ValueError(
            "Saved return calendar differs from the input calendar"
        )
    discrepancy = float(
        np.max(np.abs(book["net"][indices] - expected.net_return))
    )
    if discrepancy > 1e-9:
        raise ValueError(f"Saved orders do not reconcile: {discrepancy}")
    nav = settings["fund_capital_usd"] * np.cumprod(1 + book["net"])
    nav_difference = abs(nav[-1] - report["fund"]["execution"]["final_nav_usd"])
    if nav_difference > 1:
        raise ValueError(f"Compounded NAV does not reconcile: {nav_difference}")
    held = book["holdings"]
    rows, columns = np.nonzero(np.abs(held) > 1e-10)
    pd.DataFrame(
        {
            "date": market.dates[rows],
            "permno": panel.securities[columns],
            "weight": held[rows, columns],
            "notional_usd": held[rows, columns] * nav[rows],
        }
    ).to_parquet(output / "actual_daily_holdings.parquet", index=False)
    pd.DataFrame(
        {
            "date": market.dates,
            "net_return": book["net"],
            "nav_usd": nav,
        }
    ).to_csv(output / "reconciled_daily_nav.csv", index=False)
    review = {
        "run_hash": report["run_hash"],
        "source_and_data_hashes_match": True,
        "orders_reconcile_with_saved_returns": True,
        "maximum_daily_return_discrepancy": discrepancy,
        "ending_nav_discrepancy_usd": float(nav_difference),
        "execution": book["execution"],
        "mandate_limits_passed": mandate_passed(book["execution"], settings),
    }
    forward = replay_saved_schedule(
        output, "walk_forward", panel, market, settings
    )
    forward_expected = pd.read_csv(
        output / "walk_forward_daily_returns.csv", parse_dates=["date"]
    )
    forward_indices = market.dates.get_indexer(forward_expected.date)
    if (forward_indices < 0).any():
        raise ValueError("Chronological return calendar differs from inputs")
    forward_difference = float(
        np.max(
            np.abs(
                forward["net"][forward_indices]
                - forward_expected.fund_net_return
            )
        )
    )
    if forward_difference > 1e-9:
        raise ValueError(
            f"Chronological orders do not reconcile: {forward_difference}"
        )
    forward_nav = settings["fund_capital_usd"] * np.cumprod(1 + forward["net"])
    pd.DataFrame(
        {
            "date": market.dates[forward_indices],
            "net_return": forward["net"][forward_indices],
            "nav_usd": forward_nav[forward_indices],
        }
    ).to_csv(output / "walk_forward_reconciled_nav.csv", index=False)
    rows, columns = np.nonzero(np.abs(forward["holdings"]) > 1e-10)
    pd.DataFrame(
        {
            "date": market.dates[rows],
            "permno": panel.securities[columns],
            "weight": forward["holdings"][rows, columns],
            "notional_usd": forward["holdings"][rows, columns]
            * forward_nav[rows],
        }
    ).to_parquet(output / "walk_forward_actual_holdings.parquet", index=False)
    review["walk_forward"] = {
        "orders_reconcile_with_saved_returns": True,
        "maximum_daily_return_discrepancy": forward_difference,
        "execution": forward["execution"],
        "mandate_limits_passed": mandate_passed(forward["execution"], settings),
    }
    atomic_write_json(output / "execution_review.json", review)
    return review


if __name__ == "__main__":
    print(json.dumps(verify_training_run(), sort_keys=True))
