"""Replay a completed campaign's saved instructions against supplied daily prices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pod_platform.application.campaign_sources import load_reporting_benchmarks
from pod_platform.application.common import atomic_write_json, file_sha256
from pod_platform.application.configuration import config_path, load_config
from pod_platform.application.memory_control import (
    memory_watchdog,
    research_job_lock,
)
from pod_platform.research.training_data import prepare_panel
from pod_platform.research.training_tactical import daily_market, replay_daily


def verify(report_path):
    report_path = Path(report_path)
    output = report_path.parent
    report = json.loads(report_path.read_text())
    experiment = json.loads((output / "experiment.json").read_text())
    config = json.loads((output / "config.json").read_text())
    settings = experiment["settings"]
    for field, digest in (
        ("panel_path", "panel_sha256"),
        ("daily_path", "daily_sha256"),
    ):
        if file_sha256(config_path(settings[field])) != experiment[digest]:
            raise ValueError("Saved experiment input hash changed")
    if "filings_sha256" in experiment:
        if (
            file_sha256(config_path(settings["filings_path"]))
            != experiment["filings_sha256"]
        ):
            raise ValueError("Saved filing corpus hash changed")
    panel = prepare_panel(
        config_path(settings["panel_path"]),
        config_path(settings["daily_path"]),
        settings,
        output.parent
        / (experiment["daily_sha256"][:16] + "_campaign_daily.parquet"),
    )
    market = daily_market(panel, settings)
    active = market.dates >= panel.execution_dates[0]
    benchmark = load_reporting_benchmarks(
        config_path(config["research_campaign"]["benchmark_fred_path"]),
        market.dates[active],
    )
    market.cash_returns = (
        benchmark.tb3ms.reindex(market.dates).fillna(0).to_numpy() / 25200
    )
    dates = json.loads((output / "execution_dates.json").read_text())
    schedule = {
        int(market.dates.get_loc(pd.Timestamp(d))): np.zeros(
            len(panel.securities)
        )
        for d in dates
    }
    index = {int(p): i for i, p in enumerate(panel.securities)}
    targets = pd.read_parquet(output / "execution_targets.parquet")
    for date, group in targets.groupby("execution_date"):
        day = int(market.dates.get_loc(date))
        for row in group.itertuples():
            schedule[day][index[int(row.permno)]] = row.target_weight
    replay = replay_daily(
        market,
        schedule,
        {**settings, "execution_start_nav": settings["fund_capital_usd"]},
        keep_holdings=True,
    )
    saved = pd.read_csv(
        output / "daily_ledger.csv", parse_dates=["date"]
    ).set_index("date")
    if not saved.index.equals(market.dates[active]):
        raise ValueError("Recorded daily return calendar changed")
    difference = float(
        np.max(np.abs(saved.net.to_numpy() - replay["net"][active]))
    )
    if difference > 1e-10:
        raise ValueError("Daily net return replay failed")
    holdings = pd.read_parquet(output / "actual_holdings.parquet")
    for date, group in holdings.groupby("date"):
        day = market.dates.get_loc(date)
        columns = [index[int(p)] for p in group.permno]
        np.testing.assert_allclose(
            replay["holdings"][day, columns], group.weight, rtol=1e-6, atol=1e-9
        )
    total_saved = int((np.abs(replay["holdings"][active]) > 1e-10).sum())
    if total_saved != len(holdings):
        raise ValueError("Saved holdings omit or add positions")
    allocations = json.loads((output / "capital_allocations.json").read_text())
    for row in allocations:
        if pd.Timestamp(row["decision_asof"]) >= pd.Timestamp(
            row["execution_date"]
        ):
            raise ValueError(
                "Decision uses information on or after execution date"
            )
        if sum(r["capital_usd"] for r in row["funded"]) > row["nav_usd"] + 1:
            raise ValueError("Capital exceeds fund NAV")
        if any(
            r["capital_usd"] > r["hard_capacity_usd"] + 1 for r in row["funded"]
        ):
            raise ValueError("Capital exceeds measured capacity")
    books = json.loads((output / "pod_books.json").read_text())
    if set(books) != {"fundamentals", "events", "valuation", "dynamics"} or any(
        not 1 <= len(rows) <= 3 for rows in books.values()
    ):
        raise ValueError("Pod ownership invariant failed")
    expected_nav = settings["fund_capital_usd"] * np.prod(1 + saved.net)
    np.testing.assert_allclose(
        replay["execution"]["final_nav_usd"], expected_nav, rtol=1e-10
    )
    np.testing.assert_allclose(
        report["execution"]["final_nav_usd"], expected_nav, rtol=1e-10
    )
    review = {
        "status": "verified_saved_instruction_replay",
        "maximum_daily_return_difference": difference,
        "daily_sessions": len(saved),
        "holdings_rows": len(holdings),
        "cash_is_inside_execution_nav": True,
        "pod_rosters_valid": True,
        "allocation_timing_and_capacity_valid": True,
        "final_nav_usd": float(expected_nav),
        "competition_constraints_pass": report["constraints_pass"],
        "profitable_strategy_admission": False,
        "scope": "Independent artifact reload and static replay; this does not independently validate source vintage, borrow proxies or future alpha",
    }
    atomic_write_json(output / "independent_replay_review.json", review)
    return review


def main():
    state = config_path(load_config()["application"]["state_dir"])
    pointer = json.loads((state / "latest_campaign.json").read_text())
    with research_job_lock(state), memory_watchdog():
        print(json.dumps(verify(pointer["report"]), sort_keys=True))


if __name__ == "__main__":
    main()
