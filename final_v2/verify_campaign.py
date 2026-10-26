"""Replay a completed campaign's saved instructions against supplied daily prices."""

from __future__ import annotations

import hashlib
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
from pod_platform.research.campaign_metrics import competition_report
from pod_platform.research.sleeve_execution import RecordedInstruction
from pod_platform.research.training_data import prepare_panel
from pod_platform.research.training_tactical import daily_market, replay_daily


def verify_cost_lineage(output, experiment, report):
    calibration = experiment.get("borrow_fee_calibration")
    settings = experiment["settings"]
    if calibration is None:
        if settings.get("borrow_fee_calibration_path"):
            raise ValueError(
                "Declared borrow calibration is missing from the experiment"
            )
        return False
    copy = Path(output) / "borrow_fee_calibration.json"
    if file_sha256(copy) != settings.get("borrow_fee_calibration_sha256"):
        raise ValueError("Saved borrow calibration receipt hash changed")
    if (
        json.loads(copy.read_text()) != calibration
        or report.get("borrow_fee_model") != calibration
    ):
        raise ValueError(
            "Borrow calibration lineage differs across report and experiment"
        )
    if any(
        settings.get(key) != value
        for key, value in calibration["parameters"].items()
    ):
        raise ValueError(
            "Configured execution fees differ from the saved calibration"
        )
    return True


def verify(report_path):
    report_path = Path(report_path)
    output = report_path.parent
    report = json.loads(report_path.read_text())
    experiment = json.loads((output / "experiment.json").read_text())
    config = json.loads((output / "config.json").read_text())
    settings = experiment["settings"]
    cost_lineage_verified = verify_cost_lineage(output, experiment, report)
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
    if experiment.get("benchmark_reporting_sha256") != file_sha256(
        config_path(config["research_campaign"]["benchmark_fred_path"])
    ):
        raise ValueError("Benchmark reporting input hash changed")
    if experiment.get("evidence_sha256") != file_sha256(
        output / "evidence.json"
    ):
        raise ValueError("Research evidence hash changed")
    source_root = output / "source_snapshot" / "pod_platform"
    snapshot_digest = hashlib.sha256()
    for source in sorted(source_root.rglob("*.py")):
        snapshot_digest.update(str(source.relative_to(source_root)).encode())
        snapshot_digest.update(source.read_bytes())
    if snapshot_digest.hexdigest() != experiment["source_sha256"]:
        raise ValueError("Saved source snapshot hash changed")
    panel = prepare_panel(
        config_path(settings["panel_path"]),
        config_path(settings["daily_path"]),
        settings,
        output.parent
        / (experiment["daily_sha256"][:16] + "_campaign_daily.parquet"),
    )
    market = daily_market(panel, settings)
    panel.daily = None
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
    if targets.duplicated(["execution_date", "permno"]).any():
        raise ValueError("Duplicate saved execution target")
    if len(dates) != len(set(dates)):
        raise ValueError("Duplicate execution date")
    for date, group in targets.groupby("execution_date"):
        day = int(market.dates.get_loc(date))
        for row in group.itertuples():
            schedule[day][index[int(row.permno)]] = row.target_weight
    if (output / "execution_renewals.parquet").exists():
        masks = {day: np.zeros(len(panel.securities), bool) for day in schedule}
        renewal_rows = pd.read_parquet(output / "execution_renewals.parquet")
        if renewal_rows.duplicated(["execution_date", "permno"]).any():
            raise ValueError("Duplicate saved order renewal")
        for date, group in renewal_rows.groupby("execution_date"):
            day = market.dates.get_loc(date)
            masks[day][[index[int(p)] for p in group.permno]] = True
        schedule = {
            day: RecordedInstruction(target, masks[day])
            for day, target in schedule.items()
        }
    replay = replay_daily(
        market,
        schedule,
        {
            **settings,
            "execution_start_nav": settings["fund_capital_usd"],
            "holdings_storage_path": str(
                output / "verified_actual_holdings.npy"
            ),
        },
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
    if holdings.duplicated(["date", "permno"]).any():
        raise ValueError("Duplicate saved holding")
    for component, values in replay["attribution"].items():
        np.testing.assert_allclose(saved[component], values[active], atol=1e-12)
    components = (
        saved.long_return
        + saved.short_return
        + saved.cash_return
        - saved.trading_cost_return
        - saved.borrow_cost_return
    )
    np.testing.assert_allclose(components, saved.net, atol=1e-12)
    metrics, _, _ = competition_report(
        market.dates[active],
        replay["net"][active],
        benchmark,
        turnover=replay["turnover"][active],
        returns_include_cash=True,
    )
    for frequency in ("daily", "monthly"):
        for metric, value in metrics[frequency].items():
            recorded = report["fund"][frequency][metric]
            if value is None:
                if recorded is not None:
                    raise ValueError(
                        "Undefined performance metric was fabricated"
                    )
            else:
                np.testing.assert_allclose(
                    value, recorded, rtol=1e-8, atol=1e-10
                )
    actual = replay["holdings"][active]
    counts = (np.abs(actual) > 1e-10).sum(axis=1)
    constraints = {
        "minimum_position_failures": int(
            (counts < settings["min_positions"]).sum()
        ),
        "maximum_position_failures": int(
            (counts > settings["universe_size"]).sum()
        ),
        "maximum_gross": float(np.abs(actual).sum(axis=1).max()),
        "maximum_absolute_net": float(np.abs(actual.sum(axis=1)).max()),
        "maximum_absolute_name": float(np.abs(actual).max()),
    }
    for key, value in constraints.items():
        np.testing.assert_allclose(value, report["constraints"][key], atol=1e-6)
    constraints_pass = (
        constraints["minimum_position_failures"] == 0
        and constraints["maximum_position_failures"] == 0
        and constraints["maximum_gross"] <= 2 + 1e-5
        and constraints["maximum_absolute_net"] <= 0.5 + 1e-5
        and constraints["maximum_absolute_name"] <= 0.05 + 1e-5
    )
    if constraints_pass != report["constraints_pass"]:
        raise ValueError(
            "Reported constraint admission disagrees with actual holdings"
        )
    for date, group in holdings.groupby("date"):
        day = market.dates.get_loc(date)
        columns = [index[int(p)] for p in group.permno]
        np.testing.assert_allclose(
            replay["holdings"][day, columns], group.weight, rtol=1e-6, atol=1e-9
        )
    total_saved = int((np.abs(replay["holdings"][active]) > 1e-10).sum())
    if total_saved != len(holdings):
        raise ValueError("Saved holdings omit or add positions")
    # Reconstruct short increases in actual dollars; drift alone is not a new
    # borrow. Formation cap is checked independently of the engine's counters.
    full_nav = settings["fund_capital_usd"] * np.cumprod(1 + replay["net"])
    previous_dollars = np.zeros(len(panel.securities))
    ineligible_short_increases = 0
    for day in range(len(market.dates)):
        raw = market.returns[day]
        valid = np.isfinite(raw) & (raw >= -1)
        marked = np.where(valid, previous_dollars * (1 + np.nan_to_num(raw)), 0)
        current = replay["holdings"][day] * full_nav[day]
        rounding = (
            np.finfo(np.float32).eps
            * 8
            * np.maximum(np.maximum(np.abs(current), np.abs(marked)), 1.0)
        )
        increased = np.maximum(-current, 0) - np.maximum(
            -marked, 0
        ) > np.maximum(1.0, rounding)
        formation = int(market.formation[day])
        if formation >= 0:
            caps = panel.frames[formation].me.fillna(0).to_numpy(float) * 1e6
            ineligible_short_increases += int(
                np.count_nonzero(increased & (caps < 500e6))
            )
        previous_dollars = current
    if ineligible_short_increases:
        raise ValueError(
            "Actual short increases violate the formation market-cap floor"
        )
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
            r["capital_usd"] < 0
            or r["capital_usd"] > r["hard_capacity_usd"] + 1
            for r in row["funded"]
        ):
            raise ValueError("Capital exceeds measured capacity")
        by_pod = {}
        for funded in row["funded"]:
            pod = funded["strategy_id"].split(":", 1)[0]
            by_pod[pod] = by_pod.get(pod, 0) + funded["capital_usd"]
        if any(
            amount > row["nav_usd"] * settings["max_pod_capital"] + 1
            for amount in by_pod.values()
        ):
            raise ValueError("Funded capital exceeds a pod budget")
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
        "competition_constraints_pass": bool(constraints_pass),
        "recomputed_constraints": constraints,
        "benchmark_source_evidence_hashes_verified": True,
        "cost_calibration_provenance_verified": cost_lineage_verified,
        "headline_metrics_recomputed": True,
        "ineligible_actual_short_increases": ineligible_short_increases,
        "attribution_reconciled": True,
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
