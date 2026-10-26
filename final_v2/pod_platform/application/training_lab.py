"""Run reproducible four-pod development and capital allocation through 2020."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts.base import canonical_digest
from ..research.desk_metrics import compute_desk_metrics
from ..research.training_allocation import allocation_search
from ..research.training_boosted import search_boosted
from ..research.training_data import market_return_series, prepare_panel
from ..research.training_mechanisms import search_mechanisms, selection_universe
from ..research.training_models import search_models
from ..research.training_search import (
    objective,
    project_target,
    search,
    select_strategies,
    summarize,
)
from ..research.training_tactical import (
    combined_schedule,
    daily_market,
    monthly_net,
    replay_daily,
    schedule_for,
    search_frequencies,
)
from ..research.training_walkforward import walk_forward
from .common import atomic_write_json, file_sha256
from .configuration import config_path
from .research_access import additional_input_hashes, write_access_audit

LOGGER = logging.getLogger(__name__)


def _progress(name, count):
    LOGGER.warning("Training research: %s complete (%s trials)", name, count)


def _holdings(weights, panel):
    rows = []
    for t, date in enumerate(panel.dates):
        for i in np.flatnonzero(np.abs(weights[t]) > 1e-10):
            rows.append(
                {
                    "formation_date": date,
                    "permno": int(panel.securities[i]),
                    "weight": float(weights[t, i]),
                    "execution_date": (
                        panel.execution_dates[t]
                        if t < len(panel.execution_dates)
                        else pd.NaT
                    ),
                }
            )
    return pd.DataFrame(rows)


def _record(candidate):
    return {
        "strategy_id": candidate.strategy_id,
        "pod": candidate.pod,
        "recipe": candidate.recipe,
        "metrics": candidate.metrics,
        "selection_score": candidate.score,
        "monthly_net_returns": candidate.net.tolist(),
    }


def _execution_holdings(schedule, panel, market):
    frames = []
    for d, weights in sorted(schedule.items()):
        selected = np.flatnonzero(np.abs(weights) > 1e-10)
        frames.append(
            pd.DataFrame(
                {
                    "execution_date": market.dates[d],
                    "permno": panel.securities[selected],
                    "target_weight": weights[selected],
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _constraints(weights, panel, settings):
    evidence = []
    for t, values in enumerate(weights):
        frame = panel.frames[t]
        beta = frame.beta_60m.fillna(1).to_numpy(dtype=float)
        sectors = frame.gics.fillna("unknown").astype(str).str[:2]
        sector_exposures = pd.Series(values).groupby(sectors.to_numpy()).sum()
        evidence.append(
            {
                "formation_date": str(panel.dates[t].date()),
                "positions": int((np.abs(values) > 1e-10).sum()),
                "gross": float(np.abs(values).sum()),
                "net": float(values.sum()),
                "largest_name": float(np.abs(values).max()),
                "estimated_beta": float(values @ beta),
                "largest_sector_net": float(sector_exposures.abs().max()),
                "short_min_market_cap_millions": float(
                    frame.me.to_numpy()[values < -1e-10].min()
                )
                if np.any(values < -1e-10)
                else None,
            }
        )
    for row in evidence:
        if (
            row["gross"] > settings["gross_limit"] + 1e-6
            or abs(row["net"]) > settings.get("target_net_band", 0.01) + 1e-6
            or row["largest_name"] > settings["single_name_limit"] + 1e-6
            or row["positions"] > settings["universe_size"]
            or (0 < row["positions"] < settings["min_positions"])
            or abs(row["estimated_beta"])
            > settings.get("target_beta_band", 0.05) + 1e-6
            or row["largest_sector_net"]
            > settings.get("target_sector_band", 0.02) + 1e-6
        ):
            raise ValueError(
                f"Training target violates portfolio limits: {row}"
            )
    return evidence


def run_training_lab(config):
    """Search development data, freeze selected books and write reviewable results."""
    settings = config["training_lab"]
    if not 1 <= settings["max_strategies_per_pod"] <= 3:
        raise ValueError(
            "Each pod must select between one and three strategies"
        )
    if not 200 <= settings["min_positions"] <= settings["universe_size"] <= 500:
        raise ValueError("Training portfolio requires 200–500 positions")
    state = config_path(config["application"]["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    panel_path = config_path(settings["panel_path"])
    daily_path = config_path(settings["daily_path"])
    fingerprints = {
        "panel_sha256": file_sha256(panel_path),
        "daily_sha256": file_sha256(daily_path),
        "settings": settings,
        **additional_input_hashes(settings),
    }
    runtime = {"python": platform.python_version()}
    for package in (
        "numpy",
        "pandas",
        "pyarrow",
        "pydantic",
        "scipy",
        "xgboost",
    ):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            continue
    fingerprints["runtime"] = runtime
    code = hashlib.sha256()
    for source in sorted(Path(__file__).parents[1].rglob("*.py")):
        code.update(str(source.relative_to(Path(__file__).parents[1])).encode())
        code.update(source.read_bytes())
    fingerprints["code_sha256"] = code.hexdigest()
    run_hash = canonical_digest(fingerprints)
    output = state / run_hash[:16]
    output.mkdir(exist_ok=True)
    if (output / "training_report.json").exists():
        atomic_write_json(
            state / "latest.json",
            {
                "report": str(output / "training_report.json"),
                "run_hash": run_hash,
            },
        )
        return json.loads((output / "training_report.json").read_text())
    snapshot = output / "source_snapshot"
    package_root = Path(__file__).parents[1]
    for source in package_root.rglob("*.py"):
        destination = (
            snapshot / "pod_platform" / source.relative_to(package_root)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in ("run.py", "verify_training_run.py", "pyproject.toml"):
        shutil.copy2(package_root.parent / name, snapshot / name)
    frozen_config = json.loads(json.dumps(config))
    frozen_config["application"]["state_dir"] = str(output)
    atomic_write_json(snapshot / "config.json", frozen_config)
    atomic_write_json(
        output / "latest.json",
        {
            "report": str(output / "training_report.json"),
            "run_hash": run_hash,
        },
    )
    cache_name = (
        canonical_digest(
            {
                key: value
                for key, value in fingerprints.items()
                if key != "code_sha256"
            }
        )[:16]
        + "_daily.parquet"
    )
    if (
        not (output / cache_name).exists()
        and not (output / cache_name).is_symlink()
    ):
        (output / cache_name).symlink_to(Path("..") / cache_name)
    atomic_write_json(
        output / "experiment.json",
        {
            **fingerprints,
            "run_hash": run_hash,
            "goal": "1–3 positive net-return strategies per pod and performance allocation",
            "scope": "all supplied development observations through 2020",
            "baseline": "equal capital in selected pod strategies",
            "selection_objective": "Sharpe - 0.15*half_Sharpe_gap - 0.5*drawdown",
            "fallback": "report unmet pod goal rather than invent a strategy",
            "evaluation_outcomes_used": False,
        },
    )
    LOGGER.warning(
        "Preparing supplied development panel and daily execution returns"
    )
    panel = prepare_panel(
        panel_path,
        daily_path,
        settings,
        state
        / (
            canonical_digest(
                {k: v for k, v in fingerprints.items() if k != "code_sha256"}
            )[:16]
            + "_daily.parquet"
        ),
    )
    write_access_audit(config, output, panel.metadata)
    candidates = search(panel, settings, _progress)
    market = daily_market(panel, settings)
    candidates.extend(
        search_frequencies(panel, settings, candidates, market, _progress)
    )
    candidates.extend(search_mechanisms(panel, market, settings))
    candidates.extend(search_models(panel, market, settings, _progress))
    candidates.extend(
        search_boosted(
            panel,
            market,
            settings,
            model_dir=output / "models",
            progress=_progress,
        )
    )
    if settings.get("capacity_execution", False):
        for number, trial in enumerate(candidates):
            if (
                "feature" not in trial.recipe
                and "components" not in trial.recipe
            ):
                continue
            book = replay_daily(
                market, schedule_for(trial, panel, market, settings), settings
            )
            trial.net = monthly_net(book["net"], market, panel)
            trial.metrics = summarize(trial.net, market_return_series(panel))
            trial.score = objective(trial.metrics)
            if number % 100 == 0:
                _progress("capacity_rescore", number)
    atomic_write_json(
        output / "candidate_ledger.json", [_record(c) for c in candidates]
    )
    selected = select_strategies(
        selection_universe(candidates, settings), settings
    )
    missing = [pod for pod, strategies in selected.items() if not strategies]
    if missing:
        atomic_write_json(output / "unmet_pods.json", {"pods": missing})
        LOGGER.warning(
            "No positive net training candidate for pods: %s", missing
        )
    best, allocation_trials = allocation_search(
        selected, panel, settings, _progress, market=market
    )
    schedule = combined_schedule(
        selected, best["allocations"], panel, market, settings
    )
    fund_execution = {
        **settings,
        "execution_start_nav": settings.get("fund_capital_usd", 1_000_000_000),
    }
    exact_book = replay_daily(
        market, schedule, fund_execution, keep_holdings=True
    )
    exact_monthly = monthly_net(exact_book["net"], market, panel)
    # Monthly allocator approximation is used only to screen allocation policy;
    # final reported returns come from netted daily shares and actual schedules.
    best["returns"] = exact_monthly
    turnover_series = pd.Series(exact_book["turnover"], index=market.dates)
    monthly_turnover = turnover_series.groupby(
        turnover_series.index.to_period("M")
    ).sum()
    best["turnover"] = np.asarray(
        [
            monthly_turnover.loc[(date + pd.offsets.MonthEnd(1)).to_period("M")]
            for date in panel.dates[:-1]
        ]
    )
    snapshots = best["weights"].copy()
    for t, day in enumerate(panel.execution_dates):
        snapshots[t] = schedule.get(
            int(market.dates.searchsorted(day)), np.zeros(len(panel.securities))
        )
    # A final target is frozen from formation-time information without earning
    # any return after the development boundary.
    mask, design, inverse = market.designs[-1]
    last = snapshots[-1, mask].astype(float)
    last = project_target(
        last,
        design,
        settings,
        shortable=getattr(market.designs[-1], "shortable", None),
    )
    snapshots[-1] = 0
    snapshots[-1, mask] = last
    best["weights"] = snapshots
    constraints = _constraints(snapshots, panel, settings)
    chronological = walk_forward(candidates, panel, market, settings, _progress)
    # Aggregate-level results include all signal and allocator trials in DSR.
    trial_sharpes = [c.metrics["sharpe_periodic"] for c in candidates] + [
        t["metrics"]["sharpe_periodic"] for t in allocation_trials
    ]
    best["metrics"] = summarize(
        best["returns"],
        market_return_series(panel),
        trial_sharpes=[value for value in trial_sharpes if value is not None],
        total_attempts=len(trial_sharpes),
        trial_periods_per_year=12,
        attempts_complete=True,
        unmeasured_attempts=sum(value is None for value in trial_sharpes),
    )
    pod_reports = {}
    return_table = pd.DataFrame(
        {
            "formation_date": panel.dates[:-1],
            "execution_date": panel.execution_dates,
            "settlement_date": panel.settlement_dates,
            "fund_net_return": best["returns"],
            "fund_turnover": best["turnover"],
        }
    )
    for pod, strategies in selected.items():
        pod_allocations = [
            {**r, "capital_weight": r["within_pod_weight"]}
            for r in best["allocations"]
            if r["pod"] == pod
        ]
        pod_schedule = combined_schedule(
            {pod: strategies}, pod_allocations, panel, market, settings
        )
        pod_book = replay_daily(market, pod_schedule, settings)
        pod_net = monthly_net(pod_book["net"], market, panel)
        stressed_book = replay_daily(
            market,
            pod_schedule,
            {**settings, "cost_bps_one_way": settings["cost_bps_one_way"] * 2},
        )
        stressed = monthly_net(stressed_book["net"], market, panel)
        pod_reports[pod] = {
            "strategies": [_record(s) for s in strategies],
            "strategy_count": len(strategies),
            "allocated_pod_metrics": summarize(
                pod_net, market_return_series(panel)
            ),
            "double_cost_metrics": summarize(stressed),
        }
        return_table[pod + "_net_return"] = pod_net
        for strategy in strategies:
            name = strategy.strategy_id.replace(":", "_").replace("+", "p")
            _holdings(strategy.weights, panel).to_parquet(
                output / (name + "_holdings.parquet"), index=False
            )
            return_table[strategy.strategy_id] = strategy.net
    stressed_book = replay_daily(
        market,
        schedule,
        {
            **fund_execution,
            "cost_bps_one_way": settings["cost_bps_one_way"] * 2,
        },
    )
    stress = monthly_net(stressed_book["net"], market, panel)
    observed = market.dates >= panel.execution_dates[0]
    pd.DataFrame(
        {
            "date": market.dates[observed],
            "net_return": exact_book["net"][observed],
            "turnover": exact_book["turnover"][observed],
        }
    ).to_csv(output / "daily_returns.csv", index=False)
    walkforward_table = pd.DataFrame(
        {
            "date": chronological["dates"],
            "fund_net_return": chronological["daily_returns"],
            **{
                pod: value["daily_returns"]
                for pod, value in chronological["pods"].items()
            },
        }
    )
    walkforward_table.to_csv(
        output / "walk_forward_daily_returns.csv", index=False
    )
    walkforward_summary = {
        k: v
        for k, v in chronological.items()
        if k not in {"daily_returns", "dates", "schedule", "book", "pods"}
    }
    walkforward_summary["pods"] = {
        pod: {k: v for k, v in value.items() if k != "daily_returns"}
        for pod, value in chronological["pods"].items()
    }
    atomic_write_json(output / "walk_forward.json", walkforward_summary)
    _execution_holdings(schedule, panel, market).to_parquet(
        output / "fund_execution_targets.parquet", index=False
    )
    atomic_write_json(
        output / "fund_execution_dates.json",
        [str(market.dates[d].date()) for d in sorted(schedule)],
    )
    _execution_holdings(chronological["schedule"], panel, market).to_parquet(
        output / "walk_forward_execution_targets.parquet", index=False
    )
    atomic_write_json(
        output / "walk_forward_execution_dates.json",
        [
            str(market.dates[d].date())
            for d in sorted(chronological["schedule"])
        ],
    )
    monthly_allocations = pd.DataFrame(best["allocations"])
    monthly_allocations.to_csv(output / "capital_allocations.csv", index=False)
    return_table.to_csv(output / "monthly_returns.csv", index=False)
    _holdings(best["weights"], panel).to_parquet(
        output / "fund_holdings.parquet", index=False
    )
    atomic_write_json(
        output / "candidate_ledger.json", [_record(c) for c in candidates]
    )
    allocation_records = [
        {"policy": t["policy"], "metrics": t["metrics"], "score": t["score"]}
        for t in allocation_trials
    ]
    atomic_write_json(output / "allocation_trials.json", allocation_records)
    atomic_write_json(output / "constraints.json", constraints)
    final_allocations = [
        r
        for r in best["allocations"]
        if r["formation_date"] == str(panel.dates[-1].date())
    ]
    atomic_write_json(
        output / "frozen_model.json",
        {
            "training_cutoff": settings["end"],
            "run_hash": run_hash,
            "code_sha256": fingerprints["code_sha256"],
            "strategies": {
                pod: [_record(s) for s in strategies]
                for pod, strategies in selected.items()
            },
            "capital_policy": best["policy"],
            "capital_allocations": final_allocations,
            "formation_targets": [
                {
                    "formation_date": str(panel.dates[-1].date()),
                    "permno": int(security),
                    "weight": float(weight),
                }
                for security, weight in zip(panel.securities, snapshots[-1])
                if abs(weight) > 1e-10
            ],
            "production_admission": False,
        },
    )
    report = {
        "status": "completed_training_research",
        "run_hash": run_hash,
        "output_dir": str(output),
        "data": panel.metadata,
        "signal_trials": len(candidates),
        "allocation_trials": len(allocation_trials),
        "selection_uses_entire_training_period": True,
        "allocation_uses_only_settled_prior_returns": True,
        "training_selection_bias": True,
        "attempt_history_scope": "this_declared_training_battery_prior_project_exploration_not_included",
        "production_admission": False,
        "fund": {
            "metrics": best["metrics"],
            "double_cost_metrics": summarize(stress),
            "allocation_policy": best["policy"],
            "final_capital_allocations": final_allocations,
            "constraints_all_months_passed": True,
            "daily_metrics": compute_desk_metrics(
                exact_book["net"][observed], periods_per_year=252
            ),
            "execution": exact_book.get("execution"),
        },
        "pods": pod_reports,
        "walk_forward": walkforward_summary,
        "research_goals": {
            "one_to_three_strategies_every_pod": all(
                1 <= len(s) <= 3 for s in selected.values()
            ),
            "positive_chronological_fund_return": chronological[
                "monthly_metrics"
            ]["annualized_arithmetic_net_return"]
            > 0,
            "positive_chronological_return_every_pod": all(
                v["monthly_metrics"]["annualized_arithmetic_net_return"] > 0
                for v in chronological["pods"].values()
            ),
        },
        "assumptions": {
            "transaction_cost_bps_one_way": settings["cost_bps_one_way"],
            "short_borrow_bps_annual": settings["borrow_bps_annual"],
            "borrow_availability": "market_cap_adv_and_inventory_proxy_no_actual_locates",
            "borrow_large_cap_bps": settings.get("large_cap_borrow_bps"),
            "borrow_other_bps": settings.get("other_borrow_bps"),
            "capacity": "participation_limited_compounding_fund_simulation_not_broker_certified",
            "research_universe_cap": settings.get("research_universe_size"),
            "spread": "half_of_month_end_high_low_spread_estimate_with_cost_floor",
            "actual_daily_volume": "not_supplied_trailing_126_day_adv_used",
            "cash_rate": "zero_for_research_nav_returns_are_cash_excess_pnl",
            "events_pod": "issuance_accruals_and_corporate_action_proxies",
            "benchmark": "long_short_total_return_spread_cash_rate_not_added",
            "alpha_regression": "intercept_vs_equal_weight_formation_universe",
        },
        "reference": "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf",
        "borrow_reference": "https://www.interactivebrokers.com/en/trading/short-securities-availability.php",
    }
    atomic_write_json(output / "training_report.json", report)
    atomic_write_json(
        state / "latest.json",
        {"report": str(output / "training_report.json"), "run_hash": run_hash},
    )
    return report
