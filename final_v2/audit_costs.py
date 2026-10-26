"""Replay recorded unit instructions across disclosed, unfitted cost scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from pod_platform.application.common import atomic_write_json, file_sha256
from pod_platform.application.configuration import config_path, load_config
from pod_platform.application.edge_research import source_digest
from pod_platform.application.memory_control import (
    collect_research_memory,
    memory_watchdog,
    research_job_lock,
)
from pod_platform.contracts.base import canonical_digest
from pod_platform.research.cost_sensitivity import (
    CASES,
    frozen_cost_sensitivity,
)
from pod_platform.research.edge_capacity import measure_edge
from pod_platform.research.strategy_code import (
    CodeStrategy,
    native_review_calendar,
)
from pod_platform.research.training_data import prepare_panel
from pod_platform.research.training_tactical import daily_market

DEFAULT_NAMES = [
    "development_liquidity_fundamentals",
    "development_liquidity_events",
    "development_liquidity_valuation",
    "control_liquid_hedged_dynamics",
]


def audit(report_path, names):
    output = Path(report_path).parent
    experiment = json.loads((output / "experiment.json").read_text())
    settings = {
        **experiment["settings"],
        "compact_storage": True,
        "filings_research_enabled": False,
    }
    # Alpha and memberships are already frozen in saved weights. Raw filing text
    # and model fitting are unnecessary for this implementation-cost audit.
    for field, digest in (
        ("panel_path", "panel_sha256"),
        ("daily_path", "daily_sha256"),
    ):
        if file_sha256(config_path(settings[field])) != experiment[digest]:
            raise ValueError("Frozen experiment inputs changed")
    ledger = json.loads((output / "attempt_ledger.json").read_text())
    selected = [
        row
        for row in ledger
        if row["status"] == "evaluated" and row["strategy"]["name"] in names
    ]
    if {row["strategy"]["name"] for row in selected} != set(names):
        raise ValueError(
            "Every requested strategy must have an evaluated recorded unit book"
        )
    panel = prepare_panel(
        config_path(settings["panel_path"]),
        config_path(settings["daily_path"]),
        settings,
        output.parent
        / (experiment["daily_sha256"][:16] + "_campaign_daily.parquet"),
    )
    market = daily_market(panel, settings)
    panel.daily = None
    collect_research_memory(panel.features.values())
    audit_root = output / "frozen_cost_audit"
    destination = audit_root / ("audit_" + uuid4().hex)
    destination.mkdir(parents=True, exist_ok=False)
    audit_provenance = {
        "auditor_sha256": file_sha256(Path(__file__)),
        "evaluator_source_sha256": source_digest(),
        "parent_experiment_sha256": file_sha256(output / "experiment.json"),
        "parent_report_sha256": file_sha256(Path(report_path)),
        "settings_sha256": canonical_digest(settings),
        "scenario_manifest": [
            {"name": name, "overrides": values} for name, values in CASES
        ],
        "source_policy": "Audit uses the current evaluator, independently reconciled against the recorded parent; parent source snapshot is not claimed as the audit runtime",
    }
    rows, checks = [], []
    for record in selected:
        identity = record["strategy_id"]
        strategy = CodeStrategy.model_validate(record["strategy"])
        stem = identity.replace(":", "_")
        weights_path = output / (stem + "_schedule.npy")
        weights = np.load(weights_path, mmap_mode="r")
        calendar = [
            day
            for day, _, update in native_review_calendar(
                strategy, panel, market
            )
            if update or strategy.review != "monthly" or strategy.entry_rank
        ]
        if (output / (stem + "_schedule_dates.json")).exists():
            saved_dates = json.loads(
                (output / (stem + "_schedule_dates.json")).read_text()
            )
            if saved_dates != [str(market.dates[d].date()) for d in calendar]:
                raise ValueError(
                    "Recorded and independently reconstructed review calendars differ"
                )
        if len(weights) != len(calendar) or weights.shape[1] != len(
            panel.securities
        ):
            raise ValueError("Frozen schedule identity or timing differs")
        schedule = {day: weights[row] for row, day in enumerate(calendar)}
        original = pd.read_parquet(output / (stem + "_unit_returns.parquet"))
        if not pd.DatetimeIndex(original.date).equals(market.dates):
            raise ValueError("Recorded unit-return calendar differs")
        reference = measure_edge(
            market, schedule, settings, keep_holdings=False
        )
        difference = float(
            np.max(np.abs(reference["net"] - original.net_fixed_friction))
        )
        if difference > 1e-10:
            raise ValueError(
                "Reference unit replay does not reconcile; cost conclusions are withheld"
            )
        scenarios = frozen_cost_sensitivity(
            market, schedule, panel, settings, reference
        )
        for scenario in scenarios:
            rows.append(
                {
                    "strategy_id": identity,
                    "strategy_name": strategy.name,
                    **scenario,
                }
            )
        checks.append(
            {
                "strategy_id": identity,
                "maximum_reference_return_difference": difference,
                "frozen_schedule_sha256": file_sha256(weights_path),
                "instructions": len(calendar),
            }
        )
        collect_research_memory([weights, *panel.features.values()])
        print(f"Cost audit reconciled {identity}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "cost_sensitivity.csv", index=False)
    review = {
        "audit_provenance": audit_provenance,
        "status": "completed_frozen_instruction_cost_sensitivity",
        "source_report": str(report_path),
        "reference_replay_checks": checks,
        "strategies": len(selected),
        "scenarios_per_strategy": len(frame) // len(selected),
        "new_model_calls": 0,
        "signal_retuning": False,
        "includes_market_impact": False,
        "scope": "Same saved target weights and review dates; NAV compounds separately under each scenario. Zero-size edge excludes impact and dollar capacity. These are unfitted sensitivities, not calibrated execution quotes or financial admission.",
        "csv": str(destination / "cost_sensitivity.csv"),
    }
    atomic_write_json(destination / "cost_review.json", review)
    atomic_write_json(
        audit_root / "latest_audit.json",
        {"review": str(destination / "cost_review.json")},
    )
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strategy", nargs="+", default=DEFAULT_NAMES)
    arguments = parser.parse_args()
    state = config_path(load_config()["application"]["state_dir"])
    report = arguments.report or Path(
        json.loads((state / "latest_campaign.json").read_text())["report"]
    )
    with research_job_lock(state), memory_watchdog():
        print(json.dumps(audit(report, arguments.strategy), sort_keys=True))


if __name__ == "__main__":
    main()
