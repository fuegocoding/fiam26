#!/usr/bin/env python3
"""One-year bounded contemporary research trial. Never backdates model/admission receipts."""

import json

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.application.year_trial import (
    CappedResearchProvider,
    YearResearchTrial,
)

SCRIPT_SETTINGS = load_config("_utility_schemas")["run_year_research_trial"]


def main():

    a = load_script_settings("run_year_research_trial", SCRIPT_SETTINGS, None)
    if not a.online:
        raise ValueError("actual Groq inference requires --online")
    trial = YearResearchTrial(
        a.state_dir,
        manifest_path=a.manifest,
        dataset_id=a.dataset_id,
        papers_path=a.papers_db,
        year=a.year,
        max_calls=a.max_calls,
        active_months=a.active_months,
        predecessor=a.predecessor,
        provider=CappedResearchProvider(
            a.budget_dir or a.state_dir,
            max_calls=a.max_calls,
            min_interval_seconds=a.min_call_interval,
        ),
    )
    result = trial.run(
        a.prepared_papers,
        daily_csv=a.daily_csv,
        source_discovery=a.source_discovery,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "year",
                    "mode",
                    "calendar_ticks",
                    "submitted_comparisons",
                    "papers_retrieved",
                    "model_budget",
                    "jobs",
                    "trading_status",
                    "strategy_return",
                )
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
