"""Run the strategy platform using settings in config.json."""

from __future__ import annotations

import json
from pathlib import Path

from pod_platform.application.strategy_workflow import run_configured


def main():
    """Execute the configured workflow and print its stage results."""
    result = run_configured()
    if result.get("status") in {
        "integrated_campaign_completed",
        "research_completed_agent_stage_degraded",
        "research_completed_pool_incomplete",
        "recorded_campaign_replayed",
    }:
        result = {
            k: result[k]
            for k in (
                "status",
                "output_dir",
                "attempts",
                "successful_strategies",
                "owned_strategy_counts",
                "research_pool_target_met",
                "constraints_pass",
                "new_model_call_attempts",
            )
        } | {"report": str(Path(result["output_dir"]) / "campaign_report.json"),
             "supported_strategy_counts": {p: row["supported_count"]
                 for p, row in result["research_pool"].items()}}
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
