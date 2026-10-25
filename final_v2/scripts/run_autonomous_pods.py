#!/usr/bin/env python3
"""Run a finite four-pod research tick. Offline labelled fixtures by default opt-in."""

import json

from pod_platform.agents.strategy_models import (
    StrategyModelProvider,
    load_profiles,
)
from pod_platform.application.configuration import load_config
from pod_platform.application.research_loop import (
    AutonomousResearchLoop,
    RoleProviders,
    build_fixture_loop,
)
from pod_platform.application.script_settings import load_script_settings

SCRIPT_SETTINGS = load_config("_utility_schemas")["run_autonomous_pods"]


def main(argv=None):

    a = load_script_settings("run_autonomous_pods", SCRIPT_SETTINGS, argv)
    if a.fixture and a.online:
        raise ValueError("--fixture and --online are separate modes")
    if a.fixture:
        loop = build_fixture_loop(a.state_dir)
    else:
        if not (a.online and a.serving_profiles and a.manifest):
            print(
                json.dumps(
                    {
                        "status": "BLOCKED_BACKEND_UNAVAILABLE",
                        "requires": [
                            "--online",
                            "--serving-profiles",
                            "--manifest",
                            "three explicit model roles",
                        ],
                    }
                )
            )
            return 2
        profiles = load_profiles(a.serving_profiles)
        if set(profiles) != {"lead", "challenger", "extractor"}:
            raise ValueError(
                "exactly three explicit serving roles are required"
            )
        loop = AutonomousResearchLoop(
            a.state_dir,
            providers=RoleProviders(
                **{r: StrategyModelProvider(profiles[r]) for r in profiles}
            ),
            manifest_path=a.manifest,
            run_id=a.run_id,
        )
    print(
        json.dumps(
            loop.tick(max_jobs=a.max_jobs, max_iterations=a.max_iterations),
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
