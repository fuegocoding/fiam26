#!/usr/bin/env python3
"""Replay saved research through bounded component tests without calling model APIs."""

import json

from pod_platform.application.component_replay import replay_components
from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings

SCRIPT_SETTINGS = load_config("_utility_schemas")["replay_component_research"]


def main():

    args = load_script_settings(
        "replay_component_research", SCRIPT_SETTINGS, None
    )
    result = replay_components(
        from_research=args.from_research,
        state_dir=args.state_dir,
        manifest_path=args.manifest,
        papers_db=args.papers_db,
    )
    print(
        json.dumps(
            {
                "new_model_api_calls": 0,
                "admissions": result["admissions"],
                "comparisons": [
                    {
                        "namespace": c["namespace"],
                        "status": c["route"]["status"],
                        "measurement": c["route"]
                        .get("probe_report", {})
                        .get("status"),
                    }
                    for c in result["comparisons"]
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
