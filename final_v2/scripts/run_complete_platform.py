#!/usr/bin/env python3
"""Finite offline source-to-book platform run; all demo economics are synthetic."""

from __future__ import annotations

import json
import sys

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.application.system_workflow import CompletePlatformWorkflow

SCRIPT_SETTINGS = load_config("_utility_schemas")["run_complete_platform"]


def main(argv=None):

    args = load_script_settings("run_complete_platform", SCRIPT_SETTINGS, argv)
    try:
        if not 1 <= args.max_jobs <= 64 or not 1 <= args.max_iterations <= 8:
            raise ValueError(
                "finite bounds require 1..64 jobs and 1..8 iterations"
            )
        workflow = CompletePlatformWorkflow(
            args.state_dir,
            args.output_dir,
            run_id=args.run_id,
            policy_path=args.coordination_policy,
        )
        if args.mode == "demo":
            result = workflow.run_demo(
                max_jobs=args.max_jobs,
                max_iterations=args.max_iterations,
                year=args.fixture_year,
            )
        else:
            if not all((args.manifest, args.dataset_id, args.research_asof)):
                raise ValueError(
                    "research mode requires --manifest --dataset-id --research-asof"
                )
            result = workflow.run_research(
                args.manifest, args.dataset_id, args.research_asof
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED_CLOSED",
                    "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
