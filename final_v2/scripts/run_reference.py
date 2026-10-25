#!/usr/bin/env python3
"""Offline reference workflow; fixture evidence never authorizes an investment book."""

from __future__ import annotations

import json
import sys

from pod_platform.application.configuration import load_config
from pod_platform.application.reference import ReferenceWorkflow
from pod_platform.application.script_settings import load_script_settings

SCRIPT_SETTINGS = load_config("_utility_schemas")["run_reference"]


def main(argv=None):

    args = load_script_settings("run_reference", SCRIPT_SETTINGS, argv)
    try:
        workflow = ReferenceWorkflow(
            args.state_dir, args.output_dir, run_id=args.run_id
        )
        if args.mode == "demo":
            result = workflow.run_demo()
        elif args.mode == "research":
            if not all((args.manifest, args.dataset_id, args.research_asof)):
                raise ValueError(
                    "research requires --manifest --dataset-id --research-asof"
                )
            result = workflow.run_research(
                args.manifest, args.dataset_id, args.research_asof, args.data
            )
        else:
            if not args.fixture:
                raise ValueError("replay requires --fixture")
            result = workflow.run_replay(args.fixture, args.label)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps({"status": "FAILED_CLOSED", "error": str(exc)[:1000]}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
