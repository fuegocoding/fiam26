#!/usr/bin/env python3
"""Finite offline source-to-book platform run; all demo economics are synthetic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pod_platform.application.system_workflow import CompletePlatformWorkflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "research"), default="demo")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", default="complete_fixture_v1")
    parser.add_argument("--coordination-policy", type=Path)
    parser.add_argument("--max-jobs", default=16, type=int)
    parser.add_argument("--max-iterations", default=4, type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument("--research-asof")
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.max_jobs <= 64 or not 1 <= args.max_iterations <= 8:
            raise ValueError("finite bounds require 1..64 jobs and 1..8 iterations")
        workflow = CompletePlatformWorkflow(args.state_dir, args.output_dir, run_id=args.run_id,
                                            policy_path=args.coordination_policy)
        if args.mode == "demo":
            result = workflow.run_demo(max_jobs=args.max_jobs, max_iterations=args.max_iterations)
        else:
            if not all((args.manifest, args.dataset_id, args.research_asof)):
                raise ValueError("research mode requires --manifest --dataset-id --research-asof")
            result = workflow.run_research(args.manifest, args.dataset_id, args.research_asof)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error": f"{type(exc).__name__}: {str(exc)[:1000]}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
