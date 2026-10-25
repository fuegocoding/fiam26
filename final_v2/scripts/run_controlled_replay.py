"""Controlled replay CLI: approved forecasts -> independent risk -> fills -> ledger -> compliance -> exports.

Consumes an explicit fixture directory (config.json, source_cards.json,
certificates.json, forecasts.json, market.csv, periods.csv; optional
liquidations.json and lineage.json). No LLM is called, nothing is trained, and
a frozen status is reported only when every measured prerequisite holds.

Monthly inputs only: no daily-path realism is claimed (see run_summary.json).

Usage:
    python final_v2/scripts/run_controlled_replay.py --fixture DIR --out DIR
        [--research-only] [--overwrite]

Exit codes: 0 run completed (see freeze_status in run_summary.json),
            2 failed closed (failure.json written to --out).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pod_platform.contracts.errors import PodPlatformError  # noqa: E402
from pod_platform.orchestration.replay import ControlledReplay, load_fixture  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="fresh output directory")
    parser.add_argument(
        "--research-only",
        action="store_true",
        help="allow sources that are not formation-eligible; labeled, never freezable",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty --out")
    args = parser.parse_args(argv)

    try:
        fixture = load_fixture(args.fixture)
        runner = ControlledReplay(fixture, args.out, research_only=args.research_only, overwrite=args.overwrite)
        outcome = runner.run()
    except PodPlatformError as error:
        code = error.response.error.code.value if error.response.error else "UNKNOWN"
        print(f"FAILED CLOSED [{code}]: {error.response.summary}", file=sys.stderr)
        return 2
    summary = outcome["summary"]
    print(json.dumps(
        {
            "run_id": summary["run_id"],
            "labels": summary["labels"],
            "run_label": summary["source_policy"]["run_label"],
            "periods": summary["periods"],
            "ending_nav": summary["ending_nav"],
            "freeze_status": summary["freeze_status"],
            "freeze_reasons": summary["freeze_reasons"],
            "llm_called": summary["llm_called"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
