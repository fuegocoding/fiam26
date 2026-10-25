"""Run the four-charter research campaign on an operator-approved dataset (design partition only).

Settings are read from the matching section of final_v2/config.json.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.research.builtins import write_fixture_dataset
from pod_platform.research.builtins.fixtures import (
    FIXTURE_DATASET_ID,  # noqa: E402
)
from pod_platform.strategies.charter_campaign import (
    run_charter_campaign,  # noqa: E402
)

ROOT = Path(__file__).resolve().parent.parent


def _parse_asof(text: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise SystemExit(
            f"--research-asof {text!r} is not an ISO timestamp"
        ) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SystemExit("--research-asof must include a timezone offset")
    return parsed.astimezone(dt.timezone.utc)


def _summary(report: dict) -> str:
    lines = [
        f"campaign {report['campaign_id']}  mode={report['research_mode']}  data_class={report['data_class']}"
    ]
    for cand in report["candidates"]:
        m = cand["metrics"]
        lines.append(
            f"  {cand['pod_id']:<12} {cand['status']:<12} belief_ic_cand={m.get('belief_ic_candidate', float('nan')):+.4f} "
            f"base={m.get('belief_ic_baseline', float('nan')):+.4f} "
            f"net_spread={m.get('bridge_net_spread_candidate', float('nan')):+.5f} "
            f"{'FAILED ' + cand['failure_notes'][0]['message'][:70] if cand['failure_notes'] else ''}"
        )
    return "\n".join(lines)


SCRIPT_SETTINGS = load_config("_utility_schemas")["run_strategy_research"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = load_script_settings("run_strategy_research", SCRIPT_SETTINGS, argv)
    try:
        if args.fixture_demo:
            if args.manifest or args.dataset_id:
                raise SystemExit(
                    "--fixture-demo is a separate path; do not combine it with --manifest/--dataset-id"
                )
            print(
                "FIXTURE DEMO: synthetic data; results demonstrate mechanics only.",
                file=sys.stderr,
            )
            manifest = write_fixture_dataset(args.fixture_demo)
            dataset_id = FIXTURE_DATASET_ID
            asof = (
                _parse_asof(args.research_asof)
                if args.research_asof
                else dt.datetime.now(dt.timezone.utc)
            )
        else:
            if not (args.manifest and args.dataset_id and args.research_asof):
                raise SystemExit(
                    "--manifest, --dataset-id and --research-asof are required (or use --fixture-demo)"
                )
            manifest, dataset_id, asof = (
                Path(args.manifest),
                args.dataset_id,
                _parse_asof(args.research_asof),
            )
        report = run_charter_campaign(
            manifest,
            args.state_dir,
            dataset_id=dataset_id,
            research_asof=asof,
            campaign_config=args.campaign_config,
            charters=args.charter,
            variant_budget=args.variant_budget,
        )
    except (ValueError, KeyError, PermissionError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.summary:
        print(_summary(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
