"""Run the four-charter research campaign on an operator-approved dataset (design partition only).

Real data (explicit manifest, dataset id and research clock are all required):

  python final_v2/scripts/run_strategy_research.py --manifest /path/manifest.json --dataset-id comp_chars_design \\
      --state-dir /tmp/campaign --research-asof 2026-10-25T00:00:00+00:00 [--output report.json]

Separate synthetic fixture demo (mechanics only; every result is labelled fixture and authorizes nothing):

  python final_v2/scripts/run_strategy_research.py --fixture-demo /tmp/fixture --state-dir /tmp/fixture_state

Hypotheses and specifications are written to <state-dir>/registry.sqlite BEFORE any data row is read; reruns are
resumable and spend no extra variant budget. The report never contains an admission certificate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pod_platform.research.builtins import write_fixture_dataset  # noqa: E402
from pod_platform.research.builtins.fixtures import FIXTURE_DATASET_ID  # noqa: E402
from pod_platform.strategies.charter_campaign import run_charter_campaign  # noqa: E402


def _parse_asof(text: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise SystemExit(f"--research-asof {text!r} is not an ISO timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SystemExit("--research-asof must include a timezone offset")
    return parsed.astimezone(dt.timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", help="operator dataset manifest JSON")
    parser.add_argument("--dataset-id", help="approved dataset id declared in the manifest")
    parser.add_argument("--state-dir", required=True, help="registry, trial ledger and campaign reports live here")
    parser.add_argument("--research-asof", help="ISO timestamp with offset; required for real data")
    parser.add_argument("--campaign-config", help="campaign config JSON (default configs/campaigns/charter_campaign_v1.json)")
    parser.add_argument("--charter", action="append", help="restrict to a pod id / charter (repeatable); default all four")
    parser.add_argument("--variant-budget", type=int, help="override the per-hypothesis variant budget")
    parser.add_argument("--output", help="also write the report JSON here")
    parser.add_argument("--summary", action="store_true", help="print a compact table to stderr")
    parser.add_argument("--fixture-demo", metavar="DIR", help="write a synthetic fixture dataset into DIR and run on it")
    return parser


def _summary(report: dict) -> str:
    lines = [f"campaign {report['campaign_id']}  mode={report['research_mode']}  data_class={report['data_class']}"]
    for cand in report["candidates"]:
        m = cand["metrics"]
        lines.append(
            f"  {cand['pod_id']:<12} {cand['status']:<12} belief_ic_cand={m.get('belief_ic_candidate', float('nan')):+.4f} "
            f"base={m.get('belief_ic_baseline', float('nan')):+.4f} "
            f"net_spread={m.get('bridge_net_spread_candidate', float('nan')):+.5f} "
            f"{'FAILED ' + cand['failure_notes'][0]['message'][:70] if cand['failure_notes'] else ''}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.fixture_demo:
            if args.manifest or args.dataset_id:
                raise SystemExit("--fixture-demo is a separate path; do not combine it with --manifest/--dataset-id")
            print("FIXTURE DEMO: synthetic data; results demonstrate mechanics only.", file=sys.stderr)
            manifest = write_fixture_dataset(args.fixture_demo)
            dataset_id = FIXTURE_DATASET_ID
            asof = _parse_asof(args.research_asof) if args.research_asof else dt.datetime.now(dt.timezone.utc)
        else:
            if not (args.manifest and args.dataset_id and args.research_asof):
                raise SystemExit("--manifest, --dataset-id and --research-asof are required (or use --fixture-demo)")
            manifest, dataset_id, asof = Path(args.manifest), args.dataset_id, _parse_asof(args.research_asof)
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
