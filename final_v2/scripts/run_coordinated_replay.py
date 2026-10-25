"""Coordinated controlled replay CLI: certified forecasts -> CIO coordination -> risk -> fills -> ledger.

Runs a controlled-replay fixture directory through the CIO coordination layer (covariance sizing,
duplicate-family de-duplication, one netted hedge per exposure dimension) under a FIXED policy file and
EXPLICIT dated risk inputs. Before each decision only risk rows observed at or before that decision are
used; future rows are never read. A coordination failure stops the run (exit 2); it never falls back to
the legacy trading path. No LLM is called and nothing is fitted.

Usage:
    python final_v2/scripts/run_coordinated_replay.py --fixture DIR --out DIR
        --coordination-policy policy.json --risk-inputs risk.json [--research-only] [--overwrite]

policy.json keys (all optional): coordination_policy, covariance_policy (field overrides),
    hedge_allowances [{security_id, max_abs_weight, shortable}],
    hedge_requests [{requester, security_id, dimension, max_abs_weight}],
    declarations {strategy_id: {family_label, event_ids, measurement_ids, exposure_ids}}.
risk.json: [{"observed_at": ISO-8601 with offset, "returns": {security_id: period_return}}, ...].
The fixture config is stamped with ``coordination: {mode, policy_digest}`` so the declaration and the
supplied policy/risk history are pinned together.

Exit codes: 0 run completed, 2 failed closed (failure.json and coordination_receipts.partial.json in --out).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pod_platform.contracts.errors import PodPlatformError  # noqa: E402
from pod_platform.orchestration.replay import (  # noqa: E402
    COORDINATION_MODE,
    ControlledReplay,
    coordination_hook_from_records,
    load_fixture,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="fresh output directory")
    parser.add_argument("--coordination-policy", required=True, type=Path)
    parser.add_argument("--risk-inputs", required=True, type=Path)
    parser.add_argument("--research-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty --out")
    args = parser.parse_args(argv)

    try:
        policy = json.loads(args.coordination_policy.read_text(encoding="utf-8"))
        risk_rows = json.loads(args.risk_inputs.read_text(encoding="utf-8"))
        hook = coordination_hook_from_records(policy, risk_rows)
        fixture = load_fixture(args.fixture)
        fixture.config["coordination"] = {"mode": COORDINATION_MODE, "policy_digest": hook.digest()}
        runner = ControlledReplay(
            fixture, args.out, research_only=args.research_only, overwrite=args.overwrite, coordination=hook
        )
        outcome = runner.run()
    except PodPlatformError as error:
        code = error.response.error.code.value if error.response.error else "UNKNOWN"
        print(f"FAILED CLOSED [{code}]: {error.response.summary}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as error:  # unreadable/invalid operator files
        print(f"FAILED CLOSED [OPERATOR_INPUT]: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    summary = outcome["summary"]
    print(json.dumps(
        {
            "run_id": summary["run_id"],
            "run_label": summary["source_policy"]["run_label"],
            "periods": summary["periods"],
            "ending_nav": summary["ending_nav"],
            "freeze_status": summary["freeze_status"],
            "freeze_reasons": summary["freeze_reasons"],
            "coordination": summary["coordination"],
            "llm_called": summary["llm_called"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
