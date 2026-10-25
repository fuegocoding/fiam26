"""Coordinated controlled replay utility: certified forecasts -> CIO coordination -> risk -> fills -> ledger.

Settings are read from the matching section of final_v2/config.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.contracts.errors import PodPlatformError
from pod_platform.orchestration.replay import (  # noqa: E402
    COORDINATION_MODE,
    ControlledReplay,
    coordination_hook_from_records,
    load_fixture,
)

ROOT = Path(__file__).resolve().parent.parent


SCRIPT_SETTINGS = load_config("_utility_schemas")["run_coordinated_replay"]


def main(argv=None) -> int:

    args = load_script_settings("run_coordinated_replay", SCRIPT_SETTINGS, argv)

    try:
        policy = json.loads(
            args.coordination_policy.read_text(encoding="utf-8")
        )
        risk_rows = json.loads(args.risk_inputs.read_text(encoding="utf-8"))
        hook = coordination_hook_from_records(policy, risk_rows)
        fixture = load_fixture(args.fixture)
        fixture.config["coordination"] = {
            "mode": COORDINATION_MODE,
            "policy_digest": hook.digest(),
        }
        runner = ControlledReplay(
            fixture,
            args.out,
            research_only=args.research_only,
            overwrite=args.overwrite,
            coordination=hook,
        )
        outcome = runner.run()
    except PodPlatformError as error:
        code = (
            error.response.error.code.value
            if error.response.error
            else "UNKNOWN"
        )
        print(
            f"FAILED CLOSED [{code}]: {error.response.summary}", file=sys.stderr
        )
        return 2
    except (
        OSError,
        ValueError,
        TypeError,
    ) as error:  # unreadable/invalid operator files
        print(
            f"FAILED CLOSED [OPERATOR_INPUT]: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    summary = outcome["summary"]
    print(
        json.dumps(
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
