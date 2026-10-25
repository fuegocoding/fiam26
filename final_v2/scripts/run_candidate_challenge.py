#!/usr/bin/env python3
"""Trusted-host utility: freeze one measured design candidate, issue an integrity-owned check authorization, run the challenge.

Settings are read from the matching section of final_v2/config.json.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from pod_platform.application.configuration import load_config
from pod_platform.application.integrity import (  # noqa: E402
    IntegrityPolicy,
    IntegrityService,
)
from pod_platform.application.namespaces import RoleContext
from pod_platform.application.script_settings import load_script_settings
from pod_platform.research.builtins import BuiltinContext
from pod_platform.research.challenge_bridge import (  # noqa: E402
    ChallengeBridge,
    CheckAuthority,
)
from pod_platform.research.evaluation import RegisteredEvaluator
from pod_platform.research.registration import ResearchLedger
from pod_platform.research.registry import Registry


def _dt(text: str) -> dt.datetime:
    value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError(f"{text!r} must carry a timezone")
    return value


SCRIPT_SETTINGS = load_config("_utility_schemas")["run_candidate_challenge"]


def main(argv=None) -> int:

    a = load_script_settings("run_candidate_challenge", SCRIPT_SETTINGS, argv)

    state = Path(a.state_dir)
    context = BuiltinContext(
        a.manifest,
        state,
        clock=lambda: a.evaluation_asof,
        research_asof=a.registered_at,
    )
    registry = Registry(state / "challenge_registry.sqlite")
    evaluator = RegisteredEvaluator(state / "challenge_evaluation.sqlite")
    # Host-owned key: created once as a private 0600 file (directory is git-ignored), reloaded on restart; never printed.
    authority = CheckAuthority.load_or_init(
        a.authority_key or state / "private" / "check_authority.key"
    )
    bridge = ChallengeBridge(state, authority=authority)
    try:
        ctx = RoleContext(
            "integrity",
            "integrity",
            "challenge_cli",
            a.evaluation_asof,
            ("design", "check"),
        )
        policy = IntegrityPolicy(
            policy_id="challenge_cli_v1", frozen_at=a.policy_frozen_at
        )
        frozen = bridge.freeze_candidate(
            ctx,
            context,
            a.registration_id,
            a.variant,
            policy=policy,
            registered_at=a.registered_at,
            multiplicity=a.multiplicity,
        )
        service = IntegrityService(
            registry=registry,
            ledger=ResearchLedger(registry),
            evaluator=evaluator,
            policy=policy,
            source_cards=[bridge.integrity_source_card(context, frozen)],
            evidence_seal=authority.evidence_seal(),
            allow_unbound_fixture=False,
        )
        auth = authority.issue(
            ctx,
            frozen,
            check_start=a.check_start,
            check_end=a.check_end,
            issued_at=a.policy_frozen_at,
            expires_at=a.evaluation_asof + dt.timedelta(days=1),
            max_family_accesses=a.family_budget,
        )
        result = bridge.run_challenge(
            ctx,
            context,
            service,
            frozen.candidate_id,
            auth,
            evaluation_asof=a.evaluation_asof,
        )
        print(
            json.dumps(
                {
                    "candidate_id": frozen.candidate_id,
                    "scope": frozen.scope,
                    "check_opened": result.check_opened,
                    "decision": result.decision.as_dict(),
                    "hidden": result.hidden,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        bridge.close()
        evaluator.close()
        registry.close()
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
