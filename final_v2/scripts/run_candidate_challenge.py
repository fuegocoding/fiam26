#!/usr/bin/env python3
"""Trusted-host CLI: freeze one measured design candidate, issue an integrity-owned check authorization, run the challenge.

Run by the operator/host only (never by a pod): it holds the CheckAuthority key and the integrity RoleContext. Requires a
persistent --state-dir that already holds the built-in executor ledger and candidate artifacts of a design run.
Fixture manifests yield fixture-scoped results; real proxy/unsupported-target recipes stop at research SHADOW.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pod_platform.application.integrity import IntegrityPolicy, IntegrityService  # noqa: E402
from pod_platform.application.namespaces import RoleContext  # noqa: E402
from pod_platform.research.builtins import BuiltinContext  # noqa: E402
from pod_platform.research.challenge_bridge import ChallengeBridge, CheckAuthority  # noqa: E402
from pod_platform.research.evaluation import RegisteredEvaluator  # noqa: E402
from pod_platform.research.registration import ResearchLedger  # noqa: E402
from pod_platform.research.registry import Registry  # noqa: E402


def _dt(text: str) -> dt.datetime:
    value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise argparse.ArgumentTypeError(f"{text!r} must carry a timezone")
    return value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--state-dir", required=True)
    p.add_argument("--registration-id", required=True, help="namespace:draft_entry_id of the built-in executor run")
    p.add_argument("--variant", required=True)
    p.add_argument("--registered-at", type=_dt, required=True)
    p.add_argument("--policy-frozen-at", type=_dt, required=True)
    p.add_argument("--evaluation-asof", type=_dt, required=True)
    p.add_argument("--check-start", type=dt.date.fromisoformat, required=True)
    p.add_argument("--check-end", type=dt.date.fromisoformat, required=True)
    p.add_argument("--family-budget", type=int, default=1)
    p.add_argument("--multiplicity", type=int, required=True)
    p.add_argument("--authority-key", default=None, help="private 0600 key file (default: <state-dir>/private/check_authority.key)")
    a = p.parse_args(argv)

    state = Path(a.state_dir)
    context = BuiltinContext(a.manifest, state, clock=lambda: a.evaluation_asof, research_asof=a.registered_at)
    registry = Registry(state / "challenge_registry.sqlite")
    evaluator = RegisteredEvaluator(state / "challenge_evaluation.sqlite")
    # Host-owned key: created once as a private 0600 file (directory is git-ignored), reloaded on restart; never printed.
    authority = CheckAuthority.load_or_init(a.authority_key or state / "private" / "check_authority.key")
    bridge = ChallengeBridge(state, authority=authority)
    try:
        ctx = RoleContext("integrity", "integrity", "challenge_cli", a.evaluation_asof, ("design", "check"))
        policy = IntegrityPolicy(policy_id="challenge_cli_v1", frozen_at=a.policy_frozen_at)
        frozen = bridge.freeze_candidate(ctx, context, a.registration_id, a.variant, policy=policy,
                                         registered_at=a.registered_at, multiplicity=a.multiplicity)
        service = IntegrityService(registry=registry, ledger=ResearchLedger(registry), evaluator=evaluator, policy=policy,
                                   source_cards=[bridge.integrity_source_card(context, frozen)],
                                   evidence_seal=authority.evidence_seal(), allow_unbound_fixture=False)
        auth = authority.issue(ctx, frozen, check_start=a.check_start, check_end=a.check_end, issued_at=a.policy_frozen_at,
                               expires_at=a.evaluation_asof + dt.timedelta(days=1), max_family_accesses=a.family_budget)
        result = bridge.run_challenge(ctx, context, service, frozen.candidate_id, auth, evaluation_asof=a.evaluation_asof)
        print(json.dumps({"candidate_id": frozen.candidate_id, "scope": frozen.scope, "check_opened": result.check_opened,
                          "decision": result.decision.as_dict(), "hidden": result.hidden}, indent=2, sort_keys=True))
        return 0
    finally:
        bridge.close()
        evaluator.close()
        registry.close()
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
