"""Self-financing, fully costed paired replays and a first-order diagnostic."""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..agents.budgets import PersistentCallBudget
from ..agents.durable_cache import (
    CachedStructuredExtractor,
    DurableExtractionCache,
)
from ..agents.provider import GenerationParams, ResearchMode
from ..agents.registry import TemporalPolicy
from ..orchestration.replay import ControlledReplay, load_fixture, utc_close
from ..research.certificates import resolve_policy
from .common import (
    SYNTHETIC_LABEL,
    atomic_write_json,
    digest,
    now_utc,
    parse_aware,
    read_json,
)


@dataclass(frozen=True)
class TwinPolicy:
    max_abs_weight: float = 0.3
    exception_weight_multiplier: float = 0.5
    policy_id: str = "paired_exception_v1"
    concentration_trigger: float = 0.1

    def __post_init__(self):
        if (
            self.max_abs_weight <= 0
            or not 0 <= self.exception_weight_multiplier <= 1
            or self.concentration_trigger <= 0
        ):
            raise ValueError(
                "exception sizing must reduce or retain exposure within a positive cap"
            )


class ExceptionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["retain", "halve"]


class FixtureExceptionAgent:
    """Injectable test-only decision function. Never presented as a language model."""

    is_fixture = True

    def __init__(self, decide, *, identity="scripted_fixture_v1"):
        self.fn = decide
        self.fixture_identity = identity
        self.calls = 0

    def identity(self):
        return {
            "kind": "FIXTURE_STAND_IN_NOT_LLM",
            "identity": self.fixture_identity,
        }

    def decide(self, context):
        self.calls += 1
        return ExceptionDecision(action=self.fn(context)).action


class ProviderExceptionAgent:
    """Reviewed provider+durable cache+budget with adoption and temporal gates.

    No provider is constructed by this module. A caller must inject the actual
    provider, append-only cache, persistent budget and checkpoint registry.
    """

    is_fixture = False

    def __init__(
        self, provider, cache, budget, *, registry, role="lead", clock=None
    ):
        if not isinstance(cache, DurableExtractionCache) or not isinstance(
            budget, PersistentCallBudget
        ):
            raise ValueError(
                "model decisions require a durable cache and persistent call budget"
            )
        self.provider, self.cache, self.budget, self.registry, self.role = (
            provider,
            cache,
            budget,
            registry,
            role,
        )
        self.calls = 0
        self.records = []
        self.extractor = CachedStructuredExtractor(
            provider,
            cache,
            ExceptionDecision,
            "For a declared portfolio concentration exception return only action retain or halve. "
            "Use supplied as-of context only. Never propose orders, new securities, source access or rule overrides.",
            budget=budget,
            registry=registry,
            clock=clock or now_utc,
        )

    def identity(self):
        adoptions = self.registry.adoption_history()
        return {
            "kind": "ADOPTED_CHECKPOINT_PROVIDER",
            "provider": self.provider.name,
            "model": self.provider.model,
            "role": self.role,
            "prompt_hash": self.extractor.prompt_hash,
            "adoptions": [a.model_dump(mode="json") for a in adoptions],
            "checkpoints": [
                self.registry.checkpoint(k).model_dump(mode="json")
                for k in sorted({a.checkpoint_id for a in adoptions})
            ],
        }

    def decide(self, context):
        allowed = {
            "decision_at",
            "information_cutoff",
            "period",
            "security_id",
            "weight",
            "exception",
            "policy",
            "paired_input_hash",
        }
        if set(context) - allowed:
            raise ValueError("unexpected fields in agent context")
        at = parse_aware(context["decision_at"])
        if parse_aware(context["information_cutoff"]) > at:
            raise ValueError("future information in agent context")
        checkpoint_id = self.registry.route(self.role, at)
        checkpoint = self.registry.checkpoint(checkpoint_id)
        adoptions = [
            a
            for a in self.registry.adoption_history()
            if a.checkpoint_id == checkpoint_id
            and a.role == self.role
            and a.effective_start <= at
            and a.temporal_policy == TemporalPolicy.RELEASE_GATED
        ]
        if (
            not checkpoint.admitted
            or not checkpoint.verify_seal()
            or not adoptions
            or not all(
                a.verify_seal() and a.shadow_comparison_id for a in adoptions
            )
        ):
            raise ValueError(
                "model use requires sealed admitted checkpoint and fixed-task adoption evidence"
            )
        # Byte count is deliberately conservative for the verified context limit.
        if (
            len(str(context).encode())
            + len(self.extractor.system_prompt.encode())
            + 1024
            > checkpoint.context_limit_verified
        ):
            raise ValueError(
                "decision context exceeds verified model context limit"
            )
        run = self.extractor.run(
            context,
            role=self.role,
            namespace="execution",
            research_asof=at,
            mode=ResearchMode.HISTORICAL,
            checkpoint_id=checkpoint_id,
            params=GenerationParams(max_tokens=128),
        )
        self.calls += not run.from_cache
        self.records.append(
            {
                "checkpoint_id": checkpoint_id,
                "cache_key": run.cache_key,
                "from_cache": run.from_cache,
                "authorization": run.authorization.as_dict(),
                "provider_is_mock": run.provider_is_mock,
                "prompt_hash": self.extractor.prompt_hash,
            }
        )
        return run.value.action


def _nav_conserved(monthly, starting):
    nav = float(starting)
    for row in monthly:
        nav *= 1 + row["net_return"]
        if abs(nav - float(row["nav_end"])) > 1e-6 * max(1, abs(nav)):
            raise ValueError("paired NAV does not compound")
        if any(abs(float(v)) > 1e-6 for v in row["reconciliation"].values()):
            raise ValueError("paired NAV does not reconcile")
    return nav


def build_twin_exhibit(
    run_dir,
    fixture,
    policy=TwinPolicy(),
    agent=None,
    *,
    certificate_policy=None,
):
    fx = (
        load_fixture(Path(fixture))
        if isinstance(fixture, (str, Path))
        else fixture
    )
    if agent is not None and not isinstance(
        agent, (ProviderExceptionAgent, FixtureExceptionAgent)
    ):
        raise ValueError(
            "use an adopted ProviderExceptionAgent or explicitly labelled FixtureExceptionAgent"
        )
    root = Path(run_dir)
    certificate_policy = resolve_policy(
        certificate_policy or fx.config.get("certificate_policy")
    )
    # Hash every actual input object, not merely filenames. Both arms share this.
    paired_hash = digest(
        {
            "config": fx.config,
            "cards": {
                k: v.model_dump(mode="json") for k, v in fx.cards.items()
            },
            "certificates": {
                k: v.model_dump(mode="json") for k, v in fx.certificates.items()
            },
            "forecasts": [f.model_dump(mode="json") for f in fx.forecasts],
            "market": {
                str(day): {sid: row.__dict__ for sid, row in rows.items()}
                for day, rows in fx.market.items()
            },
            "outcomes": {str(k): v for k, v in fx.outcomes.items()},
            "periods": fx.periods,
            "liquidations": {
                str(k): {s: v.model_dump(mode="json") for s, v in val.items()}
                for k, val in fx.liquidations.items()
            },
            "policy": policy.__dict__,
            "certificate_policy": certificate_policy.name,
        }
    )
    interventions = []
    arms = {}

    def overlay(arm):
        def apply(weights, formation):
            result = dict(weights)
            at = utc_close(formation)
            cutoff = max(
                f.information_cutoff
                for f in fx.forecasts
                if f.earliest_formation_at <= at and not f.is_expired(at)
            )
            for sid, weight in sorted(weights.items()):
                if (
                    not weight
                    or abs(float(weight)) <= policy.concentration_trigger
                ):
                    continue
                fixed = "halve"
                context = {
                    "period": formation.isoformat(),
                    "decision_at": at.isoformat(),
                    "information_cutoff": cutoff.isoformat(),
                    "security_id": sid,
                    "weight": str(weight),
                    "exception": "name concentration",
                    "policy": policy.policy_id,
                    "paired_input_hash": paired_hash,
                }
                chosen = (
                    agent.decide(context) if arm == "agent" and agent else fixed
                )
                if chosen not in ("retain", "halve"):
                    raise ValueError(
                        "exception response outside reviewed choices"
                    )
                result[sid] = weight * (
                    Decimal(str(policy.exception_weight_multiplier))
                    if chosen == "halve"
                    else Decimal(1)
                )
                if abs(float(result[sid])) > policy.max_abs_weight:
                    raise ValueError("exception exceeds policy name limit")
                interventions.append(
                    {
                        **context,
                        "arm": arm,
                        "fixed_decision": fixed,
                        "chosen_decision": chosen,
                    }
                )
            return result

        return apply

    for arm in ("fixed", "agent"):
        folder = root / "paired_replays" / arm
        try:
            outcome = ControlledReplay(
                fx,
                folder,
                certificate_policy=certificate_policy,
                risk_overlay=overlay(arm),
            ).run()
            monthly = read_json(folder / "monthly.json")
            if {r["period"] for r in monthly} != {
                str(p[0]) for p in fx.periods
            }:
                raise ValueError("paired arm omitted a scheduled period")
            ending = _nav_conserved(monthly, fx.config["starting_nav"])
            arms[arm] = {
                "status": "COMPLETED",
                "ending_nav": ending,
                "monthly": monthly,
                "summary": outcome["summary"],
                "artifact_hashes": read_json(folder / "artifact_hashes.json"),
            }
        except BaseException as exc:
            # Process interruption propagates to the outer stage retry. Completed
            # or partial arm artifacts are quarantined by its declared ownership.
            if not isinstance(exc, Exception):
                raise
            arms[arm] = {
                "status": "FAILED_CLOSED",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "partial_periods": read_json(folder / "periods.partial.json")
                if (folder / "periods.partial.json").exists()
                else [],
                "failure_artifact": "paired_replays/" + arm + "/failure.json"
                if (folder / "failure.json").exists()
                else None,
            }
    complete = all(a["status"] == "COMPLETED" for a in arms.values())
    paired = []
    failures = []
    for i, (formation, end, _) in enumerate(fx.periods):
        row = {"period": str(formation), "period_end": str(end)}
        for arm in ("fixed", "agent"):
            periods = arms[arm].get("monthly", [])
            record = next(
                (r for r in periods if r["period"] == str(formation)), None
            )
            row[arm] = record
            if record and record["net_return"] < 0:
                failures.append(
                    {
                        "arm": arm,
                        "period": str(formation),
                        "net_return": record["net_return"],
                    }
                )
        row["nav_difference"] = (
            float(row["agent"]["nav_end"]) - float(row["fixed"]["nav_end"])
            if row["agent"] and row["fixed"]
            else None
        )
        paired.append(row)
    # Keep the actual failed-position example as a separate additive diagnostic.
    diagnostics = []
    with (root / "holdings_by_period.csv").open() as handle:
        for row in csv.DictReader(handle):
            day = dt.date.fromisoformat(row["period"])
            weight = float(row["post_trade_weight"])
            ret = fx.outcomes[day][row["security_id"]]
            if ret is None:
                continue
            diagnostics.append(
                {
                    "period": row["period"],
                    "security_id": row["security_id"],
                    "weight": weight,
                    "realized_excess_return": ret,
                    "agent_contribution": weight * float(ret),
                    "basis": "base-book first-order position contribution; full paired NAV is reported separately",
                }
            )
    result = {
        "schema_version": "paired_exhibit.v2",
        "status": "COMPLETED" if complete else "INCOMPLETE",
        "label": SYNTHETIC_LABEL
        if "synthetic_signal" in fx.cards
        else "OPERATOR_REPLAY",
        "agent_kind": "DETERMINISTIC_STAND_IN_NOT_LLM"
        if agent is None
        else agent.identity()["kind"],
        "agent_identity": agent.identity()
        if agent
        else {"kind": "DETERMINISTIC_STAND_IN_NOT_LLM"},
        "llm_skill_evidence": False,
        "paired_input_hash": paired_hash,
        "policy": policy.__dict__,
        "comparison": "two full self-financing ControlledReplay arms; identical market/source/forecast/cost inputs; path-dependent fills, cash, borrow, fees and risk gates included",
        "arms": arms,
        "periods": paired,
        "interventions": interventions,
        "provider_records": getattr(agent, "records", []),
        "paired_nav_difference": arms["agent"]["ending_nav"]
        - arms["fixed"]["ending_nav"]
        if complete
        else None,
        "failed_periods": failures,
        "rows": diagnostics,
        "failures": [r for r in diagnostics if r["agent_contribution"] < 0],
        "limitations": [
            "Synthetic or mock provider results cannot establish LLM skill.",
            "Both arms include all scheduled periods, including losses and unavailable outcomes.",
        ],
    }
    atomic_write_json(root / "twin_exhibit.json", result)
    return result
