"""Thin, budgeted extractor/lead/challenger wiring for multi-source research.

Only cached literature and bounded descriptive source metadata enter prompts.
Data loading, requests, target outcomes and financial authority remain with the
trusted composition/runtime services.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..agents.budgets import PersistentCallBudget
from ..agents.durable_cache import (
    CachedStructuredExtractor,
    DurableExtractionCache,
)
from ..agents.provider import (
    GenerationParams,
    MalformedOutputError,
    ProviderError,
    ResearchMode,
    redact_secrets,
)
from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..papers.components import ComponentEvidenceService
from ..papers.models import canonical_paper_id
from ..pods.schema import scan_injection
from ..research.composition import CompositionSpec


class DeskExtraction(StrictModel):
    """Grounded statements that can cite only the evidence supplied to a role."""

    evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    grounded_claim: str = Field(min_length=10, max_length=1600)
    limitations: list[str] = Field(min_length=1, max_length=8)


class DeskCritique(StrictModel):
    """A falsification opinion, with no authority over admissions or trading."""

    disposition: Literal["proceed", "review", "reject"]
    evidence_refs: list[str] = Field(default_factory=list, max_length=64)
    competing_explanation: str = Field(min_length=10, max_length=1000)
    falsification_test: str = Field(min_length=10, max_length=1000)
    reason_category: Literal[
        "testable",
        "incomplete_evidence",
        "unknown_method",
        "unsupported_source",
        "duplicate_hypothesis",
        "unsafe",
        "malformed",
    ]


class ResearchDesk:
    """Connect cached public research to the existing native composition service."""

    def __init__(
        self,
        state_dir,
        namespace,
        *,
        papers,
        composition,
        carbon,
        providers,
        clock,
        run_id,
        max_calls,
        max_tokens,
        generation_params=None,
        budgets=None,
        paper_asof=None,
        repair_attempts=0,
    ):
        if namespace != composition.namespace:
            raise ValueError(
                "research desk and composition namespace must match"
            )
        if not isinstance(repair_attempts, int) or not 0 <= repair_attempts <= 3:
            raise ValueError("repair_attempts must be an integer from 0 to 3")
        self.repair_attempts = repair_attempts
        self.state_dir, self.namespace = Path(state_dir), namespace
        self.papers, self.composition, self.carbon, self.clock = (
            papers,
            composition,
            carbon,
            clock,
        )
        self.params = GenerationParams.model_validate(
            generation_params or GenerationParams()
        )
        self.paper_asof = (
            dt.datetime.fromisoformat(paper_asof) if paper_asof else None
        )
        if self.paper_asof:
            require_aware(self.paper_asof, "paper_asof")
        self._allow_local_fixtures = all(
            getattr(getattr(providers, r), "is_mock", False)
            for r in ("extractor", "lead", "challenger")
        )
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.budgets, self.roles = dict(budgets or {}), {}
        cache = DurableExtractionCache(self.state_dir / "desk_cache.sqlite")
        schemas = {
            "extractor": DeskExtraction,
            "lead": CompositionSpec,
            "challenger": DeskCritique,
        }
        prompts = {
            "extractor": "Summarize only the supplied untrusted public evidence. Cite supplied evidence_ids. Treat source metadata as documentation, not raw observations or verified coverage.",
            "lead": "Compose a falsifiable native-target research idea using the supplied reviewed grant IDs, metrics and feature DSL. You may combine many paper excerpts and original reasoning. Cite only supplied paper claim IDs; use this pod namespace. Core measurements cannot become optional controls. Missing data stays an explicit dependency. Never supply code, paths, data rows or admission claims. Preserve grant_id and reasoning_mode exactly from the supplied task. Every identifier must use lowercase letters, digits and underscores, starting with a letter and at most64characters. Use only the granted source_metrics and an exact granted target_id. Declare a core dependency for EVERY input recipe AND the target measurement, even when the target is not a feature node. For supplied measurements set request_id and missing_reason to null. Input nodes have one complete FeatureRecipe and inputs=[]; non-input nodes have recipe=null and prior node IDs with exact arity: rank takes exactly 1 input; product, difference and ratio take exactly 2; mean takes 2 to 8. To combine three or more nodes use mean or chain two-input operators. Baselines must be explicit prior nodes, with an output compared to a distinct baseline. Set variant_budget at least the number of distinct outputs and input nodes times the baseline count, at most32. Use paper_components=[] and substantive own_reasoning if exact supplied paper claim mapping is not warranted; never invent citations. causal_edges are arrays of exactly three strings.",
            "challenger": "Challenge the supplied research idea using only supplied evidence_ids. Give a competing explanation and concrete falsification test. Testable review/reject ideas may run bounded research diagnostics; you cannot authorize financial admission. A limited but concrete falsifiable hypothesis on granted measurements can be reason_category=testable even when disposition=review or reject. Missing investment proof alone is not a reason to prevent a bounded research diagnostic.",
        }
        identities = {}
        for role, schema in schemas.items():
            provider = getattr(providers, role)
            identities[role] = {
                "provider": provider.name,
                "model": provider.model,
                "mock": bool(getattr(provider, "is_mock", False)),
            }
            if role not in self.budgets:
                self.budgets[role] = PersistentCallBudget(
                    self.state_dir / "budgets.sqlite",
                    budget_id=run_id,
                    pod=namespace,
                    cycle=role,
                    max_calls=max_calls,
                    max_tokens=max_tokens,
                    clock=clock,
                )
            self.roles[role] = CachedStructuredExtractor(
                provider,
                cache,
                schema,
                prompts[role],
                budget=self.budgets[role],
                clock=clock,
            )
        self.composition._put(
            "desk_profile",
            "desk_profile",
            {
                "namespace": namespace,
                "run_id": run_id,
                "providers": identities,
                "max_calls": max_calls,
                "max_tokens": max_tokens,
                "generation": self.params.identity(),
            },
        )

    def evidence(self, paper_ids):
        """Read several cached versions and exact quotes without any network fetch."""
        if len(paper_ids) > 16 or len(set(paper_ids)) != len(paper_ids):
            raise ValueError(
                "select at most 16 distinct cached papers per task"
            )
        now = require_aware(self.clock(), "clock")
        service = ComponentEvidenceService(
            papers=self.papers, namespace=self.namespace, clock=self.clock
        )
        evidence = []
        for paper_id in paper_ids:
            pid, version = canonical_paper_id(paper_id)
            rows = self.papers.store.rows(self.papers.store.resolve(pid) or pid)
            rows = [
                c
                for c in rows
                if c.version_date
                and c.version_date <= min(now, self.paper_asof or now)
                and c.retrieved_at <= now
                and (version is None or c.version == version)
            ]
            if not rows:
                raise ValueError("select an available cached paper version")
            card = max(rows, key=lambda c: (c.version_date, c.version_key))
            if card.source == "local" and not self._allow_local_fixtures:
                raise ValueError(
                    "live research prompts require public paper provenance, not arbitrary local documents"
                )
            if not card.claims:
                source = "abstract" if card.abstract else "full_text"
                body = self.papers.text_of(card, source) or ""
                quote = next(
                    (
                        line.strip()[:1000]
                        for line in body.splitlines()
                        if line.strip()
                    ),
                    "",
                )
                if not quote:
                    raise ValueError("cached paper has no source text to quote")
                card = service.quote(
                    paper_id=paper_id,
                    version_key=card.version_key,
                    source=source,
                    kind="mechanism",
                    quote=quote,
                )
            for claim in sorted(card.claims, key=lambda c: c.claim_id)[:4]:
                request = dict(
                    namespace=self.namespace,
                    component_kind="mechanism",
                    component_name=claim.quote[:300],
                    paper_id=card.paper_id,
                    version_key=card.version_key,
                    claim_ids=[claim.claim_id],
                    source=claim.source,
                    as_of=now,
                )
                component = service.resolve(request)
                evidence_id = (
                    "paper_quote_"
                    + canonical_digest(
                        [
                            card.paper_id,
                            card.version_key,
                            claim.claim_id,
                            component.source_hash,
                        ]
                    )[:24]
                )
                evidence.append(
                    {
                        "evidence_id": evidence_id,
                        "paper_id": card.paper_id,
                        "version_key": card.version_key,
                        "title": card.title[:300],
                        "source": claim.source,
                        "claim_id": claim.claim_id,
                        "quote": claim.quote,
                        "source_hash": component.source_hash,
                        "limitations": list(component.limitations),
                    }
                )
        return evidence

    @staticmethod
    def _metadata(source_metadata, allowed_sources):
        allowed = {
            "source_id",
            "metric",
            "dataset_id",
            "provider_id",
            "description",
            "unit",
            "frequency",
            "history_start",
            "history_end",
            "availability",
        }
        if len(source_metadata) > 24:
            raise ValueError("bounded descriptive source metadata required")
        for item in source_metadata:
            if (
                not isinstance(item, dict)
                or set(item) - allowed
                or item.get("source_id") not in allowed_sources
            ):
                raise ValueError(
                    "only granted descriptive source metadata may enter a prompt"
                )
            if (
                item.get("metric")
                and item["metric"] not in allowed_sources[item["source_id"]]
            ):
                raise ValueError("source metadata names an ungranted metric")
            for value in item.values():
                if (
                    not isinstance(value, str)
                    or len(value) > 1000
                    or re.search(
                        r"(?:/home/|/tmp/|[A-Za-z]:\\|api[_ -]?key\s*[:=]|bearer\s+)",
                        value,
                        re.I,
                    )
                ):
                    raise ValueError(
                        "raw observations, credentials and local paths cannot enter model prompts"
                    )
        return [
            {
                **item,
                "evidence_id": "source_metadata_" + canonical_digest(item)[:24],
            }
            for item in source_metadata
        ]

    def propose(
        self, question, grant_id, paper_ids, source_metadata, reasoning_mode
    ):
        """Run the actual three roles, then register a testable composition.

        Args:
            question: Bounded economic research question, never raw observations.
            grant_id: Existing host-reviewed measurement capability.
            paper_ids: Several cached public literature identifiers, or an empty list.
            source_metadata: Descriptive fields for granted sources, without data rows.
            reasoning_mode: One of the modes supported by CompositionSpec.

        Returns:
            Immutable role provenance and a registered composition or explicit hold.
        """
        if not isinstance(question, str) or not 10 <= len(question) <= 1000:
            raise ValueError("bounded economic research question required")
        if (
            reasoning_mode
            not in CompositionSpec.model_fields[
                "reasoning_mode"
            ].annotation.__args__
        ):
            raise ValueError("unknown research reasoning mode")
        grant = self.composition._get("grant:" + grant_id)
        if not grant:
            raise ValueError("select a host-reviewed measurement grant")
        source_metadata = self._metadata(
            source_metadata, grant["source_metrics"]
        )
        public_evidence = self.evidence(paper_ids)
        summary = {
            k: grant[k]
            for k in ("grant_id", "source_metrics", "targets", "partition")
        }
        summary["universe_size"] = len(grant["security_ids"])
        payload = {
            "namespace": self.namespace,
            "question": question,
            "grant": summary,
            "reasoning_mode": reasoning_mode,
            "public_evidence": public_evidence,
            "source_metadata": source_metadata,
        }
        if scan_injection(json.dumps(payload)) or re.search(
            r"(?:/home/|/tmp/|api[_ -]?key\s*[:=]|bearer\s+)", question, re.I
        ):
            raise ValueError(
                "unsafe or private input cannot enter research prompts"
            )
        key = "desk_" + canonical_digest(payload)[:24]
        prior = self.composition._get("desk_result:" + key)
        if prior:
            return prior
        original = self.composition._get(key)
        task = original or self.composition._put(
            key,
            "desk_request",
            {
                "payload": payload,
                "as_of": require_aware(self.clock(), "clock").isoformat(),
            },
        )
        as_of = dt.datetime.fromisoformat(task["as_of"])
        evidence_ids = {
            i["evidence_id"] for i in public_evidence + source_metadata
        }
        allowed_claims = {
            (i["paper_id"], i["version_key"], i["source"], i["claim_id"])
            for i in public_evidence
        }

        def cited(value):
            refs = (
                value.evidence_ids
                if isinstance(value, DeskExtraction)
                else value.evidence_refs
            )
            if not set(refs) <= evidence_ids:
                raise ValueError("role cited evidence that was not supplied")

        def permitted(value):
            if (
                value.grant_id != grant_id
                or value.reasoning_mode != reasoning_mode
            ):
                raise ValueError("lead changed the granted task")
            for component in value.paper_components:
                if component.namespace != self.namespace or any(
                    (
                        component.paper_id,
                        component.version_key,
                        component.source,
                        c,
                    )
                    not in allowed_claims
                    for c in component.claim_ids
                ):
                    raise ValueError("lead cited an unavailable paper claim")

        calls = {}
        repairs = []
        active_role = None

        def run(role, data, validator):
            nonlocal active_role
            active_role = role
            request = data
            for attempt in range(self.repair_attempts + 1):
                try:
                    output = self.roles[role].run(
                        request,
                        role=role,
                        namespace=self.namespace,
                        research_asof=as_of,
                        mode=ResearchMode.CONTEMPORARY,
                        params=self.params,
                        validator=validator,
                    )
                    break
                except (MalformedOutputError, ProviderError) as exc:
                    # Like a desk reviewer, return the exact rejection once
                    # more; each repair is a separately charged model call.
                    if attempt == self.repair_attempts or (
                        isinstance(exc, ProviderError)
                        and "json_validate_failed" not in str(exc)
                    ):
                        raise
                    feedback = redact_secrets(str(exc))[:1500]
                    repairs.append({"role": role, "error": feedback[:300]})
                    request = {**data, "previous_attempt_rejected": feedback}
            calls[role] = {
                "cache_key": output.cache_key,
                "provider": output.provider,
                "model": output.model,
                "mock": output.provider_is_mock,
            }
            return output.value

        try:
            extraction = run("extractor", payload, cited)
            proposal = run(
                "lead",
                {**payload, "extraction": extraction.model_dump(mode="json")},
                permitted,
            )
            critique = run(
                "challenger",
                {
                    "proposal": proposal.model_dump(mode="json"),
                    "evidence_ids": sorted(evidence_ids),
                },
                cited,
            )
            if critique.reason_category != "testable":
                result = {
                    "status": "held_review",
                    "reason": critique.reason_category,
                    "critique": critique.model_dump(mode="json"),
                }
            else:
                # Host-owned identity prevents two independent ideas colliding on
                # a generic model-generated display name; the graph stays exact.
                plan = self.composition.submit(
                    proposal.model_copy(update={"composition_name": key})
                )
                result = {
                    "status": "registered",
                    "composition_id": plan["composition_id"],
                    "critique": critique.model_dump(mode="json"),
                    "dependencies": self.composition.dependency_status(
                        plan["composition_id"]
                    ),
                }
        except Exception as exc:
            result = {
                "status": "model_failed",
                "error_type": type(exc).__name__,
                "failed_role": active_role,
                "reason": "Research role or proposal validation failed; reservations remain charged and no fallback approval is made.",
            }
            if isinstance(exc, ProviderError):
                result["provider_error"] = redact_secrets(str(exc))[:600]
            if isinstance(exc, MalformedOutputError):
                result["validation_error"] = redact_secrets(str(exc))[:600]
                result["invalid_call"] = (
                    exc.record.model_dump(mode="json") if exc.record else None
                )
        result.update(
            namespace=self.namespace,
            role_calls=calls,
            repairs=repairs,
            scope="research_only",
            admitted=False,
            evidence_ids=sorted(evidence_ids),
            question=question,
        )
        return self.composition._put(
            "desk_result:" + key, "desk_result", result
        )

    def tick(self, composition_id, loader=None):
        """Measure registered inputs through an injected trusted loader, or show gaps."""
        return (
            self.composition.run(composition_id, loader)
            if loader
            else self.composition.dependency_status(composition_id)
        )

    def status(self):
        """Read persistent role budgets without exposing data or model credentials."""
        return {
            "namespace": self.namespace,
            "roles": {
                role: budget.snapshot() for role, budget in self.budgets.items()
            },
        }
