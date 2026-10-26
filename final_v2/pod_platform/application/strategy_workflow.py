"""Configuration-driven research, provider acquisition and portfolio transport.

This module connects existing services. Models never supply data loaders,
credentials, financial reviews or portfolio policy.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from ..agents.provider import (
    ClaudeCodeProvider,
    OpenAICompatibleProvider,
    ProviderError,
)
from ..agents.strategy_models import ServingProfile, StrategyModelProvider
from ..contracts.base import canonical_digest
from ..contracts.sources import SourceCard
from ..data.carbonarc_requests import (
    CarbonArcOAuthTransport,
    CarbonArcPolicy,
    CarbonArcRequests,
)
from ..orchestration.replay import (
    COORDINATION_MODE,
    ControlledReplay,
    coordination_hook_from_records,
    load_fixture,
)
from ..papers import PaperService, PaperStore
from ..papers.models import PapersError
from ..papers.transport import UrllibTransport
from ..research.composition import FeatureObservation, observation_digest
from ..research.composition_bridge import CompositionBridge
from ..research.evaluation import RegisteredEvaluator
from ..research.promotion import AdmissionCertificate
from .common import atomic_write_json, file_sha256, now_utc
from .configuration import PROJECT_ROOT, config_path, load_config
from .desk_evaluation import evaluate_training, evaluate_validation
from .edge_diagnostics import run_edge_diagnostics
from .edge_research import run_edge_research
from .integrity import IntegrityPolicy, IntegrityService
from .live_models import create_live_models
from .market_run import finish_research_run
from .namespaces import RoleContext
from .publication import PublicationService
from .research_campaign import run_research_campaign
from .research_desk import ResearchDesk
from .research_inputs import prepare_inputs
from .research_loop import NAMESPACES, AutonomousResearchLoop, RoleProviders
from .system_workflow import CompletePlatformWorkflow
from .training_lab import run_training_lab


class UnconfiguredProvider:
    """Represent a disabled backend without substituting a mock or model."""

    name = "unconfigured"
    is_mock = False

    def __init__(self, role):
        self.model = role

    def complete(self, request):
        """Refuse inference until the root configuration enables a backend."""
        raise ProviderError(
            "Configure and enable the model role before inference."
        )


def configured_providers(settings):
    """Build the three explicitly configured model roles without calling them."""
    providers = {}
    for role in ("lead", "challenger", "extractor"):
        row = settings["roles"][role]
        if not settings.get("online"):
            providers[role] = UnconfiguredProvider(role)
        elif row["provider"] == "strategy":
            providers[role] = StrategyModelProvider(
                ServingProfile.model_validate(row["serving_profile"])
            )
        elif row["provider"] == "claude-code":
            providers[role] = ClaudeCodeProvider(model=row["model"])
        elif row["provider"] == "openai":
            providers[role] = OpenAICompatibleProvider(
                row["base_url"], row["model"], row.get("api_key_env")
            )
        else:
            raise ValueError(f"Unknown model provider for {role}.")
    return RoleProviders(**providers)


class StrategyWorkflow:
    """Run the configured task graph with durable data and research services."""

    def __init__(
        self,
        settings=None,
        *,
        providers=None,
        carbon_transport=None,
        clock=now_utc,
    ):
        self.settings = settings or load_config()
        self.input_snapshot = None
        if self.settings["research"].get("prepare_inputs"):
            prepared = prepare_inputs(self.settings, clock)
            for key in (
                "measurement_grants",
                "observation_sources",
                "proposals",
            ):
                self.settings["research"][key] = prepared[key]
            self.settings["application"]["manifest"] = prepared["manifest_path"]
            self.input_snapshot = prepared["snapshot"]
        self.application = self.settings["application"]
        self.clock = clock
        self.state = config_path(self.application["state_dir"])
        self.output = config_path(self.application["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        carbon = self.settings["carbon"]
        self.carbon = CarbonArcRequests(
            self.state / "carbon",
            transport=carbon_transport
            or CarbonArcOAuthTransport(
                PROJECT_ROOT, online=carbon["allow_remote"]
            ),
            policy=CarbonArcPolicy.model_validate(
                {
                    k: v
                    for k, v in carbon.items()
                    if k not in {"capabilities", "requests"}
                }
            ),
            clock=clock,
        )
        for capability in carbon["capabilities"]:
            self.carbon.register_capability(capability)
        papers = PaperService(
            PaperStore(config_path(self.application["papers_db"])),
            clock=clock,
            online=self.application["online_papers"],
            transport=(
                UrllibTransport() if self.application["online_papers"] else None
            ),
        )
        self.global_budget = None
        if providers:
            self.providers = providers
        elif "global_max_calls" in self.settings["models"]:
            self.providers, self.global_budget = create_live_models(
                self.settings["models"],
                self.state,
                self.application["run_id"],
                clock,
            )
        else:
            self.providers = configured_providers(self.settings["models"])
        periods = self.settings["research"].get("periods", {})
        self.loop = AutonomousResearchLoop(
            self.state / "research",
            providers=self.providers,
            papers=papers,
            clock=clock,
            manifest_path=config_path(self.application["manifest"]),
            run_id=self.application["run_id"],
            carbon_requests=self.carbon,
            max_role_calls=min(
                r["max_calls"]
                for r in self.settings["models"]["roles"].values()
            ),
            max_role_tokens=min(
                r["max_tokens"]
                for r in self.settings["models"]["roles"].values()
            ),
            check_start=dt.date.fromisoformat(
                periods.get("validation_start", "2019-01-01")
            ),
            check_end=dt.date.fromisoformat(
                periods.get("validation_end", "2020-12-31")
            ),
        )
        self.loop.publication = PublicationService(
            self.state / "formation_publications.sqlite",
            registry=self.loop.registry,
            store_scope="formation",
        )
        self.desks = {}
        for namespace in NAMESPACES:
            limits = self.settings["models"]["roles"]
            self.desks[namespace] = ResearchDesk(
                self.state / "desks" / namespace,
                namespace,
                papers=papers,
                composition=self.loop.compositions[namespace],
                carbon=self.carbon,
                providers=self.providers,
                clock=clock,
                run_id=self.application["run_id"],
                max_calls=min(r["max_calls"] for r in limits.values()),
                max_tokens=min(r["max_tokens"] for r in limits.values()),
                budgets={
                    role: self.loop.budgets[namespace, role] for role in limits
                },
                generation_params=self.settings["models"].get("generation"),
                paper_asof=self.settings["research"].get("literature_asof"),
                repair_attempts=self.settings["models"].get(
                    "repair_attempts", 0
                ),
            )
        self.loop.research_desks = self.desks
        self.loop.composition_loader_factory = self.prepare_composition
        self.loop.attach_research_source = self.attach_source
        for grant in self.settings["research"]["measurement_grants"]:
            self.loop.review_composition(
                grant["namespace"],
                {**grant, "code_digest": self.loop.profile["code"]},
            )

    def attach_source(
        self, namespace, grant_id, request_id, security_id, metric
    ):
        """Attach a real requested source for research, without source admission.

        The security association is a proposed research join. Its unresolved
        source card prevents it from acquiring formation authority.
        """
        service = self.loop.compositions[namespace]
        grant = next(
            g for g in service.status()["grants"] if g["grant_id"] == grant_id
        )
        if security_id not in grant["security_ids"]:
            raise ValueError(
                "The proposed stock must belong to this research universe."
            )
        receipt = self.carbon.read(namespace, request_id)
        if receipt["status"] != "data_ready":
            raise ValueError(
                "Request actual observations before attaching a source."
            )
        card = SourceCard.model_validate_json(
            Path(receipt["source_card_path"]).read_text()
        )
        if card.formation_eligible or card.broker_rejects_formation() is False:
            raise ValueError(
                "Automatic attachment accepts only unresolved research sources."
            )
        key = canonical_digest(
            [namespace, grant_id, request_id, security_id, metric]
        )
        extended = {
            **grant,
            "grant_id": "requested_" + key[:24],
            "source_metrics": {
                **grant["source_metrics"],
                receipt["source_id"]: [metric],
            },
            "source_cards": [
                c
                for c in grant["source_cards"]
                if c["source_id"] != card.source_id
            ]
            + [card.model_dump(mode="json")],
        }
        reviewed = self.loop.review_composition(namespace, extended)
        binding = {
            "namespace": namespace,
            "kind": "carbon",
            "source_id": card.source_id,
            "metric": metric,
            "security_id": security_id,
            "request_id": request_id,
            "grant_id": reviewed["grant_id"],
        }
        self.loop._event(
            "requested_source:" + key, namespace, "requested_source", binding
        )
        return {
            "grant_id": reviewed["grant_id"],
            "source_id": card.source_id,
            "metric": metric,
            "request_id": request_id,
            "research_only": True,
            "formation_eligible": False,
            "join_status": "unverified_research_mapping",
        }

    def prepare_composition(self, namespace, composition_id):
        """Bind configured files and requested data to this exact registered graph."""
        service = self.loop.compositions[namespace]
        plan = service.get_plan(composition_id)
        bindings = [
            b
            for b in self.settings["research"]["observation_sources"]
            if b["source_id"] in plan["grant"]["source_metrics"]
        ]
        bindings.extend(
            b
            for b in self.loop._events("requested_source", namespace)
            if b["grant_id"] == plan["grant"]["grant_id"]
        )
        if not bindings:
            return None
        snapshots = {}
        for binding in bindings:
            snapshots[binding["source_id"]] = (
                observation_digest(self._carbon_observations(binding))
                if binding.get("kind") == "carbon"
                else binding["observation_digest"]
            )
        service.bind_inputs(composition_id, snapshots)
        return lambda: self._load_observations(bindings)

    def _load_observations(self, bindings):
        """Load only host-declared, hash-pinned normalized observations."""
        observations = []
        for binding in bindings:
            if binding.get("kind") == "carbon":
                observations.extend(self._carbon_observations(binding))
                continue
            path = config_path(binding["path"])
            if file_sha256(path) != binding["sha256"]:
                raise ValueError("Observation file changed after host review.")
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows:
                observation = FeatureObservation.model_validate(row)
                if observation.source_id != binding["source_id"]:
                    raise ValueError(
                        "Observation source differs from the host binding."
                    )
                observations.append(observation)
        return observations

    def _carbon_observations(self, binding):
        """Normalize an actual receipt without inventing historical knowledge."""
        receipt = self.carbon.read(binding["namespace"], binding["request_id"])
        if (
            receipt["status"] != "data_ready"
            or receipt["source_id"] != binding["source_id"]
        ):
            raise ValueError(
                "Alternative-data dependency has no matching ready receipt."
            )
        path = Path(receipt["data_path"])
        if file_sha256(path) != receipt["data_sha256"]:
            raise ValueError("Alternative-data table changed after receipt.")
        received = dt.datetime.fromisoformat(receipt["retrieved_at"])
        observations = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not row["value"]:
                    continue
                day = dt.date.fromisoformat(row["date"])
                period = dt.datetime.combine(day, dt.time.max, dt.timezone.utc)
                # A vendor release column alone does not verify historical vintages.
                known = max(period, received)
                observations.append(
                    FeatureObservation(
                        observation_id="carbon_"
                        + canonical_digest([receipt["request_id"], row])[:32],
                        security_id=binding["security_id"],
                        source_id=binding["source_id"],
                        metric=binding["metric"],
                        period_end=period,
                        available_at=known,
                        version_known_at=known,
                        value=float(row["value"]),
                    )
                )
        return observations

    def run(self):
        """Execute configured requests, research and eligible portfolio stages."""
        research = self.settings["research"]
        result = {
            "mode": "configured_strategy_workflow",
            "data_requests": [],
            "research": [],
            "model_roles_enabled": self.settings["models"]["online"],
            "orders": 0,
            "input_snapshot": self.input_snapshot,
            "literature": [],
        }
        discovered = {}
        literature_asof = research.get("literature_asof")
        for query in self.application["paper_queries"]:
            namespace = (
                query.get("namespace") if isinstance(query, dict) else None
            )
            text = query["query"] if isinstance(query, dict) else query
            query_key = "literature:" + canonical_digest(
                [
                    namespace,
                    text,
                    literature_asof,
                    self.application["papers_per_query"],
                ]
            )
            previous = self.loop._get(query_key)
            if previous is not None:
                discovered.setdefault(namespace, []).extend(
                    previous.get("paper_ids", [])
                )
                result["literature"].append(previous)
                continue
            try:
                found = self.loop.papers.search(
                    text,
                    limit=self.application["papers_per_query"],
                    as_of=literature_asof,
                )
                identifiers = [
                    c.paper_id + (f"v{c.version}" if c.version else "")
                    for c in found.cards
                    if c.abstract and c.arxiv_id and c.version
                ]
                discovered.setdefault(namespace, []).extend(identifiers)
                result["literature"].append(
                    {
                        "query": text,
                        "origin": found.origin,
                        "paper_ids": identifiers,
                        "as_of": literature_asof,
                    }
                )
            except PapersError as exc:
                result["literature"].append(
                    {
                        "query": text,
                        "status": "source_unavailable",
                        "error_type": type(exc).__name__,
                    }
                )
            self.loop._event(
                query_key, "host", "literature", result["literature"][-1]
            )
        for paper_id in self.application["paper_ids"]:
            self.loop.papers.ingest_full_text(paper_id)
        for request in self.settings["carbon"]["requests"]:
            result["data_requests"].append(
                self.carbon.request(request["namespace"], request["request"])
            )
        for grant in research["measurement_grants"]:
            reviewed = {**grant, "code_digest": self.loop.profile["code"]}
            self.loop.review_composition(reviewed["namespace"], reviewed)
        for task in research.get("proposals", []):
            namespace = task["namespace"]
            selected = list(
                dict.fromkeys(
                    task["paper_ids"]
                    + discovered.get(namespace, [])
                    + discovered.get(None, [])
                )
            )[:16]
            try:
                proposal = self.desks[namespace].propose(
                    **{
                        k: v
                        for k, v in task.items()
                        if k not in {"namespace", "paper_ids"}
                    },
                    paper_ids=selected,
                )
            except (ValueError, PapersError) as exc:
                proposal = {
                    "namespace": namespace,
                    "status": "preparation_failed",
                    "error_type": type(exc).__name__,
                    "admitted": False,
                }
            result["research"].append(proposal)
        plans = []
        for task in research["compositions"]:
            plans.append(
                self.loop.submit_composition(task["namespace"], task["spec"])
            )
        for namespace, service in self.loop.compositions.items():
            plans.extend(
                service.get_plan(p["composition_id"])
                for p in service.status()["compositions"]
            )
        unique = {p["composition_id"]: p for p in plans}
        for plan in unique.values():
            namespace, identifier = plan["namespace"], plan["composition_id"]
            result["research"].append(
                self.loop.run_composition(namespace, identifier)
            )
        if research.get("desk_evaluation", {}).get("enabled"):
            result["desk_evaluation"] = evaluate_training(
                self, list(unique.values())
            )
            result["validation"] = evaluate_validation(
                self, list(unique.values()), result["desk_evaluation"]
            )
        result["financial_reviews"] = self.challenge_compositions(
            research["financial_reviews"]
        )
        result["portfolio"] = self.run_portfolio()
        result["orders"] = result["portfolio"].get("orders", 0)
        result["status"] = (
            "completed"
            if unique
            else "completed_no_candidates"
            if research.get("proposals")
            else "needs_research_configuration"
        )
        result["temporal_mode"] = "contemporary_research_on_historical_training"
        result["investment_validation"] = "UNPROVEN"
        if self.global_budget:
            result["model_budget"] = self.global_budget.snapshot()
        atomic_write_json(self.output / "workflow.json", result)
        return result

    def challenge_compositions(self, tasks):
        """Submit exact frozen graphs to independent integrity and publication."""
        results = []
        for task in tasks:
            namespace = task["namespace"]
            composition = self.loop.compositions[namespace]
            plan = composition.get_plan(task["review"]["composition_id"])
            policy = IntegrityPolicy.model_validate(task["policy"])
            evaluator = RegisteredEvaluator(
                self.state / (namespace + "_integrity.sqlite")
            )
            ctx = RoleContext(
                "integrity",
                "integrity",
                self.application["run_id"],
                self.clock(),
                allowed_partitions=("design", "check"),
            )
            try:
                integrity = IntegrityService(
                    registry=self.loop.registry,
                    ledger=self.loop.ledger,
                    evaluator=evaluator,
                    policy=policy,
                    source_cards=[
                        SourceCard.model_validate(c)
                        for c in plan["grant"]["source_cards"]
                    ],
                    clock=self.clock,
                    evidence_seal=self.loop.authority.evidence_seal(),
                    allow_unbound_fixture=False,
                )
                bridge = CompositionBridge(
                    composition, integrity, ctx=ctx, clock=self.clock
                )
                frozen = bridge.freeze(task["review"])
                candidate_id = frozen["candidate_id"]
                result = bridge.challenge(
                    candidate_id,
                    design_loader=lambda: self._load_observations(
                        task["design_sources"]
                    ),
                    check_loader=lambda: self._load_observations(
                        task["check_sources"]
                    ),
                )
                if result.get("admitted") and task.get("publication"):
                    publication = task["publication"]
                    batch = bridge.publish(
                        candidate_id,
                        information_cutoff=dt.datetime.fromisoformat(
                            publication["information_cutoff"]
                        ),
                        observations=self._load_observations(
                            publication["sources"]
                        ),
                        publication=self.loop.publication,
                        artifacts=self.loop.artifacts,
                        published_at=self.clock(),
                        ctx=RoleContext(
                            "strategy",
                            namespace,
                            self.application["run_id"],
                            self.clock(),
                        ),
                    )
                    result = {
                        **result,
                        "publication": batch.model_dump(mode="json"),
                    }
                results.append(result)
            finally:
                evaluator.close()
        return results

    def run_portfolio(self):
        """Use actual supplied inputs with the existing CIO, risk and ledger."""
        settings = self.settings["portfolio"]
        if not settings.get("input_dir"):
            if settings.get("cash_run", {}).get("enabled"):
                published = self.loop.publication.export_forecasts(
                    RoleContext(
                        "allocation",
                        "allocation",
                        self.application["run_id"],
                        self.clock(),
                    )
                )
                return finish_research_run(
                    settings["cash_run"], self.output, published
                )
            return {
                "status": "waiting_for_admitted_forecasts_and_market_inputs",
                "orders": 0,
            }
        inputs = load_fixture(config_path(settings["input_dir"]))
        if inputs.config.get("input_scope") != "actual_market":
            raise ValueError(
                "Production transport requires an actual-market input manifest."
            )
        if settings["certificate_policy"] != "production":
            raise ValueError(
                "Actual-market transport requires production certificate policy."
            )
        published = self.loop.publication.export_forecasts(
            RoleContext(
                "allocation",
                "allocation",
                self.application["run_id"],
                self.clock(),
            )
        )
        if published:
            inputs.forecasts = published
            inputs.certificates = {
                f.admission_certificate_id: AdmissionCertificate.model_validate(
                    self.loop.registry.get(
                        "admission_certificates", f.admission_certificate_id
                    )
                )
                for f in published
            }
        policy = settings["coordination_policy"]
        risk_path = config_path(settings["risk_inputs"])
        if file_sha256(risk_path) != settings["risk_inputs_sha256"]:
            raise ValueError("Risk history changed after host review.")
        hook = coordination_hook_from_records(
            policy, json.loads(risk_path.read_text())
        )
        inputs.config["coordination"] = {
            "mode": COORDINATION_MODE,
            "policy_digest": hook.digest(),
        }
        result = ControlledReplay(
            inputs,
            self.output / "portfolio",
            coordination=hook,
            certificate_policy="production",
        ).run()
        return {
            "status": "replayed_actual_inputs",
            "summary": result["summary"],
        }


def run_configured():
    """Run the root configuration without parsing command-line arguments."""
    settings = load_config()
    application = settings["application"]
    if application["mode"] == "research_campaign":
        return run_research_campaign(settings)
    if application["mode"] == "edge_diagnostics":
        return run_edge_diagnostics(settings)
    if application["mode"] == "edge_training":
        return run_edge_research(settings)
    if application["mode"] == "training":
        return run_training_lab(settings)
    if application["mode"] == "demo":
        return CompletePlatformWorkflow(
            config_path(application["state_dir"]),
            config_path(application["output_dir"]),
            run_id=application["run_id"],
        ).run_demo()
    return StrategyWorkflow(settings).run()
