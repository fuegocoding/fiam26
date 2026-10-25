"""Pod-scoped shared tool service. Run with python -m pod_platform.services.app.

No tool can choose its caller namespace, run a shell command, read an arbitrary
path, or admit a forecast. Each process is attached to one strategy charter.
Public paper metadata is shared; jobs, comparisons and memory stay per pod.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..contracts.base import StrictModel, require_aware
from ..contracts.sources import SourceCard
from ..research.evaluation import ComparisonPlan, ForecastOutcome, RegisteredEvaluator
from ..research.features import FeatureObservation, FeatureRecipe, materialize
from ..strategies.base import Charter
from .mcp import MCPServer, Tool


class Empty(StrictModel):
    pass


class CatalogueQuery(StrictModel):
    query: str = Field(min_length=2, max_length=200)
    limit: int = Field(default=5, ge=1, le=10)


class CatalogueRead(StrictModel):
    source_id: str = Field(min_length=1, max_length=100)


class DiscoverQuery(CatalogueQuery):
    provider_id: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')


class DiscoveredSource(StrictModel):
    query_fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')
    provider_ref: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')


class SourceSample(DiscoveredSource):
    max_rows: int = Field(default=5, ge=1, le=20)


class SourceProposal(DiscoveredSource):
    rationale: str = Field(min_length=20, max_length=1000)


class Search(StrictModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=50)
    as_of: dt.datetime | None = None


class Paper(StrictModel):
    paper_id: str = Field(min_length=1, max_length=300)
    as_of: dt.datetime | None = None


class PaperRead(Paper):
    max_chars: int = Field(default=20000, ge=100, le=100000)


class Replication(Paper):
    requested_label: Literal["EXACT_REPLICATION", "CLOSE_REPLICATION", "MECHANISM_TRANSFER",
                             "METHOD_TRANSFER", "INCOMPLETE"] = "INCOMPLETE"


class Submit(StrictModel):
    """kind 'propose' payload: ProposePayload; kind 'experiment' payload: ExperimentPayload."""
    kind: Literal["propose", "experiment"]
    payload: dict
    idempotency_key: str | None = Field(default=None, max_length=200)


class ProposePayload(StrictModel):
    # Evidence is looked up server-side from ids; callers cannot supply card bodies.
    question: str = Field(min_length=1, max_length=2000)
    executors: list[str] = Field(min_length=1, max_length=8)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    source_ids: list[str] = Field(default_factory=list, max_length=20)
    mode: str = "contemporary"
    research_asof: dt.datetime | None = None
    checkpoint_id: str | None = Field(default=None, max_length=200)
    policy: str = "release_gated"
    max_attempts: int = Field(default=2, ge=1, le=5)


class ExperimentPayload(StrictModel):
    draft_entry_id: str = Field(min_length=1, max_length=200)
    variants: list[str] = Field(default_factory=list, max_length=8)
    max_attempts: int = Field(default=2, ge=1, le=5)


class Jobs(StrictModel):
    state: Literal["queued", "running", "succeeded", "failed"] | None = None
    limit: int = Field(default=50, ge=1, le=100)


class Job(StrictModel):
    job_id: str = Field(min_length=1, max_length=200)


class Register(StrictModel):
    plan: ComparisonPlan


class Evaluate(StrictModel):
    experiment_id: str = Field(min_length=1, max_length=200)
    variant_id: str = Field(min_length=1, max_length=200)
    observations: list[ForecastOutcome] = Field(min_length=1, max_length=10000)
    evaluation_asof: dt.datetime


class Features(StrictModel):
    recipe: FeatureRecipe
    observations: list[FeatureObservation] = Field(max_length=10000)
    formation_at: dt.datetime
    security_ids: list[str] = Field(min_length=1, max_length=5000)


class MemoryList(StrictModel):
    limit: int = Field(default=50, ge=1, le=200)


class MemoryNote(StrictModel):
    content: dict
    idempotency_key: str = Field(min_length=1, max_length=200)


def _job_json(job) -> dict:
    payload = dict(job.payload)
    for name in ("paper_cards", "dataset_cards", "source_cards"):  # keep listings small
        if name in payload:
            payload[name] = [c.get("evidence_id") or c.get("paper_id") or c.get("source_id")
                             for c in payload[name]]
    out = {key: getattr(job, key) for key in (
        "job_id", "namespace", "kind", "state", "attempts", "max_attempts", "error", "created_at", "updated_at")}
    return {**out, "payload": payload}


def _papers_guard(fn):
    """Paper/queue errors become tool errors with a bounded message (no stack/credential leakage)."""
    def wrapped(args):
        from ..papers.models import PapersError
        from ..research.jobs import QueueError
        try:
            return fn(args)
        except (PapersError, QueueError) as exc:
            raise ValueError(f"{type(exc).__name__}: {str(exc)[:300]}") from None
    return wrapped


def _build_strategy_server(state_dir: str | Path, pod: Charter | str, *, online: bool = False,
                 source_cards_dir: str | Path | None = None, clock=None,
                 papers_db: str | Path | None = None, transport=None) -> MCPServer:
    """State layout matches scripts/run_pod_research.py: queue.sqlite and memory/<namespace>.sqlite
    are shared with the CLI worker; every MCP read/write is pinned to this process's namespace."""
    # Imports are lazy so the transport itself has no optional data/ML dependencies.
    from ..agents.provider import ResearchMode
    from ..agents.registry import TemporalPolicy
    from ..papers.service import PaperService
    from ..papers.transport import UrllibTransport
    from ..papers.watches import paper_evidence
    from ..pods.charters import pod_spec
    from ..pods.schema import normalise_evidence
    from ..research.jobs import ResearchQueue
    from ..research.memory import PodMemory

    spec = pod_spec(pod)
    pod = spec.charter
    namespace = spec.namespace
    clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
    state = Path(state_dir).resolve()
    private = state / "pods" / namespace
    private.mkdir(parents=True, exist_ok=True)
    shared = state / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    cards_dir = Path(source_cards_dir) if source_cards_dir else Path(__file__).parents[2] / "configs/source_cards"
    cards = [SourceCard.model_validate_json(p.read_text()) for p in sorted(cards_dir.glob("*.json"))]
    cards_by_id = {card.source_id: card for card in cards}
    papers = PaperService(Path(papers_db) if papers_db else shared / "papers.sqlite", clock=clock,
                          online=online, transport=(transport or UrllibTransport()) if online else None,
                          openalex_api_key=os.environ.get("OPENALEX_API_KEY"),
                          openalex_mailto=os.environ.get("OPENALEX_MAILTO"))
    queue = ResearchQueue(state / "queue.sqlite", clock=clock)
    evaluator = RegisteredEvaluator(private / "experiments.sqlite")
    memory = PodMemory(state / "memory", namespace, clock=clock)

    def paper_read(args):
        card = papers.get(args.paper_id, as_of=args.as_of)
        text = papers.text_of(card)
        return {"paper": card.model_dump(mode="json"),
                "text": text[:args.max_chars] if text else None,
                "truncated": bool(text and len(text) > args.max_chars),
                "limitations": [] if text else ["No cached full text; metadata alone cannot establish the method."]}

    def own_job(job_id):
        job = queue.get(job_id)
        if job is None or job.namespace != namespace:
            raise KeyError("job not found in this pod")
        return job

    def get_job(args):
        job = own_job(args.job_id)
        return {"job": _job_json(job), "result": queue.result(job.job_id),
                "events": queue.events(job.job_id)}

    def build_propose(p: ProposePayload) -> dict:
        mode, policy = ResearchMode(p.mode), TemporalPolicy(p.policy)
        asof = require_aware(p.research_asof or clock(), "research_asof")
        if asof > clock():
            raise ValueError("research_asof cannot lie in the future")
        if mode == ResearchMode.HISTORICAL and not p.checkpoint_id:
            raise ValueError("historical research requires a registered checkpoint_id")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", e) for e in p.executors):
            raise ValueError("executor ids must match [a-z][a-z0-9_]{0,63}")
        if not (p.paper_ids or p.source_ids):
            raise ValueError("at least one paper_id or source_id is required as evidence")
        unknown = [s for s in p.source_ids if s not in cards_by_id]
        if unknown:
            raise ValueError(f"unknown source ids: {unknown}")
        historical = mode == ResearchMode.HISTORICAL
        paper_cards = [paper_evidence(papers.get(pid, as_of=asof if historical else None)) for pid in p.paper_ids]
        source_cards = [{"source_id": s, "title": cards_by_id[s].topic,
                         "description": (f"dataset {cards_by_id[s].dataset_id}; status {cards_by_id[s].status.value}; "
                                         f"formation_eligible={cards_by_id[s].formation_eligible}"),
                         "limitations": cards_by_id[s].unresolved_questions} for s in p.source_ids]
        normalise_evidence(paper_cards, "paper_card")  # reject malformed evidence before it is queued
        normalise_evidence(source_cards, "source_card")
        return {"question": p.question, "research_asof": asof.isoformat(), "mode": mode.value,
                "policy": policy.value, "checkpoint_id": p.checkpoint_id,
                "executors": sorted(set(p.executors)), "paper_cards": paper_cards, "dataset_cards": [],
                "source_cards": source_cards, "prior_work": memory.prior_work_summaries()}

    def submit(args):
        if args.kind == "propose":
            p = ProposePayload.model_validate(args.payload)
            kind, payload, attempts = "propose", build_propose(p), p.max_attempts
        else:
            p = ExperimentPayload.model_validate(args.payload)
            draft = memory.get(p.draft_entry_id)
            if draft is None or draft.kind != "draft_proposal":
                raise KeyError("draft proposal not found in this pod's memory")
            planned = {v["name"] for v in draft.content["proposal"]["experiment_plan"]["variants"]}
            if set(p.variants) - planned:
                raise ValueError(f"variants not in the stored plan: {sorted(set(p.variants) - planned)}")
            kind, payload, attempts = "experiment", {"draft_entry_id": p.draft_entry_id,
                                                    "variants": sorted(set(p.variants))}, p.max_attempts
        return _job_json(queue.submit(namespace, kind, payload, idempotency_key=args.idempotency_key,
                                      max_attempts=attempts))

    def register(args):
        if args.plan.owner != namespace:
            raise PermissionError("experiment owner must match the server pod")
        if args.plan.registered_at > clock():
            raise ValueError("registration cannot lie in the future")
        return {"experiment_id": args.plan.experiment_id,
                "plan_hash": evaluator.register(args.plan), "admitted": False}

    def evaluate(args):
        if require_aware(args.evaluation_asof, "evaluation_asof") > clock():
            raise ValueError("evaluation clock cannot lie in the future")
        return evaluator.evaluate(args.experiment_id, args.variant_id, args.observations,
                                  owner=namespace, evaluation_asof=args.evaluation_asof,
                                  source_cards=cards)

    tools = [
        Tool("platform.status", "Show this process's fixed identity and capabilities", Empty,
             lambda args: {"pod": pod.value, "namespace": namespace, "online_papers": online,
                           "live_llm": False, "automatic_admission": False,
                           "execution": "No shell or trading tools are exposed."}),
        Tool("sources.list", "Inspect source eligibility and unresolved questions", Empty,
             lambda args: [card.model_dump(mode="json") for card in cards]),
        Tool("papers.search", "Search academic metadata; cached/offline unless enabled at startup", Search,
             _papers_guard(lambda args: papers.search(args.query, as_of=args.as_of, limit=args.limit))),
        Tool("papers.get", "Get version-aware paper metadata", Paper,
             _papers_guard(lambda args: papers.get(args.paper_id, as_of=args.as_of))),
        Tool("papers.read", "Read cached paper text with provenance; never invent missing content", PaperRead,
             _papers_guard(paper_read)),
        Tool("papers.ingest", "Fetch permitted full text for an identified paper; requires online startup", Paper,
             _papers_guard(lambda args: papers.ingest_full_text(args.paper_id, as_of=args.as_of)), False),
        Tool("papers.replication", "Create a source-grounded replication/adaptation specification", Replication,
             _papers_guard(lambda args: papers.create_replication_spec(
                 args.paper_id, requested_label=args.requested_label, as_of=args.as_of)), False),
        Tool("research.submit", "Queue a 'propose' (paper_ids/source_ids evidence) or 'experiment' job for this pod; "
                                "a CLI worker runs it, no model is called here", Submit,
             _papers_guard(submit), False),
        Tool("research.list", "List only this pod's jobs", Jobs,
             lambda args: [_job_json(job) for job in queue.list(namespace=namespace,
                                                              state=args.state, limit=args.limit)]),
        Tool("research.show", "Read a job and its actual result in this pod", Job, get_job),
        Tool("memory.list", "Read only this pod's persistent research memory", MemoryList,
             lambda args: [asdict(entry) for entry in memory.list(limit=args.limit)]),
        Tool("memory.note", "Append a private research note; cannot assert an evaluated result", MemoryNote,
             lambda args: asdict(memory.append("note", args.content,
                                              idempotency_key=args.idempotency_key)), False),
        Tool("experiments.register", "Register an immutable forecast-comparison plan", Register, register, False),
        Tool("experiments.evaluate", "Evaluate mature supplied forecasts against baselines; never admits models", Evaluate,
             evaluate, False),
        Tool("features.materialize", "Evaluate only reviewed operators with as-of revision filtering", Features,
             lambda args: materialize(args.recipe, args.observations,
                                      formation_at=args.formation_at, security_ids=args.security_ids)),
    ]
    server = MCPServer(tools, name=f"fiam-{pod.value}")
    # The stdio launcher closes persistent handles on EOF/interrupt.
    server.close = lambda: (papers.store.close(), evaluator.close())
    return server


# Application read facade. Identity is fixed at startup; arguments contain only
# bounded ids and dates. Output paths are registered by the trusted workflow.
class RunRead(StrictModel):
    run_id: str = Field(min_length=1,max_length=100,pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
class ReportRead(RunRead):
    report: Literal['summary','performance','compliance','judge','acceptance','daily','attribution','twin'] = 'summary'
class DossierRead(RunRead):
    security_id: str = Field(min_length=1,max_length=100,pattern=r'^[A-Za-z0-9_.-]+$')
    on_date: dt.date
class MethodRead(StrictModel):
    method_id: str = Field(min_length=1,max_length=64)


def build_server(state_dir, pod=None, *, role='strategy', online=False, source_cards_dir=None,
                 clock=None,papers_db=None,transport=None, discovery=None, research_runtime=None):
    from ..application.namespaces import RoleContext, RoleGateway
    from ..application.methods import default_registry
    from ..application.common import read_json, safe_id
    from ..application.decision_store import DecisionStore
    from ..application.publication import PublicationStore
    from ..research.memory import PodMemory
    from ..pods.charters import pod_spec
    allowed=('strategy','research','allocation','risk','execution','reporting')
    if role not in allowed:raise ValueError('unknown startup role')
    if role=='strategy' and pod is None:raise ValueError('strategy startup requires --pod')
    spec=pod_spec(pod) if role=='strategy' else None
    namespace=spec.namespace if spec else role
    clock=clock or (lambda: dt.datetime.now(dt.timezone.utc))
    ctx=RoleContext(role,namespace,'mcp',clock())
    gateway=RoleGateway();state=Path(state_dir).resolve()
    server=(_build_strategy_server(state,pod,online=online,source_cards_dir=source_cards_dir,clock=clock,
        papers_db=papers_db,transport=transport) if role=='strategy' else MCPServer([],name=f'fiam-{role}'))
    if role!='strategy':server.close=lambda:None
    methods=default_registry()
    def runs():
        gateway.authorize(ctx,'runs.read')
        index=state/'application/reference_index.json'
        return read_json(index) if index.is_file() else {}
    def root(run_id):
        safe_id(run_id,'run id')
        record=runs().get(run_id)
        if not record:raise KeyError('run not registered')
        return Path(record['output_dir'])
    def read_run(args,name):
        path=root(args.run_id)/name
        if not path.is_file():raise KeyError('artifact unavailable')
        if path.stat().st_size>2_000_000:raise ValueError('artifact exceeds bounded tool size; use operator report')
        return read_json(path)
    def candidates(args):
        result=read_run(args,'candidates.json')
        return [r for r in result if not spec or r.get('charter')==spec.charter.value][:100]
    def reports(args):
        if role not in ('reporting','research','allocation','risk'):raise PermissionError('role cannot read reports')
        names={'summary':'reference_summary.json','performance':'book/performance_report.json',
            'compliance':'book/compliance_report.json','judge':'judge_package.json','acceptance':'acceptance.json',
            'daily':'book/daily_report.json','attribution':'book/attribution_report.json','twin':'book/twin_exhibit.json'}
        return read_run(args,names[args.report])
    def dossier(args):
        if role not in ('reporting','research','allocation','risk'):raise PermissionError('role cannot query position dossier')
        root(args.run_id)
        return DecisionStore(state/'decision_store.sqlite').explain(args.security_id,args.on_date,args.run_id)
    def publications(args):
        if role=='strategy':gateway.authorize(ctx,'publication.read_own',namespace)
        else:gateway.authorize(ctx,'publication.read_all')
        return PublicationStore(state/'publications.sqlite').batches([namespace] if role=='strategy' else None)[:100]
    def note(args):
        gateway.authorize(ctx,'notes.append_role')
        if len(json.dumps(args.content))>8000:raise ValueError('note exceeds 8000 characters')
        return asdict(PodMemory(state/'memory',namespace,clock=clock).append('note',args.content,idempotency_key=args.idempotency_key))
    extra=[Tool('methods.list','List reviewed methods with actual implementations',Empty,lambda args:[c.model_dump(mode='json') for c in methods.list()]),
        Tool('methods.describe','Describe a reviewed method',MethodRead,lambda args:methods.describe(args.method_id)),
        Tool('runs.list','List bounded run summaries',Empty,lambda args:[{'run_id':key,**value.get('summary',{})} for key,value in sorted(runs().items())][-100:]),
        Tool('runs.show','Show a run summary',RunRead,lambda args:read_run(args,'reference_summary.json')),
        Tool('experiments.status','Read registered challenge outcomes visible to this role',RunRead,candidates),
        Tool('candidates.list','Read visible candidate decisions',RunRead,candidates),
        Tool('admission.status','Read admission status; cannot issue certificates',RunRead,lambda args:[
            {k:r.get(k) for k in ('candidate_id','outcome','scope','certificate_id','reasons')} for r in candidates(args)]),
        Tool('publications.list','Read published forecasts visible to this role',Empty,publications),
        Tool('reports.show','Read a named bounded report',ReportRead,reports),
        Tool('dossier.lookup','Explain actual position on a supplied date',DossierRead,dossier),
        Tool('notes.append','Append a bounded private role note',MemoryNote,note,False)]
    for tool in extra:
        if tool.name in server.tools:raise ValueError('duplicate tool')
        server.tools[tool.name]=tool
    server.tools['platform.status']=Tool('platform.status','Fixed process identity',Empty,lambda args:{
        'role':role,'namespace':namespace,'pod':spec.charter.value if spec else None,'live_llm':False,'automatic_admission':False,
        'online_papers':online,'execution':'No shell or trading tools are exposed.'})
    if discovery is None:
        from ..data.discovery import DiscoveryService, CatalogueMeta
        from ..data.catalogue import CatalogueManifest
        catalog = CatalogueManifest()
        cards_dir = Path(source_cards_dir) if source_cards_dir else Path(__file__).parents[2] / 'configs/source_cards'
        for path in sorted(cards_dir.glob('*.json')):
            catalog.import_card(SourceCard.model_validate_json(path.read_text()))
        discovery = DiscoveryService(state / 'discovery.sqlite', catalogue=catalog,
            metadata=[CatalogueMeta(source_id=s, title=catalog.card(s).topic, concepts=[catalog.card(s).topic])
                      for s in catalog.source_ids()], clock=clock)
    if research_runtime is not None:
        if Path(research_runtime.state_dir).resolve() != state:
            raise ValueError('runtime and MCP must use the same configured state')
        if discovery is not research_runtime.discovery:
            raise ValueError('runtime and MCP must share the same discovery service')
    if role in ('strategy', 'research', 'integrity'):
        for tool in [Tool('catalogue.search','Search approved source concepts and availability',CatalogueQuery,
                           lambda a: discovery.search_catalogue(ctx,a.query,limit=a.limit)),
                     Tool('catalogue.describe','Read documented source eligibility',CatalogueRead,
                           lambda a: discovery.describe_catalogue(ctx,a.source_id))]:
            server.tools[tool.name] = tool
    if role == 'strategy':
        for tool in [Tool('sources.discover','Bounded host-registered connector discovery; cannot grant access',DiscoverQuery,
                           lambda a: research_runtime.discover(ctx,a.provider_id,a.query,limit=a.limit) if research_runtime
                           else discovery.discover(ctx,namespace,a.provider_id,a.query,limit=a.limit),False),
                     Tool('sources.sample','Read a bounded sample of a previously discovered eligible source',SourceSample,
                           lambda a: discovery.sample(ctx,namespace,a.query_fingerprint,a.provider_ref,max_rows=a.max_rows),False),
                     Tool('sources.propose','Queue source or connector review; never self-approve',SourceProposal,
                           lambda a: discovery.propose_source(ctx,namespace,a.query_fingerprint,a.provider_ref,a.rationale),False)]:
            server.tools[tool.name] = tool
        if research_runtime is not None:
            server.tools['scheduler.status'] = Tool('scheduler.status','Read this pod research schedule',Empty,
                lambda a: research_runtime.scheduler.status(namespace))
            server.tools['research.cycle_status'] = Tool('research.cycle_status','Read own bounded research outcomes and budgets',Empty,
                lambda a: research_runtime.status(namespace))
    return server

# Replaces the earlier launcher while preserving build_server's legacy pod API.
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir',type=Path,default=Path('.pod_state'))
    parser.add_argument('--role',choices=['strategy','research','allocation','risk','execution','reporting'],default='strategy')
    parser.add_argument('--pod',choices=[p.value for p in Charter]+['fundamentals','events','valuation','dynamics'])
    parser.add_argument('--online',action='store_true')
    parser.add_argument('--papers-db',type=Path)
    args=parser.parse_args(argv)
    if args.role=='strategy' and not args.pod:parser.error('--pod is required for the strategy role')
    server=build_server(args.state_dir,args.pod,role=args.role,online=args.online,papers_db=args.papers_db)
    try:server.serve(sys.stdin,sys.stdout)
    finally:server.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
