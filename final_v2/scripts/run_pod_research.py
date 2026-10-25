"""Pod research queue CLI: submit / list / show / run / memory.

Safe by default. ``run`` without flags is a DRY RUN: it validates queued jobs
and prints what would be sent, claims nothing and calls no model.

  --demo    use the deterministic offline mock (not a language model)
  --online  call the explicitly configured model (billable / local CLI):
            --provider openai   --base-url URL --model NAME --api-key-env VAR
            --provider claude-code [--model NAME]

State lives under --state-dir: queue.sqlite, cache.sqlite, budget.sqlite,
memory/<namespace>.sqlite. Drafts are never evaluated results and nothing here
admits a model or deploys a portfolio. ``--executors-module`` imports your own
module exposing ``register_executors(registry)``; ``--dataset-manifest PATH``
additionally registers the built-in executors
(``pod_platform.research.builtins.register_executors``, imported lazily).
LLM output never selects code, only an id.

Budgets: ``--online`` charges a durable, concurrency-safe account in
budget.sqlite named by --budget-id/--budget-cycle (stable defaults), so
restarting or running several workers cannot evade --max-calls/--max-tokens.
Re-running with lower limits tightens the cap; higher limits are refused unless
you pass --raise-budget-limits (audited) or choose a new --budget-cycle.
``--demo`` uses a separate in-memory budget that is NOT durable.

Examples:
  python final_v2/scripts/run_pod_research.py submit --state-dir /tmp/pods --pod valuation \\
      --question "Does X predict Y?" --paper-cards papers.json --executor-id value_spread_ic
  python final_v2/scripts/run_pod_research.py run --state-dir /tmp/pods            # dry run
  python final_v2/scripts/run_pod_research.py run --state-dir /tmp/pods --demo
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pod_platform.agents.budgets import (  # noqa: E402
    DEFAULT_BUDGET_ID,
    DEFAULT_CYCLE,
    BudgetIntegrityError,
    PersistentCallBudget,
)
from pod_platform.agents.durable_cache import DurableExtractionCache  # noqa: E402
from pod_platform.agents.provider import (  # noqa: E402
    CallBudget,
    ClaudeCodeProvider,
    GenerationParams,
    LLMRequest,
    OpenAICompatibleProvider,
    ResearchMode,
    utc_now,
)
from pod_platform.agents.registry import ModelCheckpoint, ModelRegistry, TemporalPolicy  # noqa: E402
from pod_platform.pods.charters import POD_SPECS, pod_spec  # noqa: E402
from pod_platform.pods.demo import demo_provider  # noqa: E402
from pod_platform.pods.runner import ExecutorRegistry, PodResearchRunner, PodWorker  # noqa: E402
from pod_platform.pods.schema import normalise_evidence  # noqa: E402
from pod_platform.research.jobs import QueueError, ResearchQueue  # noqa: E402
from pod_platform.research.memory import PodMemory  # noqa: E402

CARD_FIELDS = (("paper_cards", "paper_card"), ("dataset_cards", "dataset_card"), ("source_cards", "source_card"))


def _load_cards(path: Optional[str]) -> list[dict[str, Any]]:
    if not path:
        return []
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict):
        doc = doc.get("cards", doc.get("items"))
    if not isinstance(doc, list) or not all(isinstance(c, dict) for c in doc):
        raise SystemExit(f"{path}: expected a JSON list of card objects")
    return doc


def _paths(state_dir: str) -> tuple[Path, Path, Path]:
    root = Path(state_dir)
    return root / "queue.sqlite", root / "cache.sqlite", root / "memory"


def _parse_asof(text: Optional[str]) -> dt.datetime:
    if not text:
        return utc_now()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SystemExit("--research-asof must include a timezone offset")
    return parsed.astimezone(dt.timezone.utc)


def _registry(path: Optional[str]) -> Optional[ModelRegistry]:
    if not path:
        return None
    registry = ModelRegistry()
    for doc in json.loads(Path(path).read_text(encoding="utf-8")):
        registry.register(ModelCheckpoint.model_validate(doc))
    return registry


def cmd_submit(args: argparse.Namespace) -> int:
    pod = pod_spec(args.pod)
    queue_path, _cache, mem_root = _paths(args.state_dir)
    asof = _parse_asof(args.research_asof)
    mode = ResearchMode(args.mode)
    if mode == ResearchMode.HISTORICAL and not args.checkpoint_id:
        raise SystemExit("historical research requires --checkpoint-id of a registered checkpoint")
    payload: dict[str, Any] = {
        "question": args.question,
        "research_asof": asof.isoformat(),
        "mode": mode.value,
        "policy": args.policy,
        "checkpoint_id": args.checkpoint_id,
        "executors": sorted(set(args.executor_id)),
        "paper_cards": _load_cards(args.paper_cards),
        "dataset_cards": _load_cards(args.dataset_cards),
        "source_cards": _load_cards(args.source_cards),
        "prior_work": PodMemory(mem_root, pod.namespace).prior_work_summaries(),
    }
    total = 0
    for field_name, kind in CARD_FIELDS:  # validate eagerly so bad cards never enter the queue
        total += len(normalise_evidence(payload[field_name], kind))  # type: ignore[arg-type]
    if total == 0:
        raise ValueError("at least one evidence card (--paper-cards/--dataset-cards/--source-cards) is required")
    queue = ResearchQueue(queue_path)
    job = queue.submit(
        pod.namespace, "propose", payload, idempotency_key=args.idempotency_key, max_attempts=args.max_attempts
    )
    print(json.dumps({"job_id": job.job_id, "state": job.state, "namespace": job.namespace}))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    queue = ResearchQueue(_paths(args.state_dir)[0])
    ns = pod_spec(args.pod).namespace if args.pod else None
    for job in queue.list(namespace=ns, state=args.state, limit=args.limit):
        print(f"{job.job_id}\t{job.namespace}\t{job.kind}\t{job.state}\tattempts={job.attempts}/{job.max_attempts}")
    print(json.dumps(queue.counts(ns)))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    queue = ResearchQueue(_paths(args.state_dir)[0])
    job = queue.get(args.job_id)
    if job is None:
        print(f"unknown job {args.job_id}", file=sys.stderr)
        return 1
    payload = dict(job.payload)
    for field_name, _kind in CARD_FIELDS:
        payload[field_name] = f"<{len(payload.get(field_name) or [])} cards>"
    print(
        json.dumps(
            {
                "job_id": job.job_id,
                "namespace": job.namespace,
                "kind": job.kind,
                "state": job.state,
                "attempts": job.attempts,
                "error": job.error,
                "payload": payload,
                "result": queue.result(job.job_id),
                "events": queue.events(job.job_id),
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    pod = pod_spec(args.pod)
    memory = PodMemory(_paths(args.state_dir)[2], pod.namespace)
    for entry in memory.list(args.kind, limit=args.limit):
        print(f"{entry.entry_id}\t{entry.kind}\tevaluated={entry.evaluated}\t{entry.content.get('title') or entry.content.get('error', '')}")
    return 0


def _build_budgets(args: argparse.Namespace) -> "tuple[dict[str, CallBudget], dict[str, Any]]":
    """namespace -> budget. Online runs are durable; demo is an explicit in-memory budget."""
    if args.demo:
        shared = CallBudget(args.max_calls, args.max_tokens)
        return {spec.namespace: shared for spec in POD_SPECS.values()}, {"mode": "demo", "persistent": False}
    path = Path(args.state_dir) / "budget.sqlite"
    budget_id = args.budget_id or DEFAULT_BUDGET_ID

    def account(pod: str) -> PersistentCallBudget:
        return PersistentCallBudget(
            path,
            budget_id=budget_id,
            max_calls=args.max_calls,
            max_tokens=args.max_tokens,
            pod=pod,
            cycle=args.budget_cycle,
            raise_limits=args.raise_budget_limits,
        )

    if args.budget_scope == "per-pod":
        budgets = {spec.namespace: account(spec.pod_id) for spec in POD_SPECS.values()}
    else:
        shared_budget = account("*")
        budgets = {spec.namespace: shared_budget for spec in POD_SPECS.values()}
    return budgets, {"mode": "online", "persistent": True, "budget_file": str(path), "scope": args.budget_scope}


def _register_builtin_executors(registry: ExecutorRegistry, manifest_path: str, state_dir: str) -> None:
    """Lazy: the built-in executors (and their optional pandas/pyarrow needs) are only imported on request."""
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise ValueError(f"--dataset-manifest not found: {manifest}")
    try:
        module = importlib.import_module("pod_platform.research.builtins")
        register = module.register_executors
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"built-in executors are unavailable: {type(exc).__name__}: {exc}") from None
    work_dir = Path(state_dir) / "builtin_executors"
    work_dir.mkdir(parents=True, exist_ok=True)
    register(registry, manifest_path=manifest, state_dir=work_dir)


def _build_provider(args: argparse.Namespace):
    if args.online:
        if args.provider == "openai":
            if not args.base_url or not args.model:
                raise SystemExit("--provider openai requires --base-url and --model")
            return OpenAICompatibleProvider(args.base_url, args.model, args.api_key_env)
        if args.provider == "claude-code":
            return ClaudeCodeProvider(args.model, timeout_s=args.timeout)
        raise SystemExit("--online requires --provider openai|claude-code")
    return demo_provider()


def cmd_run(args: argparse.Namespace) -> int:
    queue_path, cache_path, mem_root = _paths(args.state_dir)
    queue = ResearchQueue(queue_path)
    if args.online and args.demo:
        raise SystemExit("choose one of --demo or --online")
    if not (args.online or args.demo):
        return _dry_run(args, queue, mem_root)
    provider = _build_provider(args)
    registry = _registry(args.registry_file)
    budgets, budget_info = _build_budgets(args)
    cache = DurableExtractionCache(cache_path)
    executors = ExecutorRegistry()
    if args.dataset_manifest:
        _register_builtin_executors(executors, args.dataset_manifest, args.state_dir)
    if args.executors_module:
        importlib.import_module(args.executors_module).register_executors(executors)
    runners = {
        spec.namespace: PodResearchRunner(
            spec, provider, cache, PodMemory(mem_root, spec.namespace), budget=budgets[spec.namespace], registry=registry
        )
        for spec in POD_SPECS.values()
    }
    if args.demo:
        print(
            "DEMO MODE: deterministic mock responder; no language model is called. "
            "The demo budget is in-memory and NOT durable.",
            file=sys.stderr,
        )
    else:
        sample = next(iter(budgets.values()))
        print(
            f"ONLINE budget account {getattr(sample, 'account_key', '?')} in {budget_info['budget_file']} "
            f"(scope={budget_info['scope']}); charges persist across restarts.",
            file=sys.stderr,
        )
    worker = PodWorker(queue, runners, executors, worker_id=args.worker_id, lease_seconds=args.lease_seconds)
    failed = 0
    for _ in range(args.max_jobs):
        outcome = worker.run_once()
        if outcome is None:
            break
        print(json.dumps({"job_id": outcome.job_id, "state": outcome.state, "error": outcome.error, "result": outcome.result}, default=str))
        failed += outcome.state in ("failed", "will_retry", "lease_lost")
    unique = {id(b): b for b in budgets.values()}
    snapshots = [b.snapshot() for b in unique.values()]
    shown = snapshots[0] if len(snapshots) == 1 else {"accounts": snapshots}
    print(
        json.dumps({"budget": {**budget_info, **shown}, "provider": provider.name, "model": provider.model}),
        file=sys.stderr,
    )
    return 1 if failed else 0


def _dry_run(args: argparse.Namespace, queue: ResearchQueue, mem_root: Path) -> int:
    print("DRY RUN: no job is claimed or changed and no model is called.")
    provider = demo_provider()
    for job in queue.list(state="queued", limit=args.max_jobs):
        if job.kind != "propose":
            print(f"{job.job_id}\t{job.kind}\t(experiment job: needs --demo/--online and registered executors)")
            continue
        spec = pod_spec(job.namespace)
        runner = PodResearchRunner(
            spec, provider, DurableExtractionCache(_paths(args.state_dir)[1]), PodMemory(mem_root, spec.namespace)
        )
        payload = job.payload
        try:
            evidence = [i for f, k in CARD_FIELDS for i in normalise_evidence(payload.get(f) or [], k)]  # type: ignore[arg-type]
            model_input, items, _allowed = runner.prepare(
                payload["question"],
                evidence,
                research_asof=dt.datetime.fromisoformat(payload["research_asof"]),
                mode=ResearchMode(payload["mode"]),
                executors=payload["executors"],
                prior_work=payload.get("prior_work"),
            )
            request = LLMRequest(
                runner.extractor.system_prompt,
                json.dumps(model_input, sort_keys=True, separators=(",", ":")),
                GenerationParams(),
            )
            flagged = sum(1 for i in items if i.injection_flags)
            print(
                f"{job.job_id}\t{spec.pod_id}\tmode={payload['mode']}\tevidence={len(items)}\t"
                f"injection_flagged={flagged}\tprompt_hash={request.prompt_hash[:12]}\tinput_hash={request.input_hash[:12]}"
            )
        except Exception as exc:
            print(f"{job.job_id}\tWOULD FAIL: {type(exc).__name__}: {exc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pods = sorted(s.pod_id for s in POD_SPECS.values())

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--state-dir", required=True, help="directory for queue, cache and memory")

    p = sub.add_parser("submit", help="enqueue a proposal job")
    common(p)
    p.add_argument("--pod", required=True, choices=pods)
    p.add_argument("--question", required=True)
    p.add_argument("--paper-cards", help="JSON list of paper evidence cards")
    p.add_argument("--dataset-cards")
    p.add_argument("--source-cards")
    p.add_argument("--mode", choices=[m.value for m in ResearchMode], default="contemporary")
    p.add_argument("--research-asof", help="ISO timestamp with offset (default: now)")
    p.add_argument("--checkpoint-id", help="registered checkpoint (required for historical)")
    p.add_argument("--policy", choices=[t.value for t in TemporalPolicy], default=TemporalPolicy.RELEASE_GATED.value)
    p.add_argument("--executor-id", action="append", required=True, help="allowed registered executor id (repeatable)")
    p.add_argument("--idempotency-key")
    p.add_argument("--max-attempts", type=int, default=2)
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("list", help="list jobs")
    common(p)
    p.add_argument("--pod", choices=pods)
    p.add_argument("--state", choices=["queued", "running", "succeeded", "failed"])
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="show one job with result and events")
    common(p)
    p.add_argument("job_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("memory", help="list a pod's private memory")
    common(p)
    p.add_argument("--pod", required=True, choices=pods)
    p.add_argument("--kind")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("run", help="process queued jobs (dry run unless --demo/--online)")
    common(p)
    p.add_argument("--demo", action="store_true", help="deterministic offline mock; not a model")
    p.add_argument("--online", action="store_true", help="call the configured model")
    p.add_argument("--provider", choices=["openai", "claude-code"])
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--api-key-env", help="NAME of the environment variable holding the API key")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--registry-file", help="JSON list of ModelCheckpoint objects (for historical mode)")
    p.add_argument("--executors-module", help="module exposing register_executors(registry)")
    p.add_argument(
        "--dataset-manifest",
        help="dataset manifest; registers the built-in executors (lazy import). Not needed for proposals/dry runs",
    )
    p.add_argument("--max-jobs", type=int, default=5)
    p.add_argument("--max-calls", type=int, default=10, help="LLM call cap of the budget account")
    p.add_argument("--max-tokens", type=int, default=100000, help="LLM token cap of the budget account")
    p.add_argument(
        "--budget-id",
        default=None,
        help=f"named durable budget account for --online (default: stable '{DEFAULT_BUDGET_ID}', not random)",
    )
    p.add_argument("--budget-cycle", default=DEFAULT_CYCLE, help="research cycle the account belongs to")
    p.add_argument(
        "--budget-scope",
        choices=["shared", "per-pod"],
        default="shared",
        help="one account for all pods (default) or one account per pod",
    )
    p.add_argument(
        "--raise-budget-limits",
        action="store_true",
        help="allow limits above the account's recorded cap (audited); charges are never reset",
    )
    p.add_argument("--worker-id", default="cli-worker")
    p.add_argument("--lease-seconds", type=float, default=600.0)
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (QueueError, ValueError, KeyError, BudgetIntegrityError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
