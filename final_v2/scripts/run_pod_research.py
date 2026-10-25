"""Pod research queue utility: submit / list / show / run / memory.

Settings are read from the matching section of final_v2/config.json.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

from pod_platform.agents.budgets import (  # noqa: E402
    DEFAULT_BUDGET_ID,
    BudgetIntegrityError,
    PersistentCallBudget,
)
from pod_platform.agents.durable_cache import (
    DurableExtractionCache,  # noqa: E402
)
from pod_platform.agents.provider import (  # noqa: E402
    CallBudget,
    ClaudeCodeProvider,
    GenerationParams,
    LLMRequest,
    OpenAICompatibleProvider,
    ResearchMode,
    utc_now,
)
from pod_platform.agents.registry import (  # noqa: E402
    ModelCheckpoint,
    ModelRegistry,
)
from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.pods.charters import POD_SPECS, pod_spec
from pod_platform.pods.demo import demo_provider
from pod_platform.pods.runner import (  # noqa: E402
    ExecutorRegistry,
    PodResearchRunner,
    PodWorker,
)
from pod_platform.pods.schema import normalise_evidence
from pod_platform.research.jobs import QueueError, ResearchQueue
from pod_platform.research.memory import PodMemory

ROOT = Path(__file__).resolve().parent.parent


CARD_FIELDS = (
    ("paper_cards", "paper_card"),
    ("dataset_cards", "dataset_card"),
    ("source_cards", "source_card"),
)


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


def cmd_submit(args: SimpleNamespace) -> int:
    pod = pod_spec(args.pod)
    queue_path, _cache, mem_root = _paths(args.state_dir)
    asof = _parse_asof(args.research_asof)
    mode = ResearchMode(args.mode)
    if mode == ResearchMode.HISTORICAL and not args.checkpoint_id:
        raise SystemExit(
            "historical research requires --checkpoint-id of a registered checkpoint"
        )
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
    for (
        field_name,
        kind,
    ) in CARD_FIELDS:  # validate eagerly so bad cards never enter the queue
        total += len(normalise_evidence(payload[field_name], kind))  # type: ignore[arg-type]
    if total == 0:
        raise ValueError(
            "at least one evidence card (--paper-cards/--dataset-cards/--source-cards) is required"
        )
    queue = ResearchQueue(queue_path)
    job = queue.submit(
        pod.namespace,
        "propose",
        payload,
        idempotency_key=args.idempotency_key,
        max_attempts=args.max_attempts,
    )
    print(
        json.dumps(
            {
                "job_id": job.job_id,
                "state": job.state,
                "namespace": job.namespace,
            }
        )
    )
    return 0


def cmd_list(args: SimpleNamespace) -> int:
    queue = ResearchQueue(_paths(args.state_dir)[0])
    ns = pod_spec(args.pod).namespace if args.pod else None
    for job in queue.list(namespace=ns, state=args.state, limit=args.limit):
        print(
            f"{job.job_id}\t{job.namespace}\t{job.kind}\t{job.state}\tattempts={job.attempts}/{job.max_attempts}"
        )
    print(json.dumps(queue.counts(ns)))
    return 0


def cmd_show(args: SimpleNamespace) -> int:
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


def cmd_memory(args: SimpleNamespace) -> int:
    pod = pod_spec(args.pod)
    memory = PodMemory(_paths(args.state_dir)[2], pod.namespace)
    for entry in memory.list(args.kind, limit=args.limit):
        print(
            f"{entry.entry_id}\t{entry.kind}\tevaluated={entry.evaluated}\t{entry.content.get('title') or entry.content.get('error', '')}"
        )
    return 0


def _build_budgets(
    args: SimpleNamespace,
) -> "tuple[dict[str, CallBudget], dict[str, Any]]":
    """namespace -> budget. Online runs are durable; demo is an explicit in-memory budget."""
    if args.demo:
        shared = CallBudget(args.max_calls, args.max_tokens)
        return {spec.namespace: shared for spec in POD_SPECS.values()}, {
            "mode": "demo",
            "persistent": False,
        }
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
        budgets = {
            spec.namespace: account(spec.pod_id) for spec in POD_SPECS.values()
        }
    else:
        shared_budget = account("*")
        budgets = {spec.namespace: shared_budget for spec in POD_SPECS.values()}
    return budgets, {
        "mode": "online",
        "persistent": True,
        "budget_file": str(path),
        "scope": args.budget_scope,
    }


def _register_builtin_executors(
    registry: ExecutorRegistry, manifest_path: str, state_dir: str
) -> None:
    """Lazy: the built-in executors (and their optional pandas/pyarrow needs) are only imported on request."""
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise ValueError(f"--dataset-manifest not found: {manifest}")
    try:
        module = importlib.import_module("pod_platform.research.builtins")
        register = module.register_executors
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"built-in executors are unavailable: {type(exc).__name__}: {exc}"
        ) from None
    work_dir = Path(state_dir) / "builtin_executors"
    work_dir.mkdir(parents=True, exist_ok=True)
    register(registry, manifest_path=manifest, state_dir=work_dir)


def _build_provider(args: SimpleNamespace):
    if args.online:
        if args.provider == "openai":
            if not args.base_url or not args.model:
                raise SystemExit(
                    "--provider openai requires --base-url and --model"
                )
            return OpenAICompatibleProvider(
                args.base_url, args.model, args.api_key_env
            )
        if args.provider == "claude-code":
            return ClaudeCodeProvider(args.model, timeout_s=args.timeout)
        raise SystemExit("--online requires --provider openai|claude-code")
    return demo_provider()


def cmd_run(args: SimpleNamespace) -> int:
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
        _register_builtin_executors(
            executors, args.dataset_manifest, args.state_dir
        )
    if args.executors_module:
        importlib.import_module(args.executors_module).register_executors(
            executors
        )
    runners = {
        spec.namespace: PodResearchRunner(
            spec,
            provider,
            cache,
            PodMemory(mem_root, spec.namespace),
            budget=budgets[spec.namespace],
            registry=registry,
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
    worker = PodWorker(
        queue,
        runners,
        executors,
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
    )
    failed = 0
    for _ in range(args.max_jobs):
        outcome = worker.run_once()
        if outcome is None:
            break
        print(
            json.dumps(
                {
                    "job_id": outcome.job_id,
                    "state": outcome.state,
                    "error": outcome.error,
                    "result": outcome.result,
                },
                default=str,
            )
        )
        failed += outcome.state in ("failed", "will_retry", "lease_lost")
    unique = {id(b): b for b in budgets.values()}
    snapshots = [b.snapshot() for b in unique.values()]
    shown = snapshots[0] if len(snapshots) == 1 else {"accounts": snapshots}
    print(
        json.dumps(
            {
                "budget": {**budget_info, **shown},
                "provider": provider.name,
                "model": provider.model,
            }
        ),
        file=sys.stderr,
    )
    return 1 if failed else 0


def _dry_run(
    args: SimpleNamespace, queue: ResearchQueue, mem_root: Path
) -> int:
    print("DRY RUN: no job is claimed or changed and no model is called.")
    provider = demo_provider()
    for job in queue.list(state="queued", limit=args.max_jobs):
        if job.kind != "propose":
            print(
                f"{job.job_id}\t{job.kind}\t(experiment job: needs --demo/--online and registered executors)"
            )
            continue
        spec = pod_spec(job.namespace)
        runner = PodResearchRunner(
            spec,
            provider,
            DurableExtractionCache(_paths(args.state_dir)[1]),
            PodMemory(mem_root, spec.namespace),
        )
        payload = job.payload
        try:
            evidence = [
                i
                for f, k in CARD_FIELDS
                for i in normalise_evidence(payload.get(f) or [], k)
            ]  # type: ignore[arg-type]
            model_input, items, _allowed = runner.prepare(
                payload["question"],
                evidence,
                research_asof=dt.datetime.fromisoformat(
                    payload["research_asof"]
                ),
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


SCRIPT_SETTINGS = load_config("_utility_schemas")["run_pod_research"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = load_script_settings("run_pod_research", SCRIPT_SETTINGS, argv)
    args.func = {
        "submit": cmd_submit,
        "list": cmd_list,
        "show": cmd_show,
        "memory": cmd_memory,
        "run": cmd_run,
    }[args.command]
    try:
        return args.func(args)
    except (QueueError, ValueError, KeyError, BudgetIntegrityError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
