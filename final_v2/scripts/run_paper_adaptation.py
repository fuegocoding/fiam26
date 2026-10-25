"""Reviewed paper adaptation: adapt (register) / submit (queue worker job) / execute (design-only, in process) / lineage.

``adapt`` reads a human-written AdaptationReview JSON, checks the cached paper (full text, grounded claims, version
date) and the operator dataset manifest descriptor, and registers HypothesisSpec + ExperimentSpec BEFORE anything runs.
It never fetches network data, calls a model or opens the dataset file. ``execute`` runs only the registered builtin
executor on the design partition; admission is never produced.

Examples:
  run_paper_adaptation.py adapt --state-dir /tmp/pods --manifest manifest.json --review review.json [--submit]
  run_paper_adaptation.py execute --state-dir /tmp/pods --manifest manifest.json --adaptation-id ad_...
  run_paper_adaptation.py lineage --state-dir /tmp/pods --manifest manifest.json --adaptation-id ad_...
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pod_platform.papers import PapersError, PaperService  # noqa: E402
from pod_platform.papers.adaptations import AdaptationRejected, PaperAdapter  # noqa: E402
from pod_platform.pods.runner import ExecutorRegistry  # noqa: E402
from pod_platform.research.builtins.executors import register_executors  # noqa: E402
from pod_platform.research.jobs import QueueError, ResearchQueue  # noqa: E402
from pod_platform.research.memory import PodMemory  # noqa: E402
from pod_platform.research.registration import ResearchLedger  # noqa: E402
from pod_platform.research.registry import Registry  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("adapt", "submit", "execute", "lineage"))
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--manifest", help="operator dataset manifest (default $POD_DATASET_MANIFEST)")
    parser.add_argument("--review", help="AdaptationReview JSON file (adapt)")
    parser.add_argument("--adaptation-id")
    parser.add_argument("--submit", action="store_true", help="after adapt, queue the worker 'experiment' job")
    parser.add_argument("--papers-db", help="paper cache (default <state-dir>/shared/papers.sqlite)")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, clock: Optional[Callable[[], datetime]] = None, out=None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    clock = clock or (lambda: datetime.now(timezone.utc))
    state = Path(args.state_dir)
    registry = papers = None
    try:
        state.mkdir(parents=True, exist_ok=True)
        executors = ExecutorRegistry()
        ctx = register_executors(executors, manifest_path=args.manifest, state_dir=state / "builtin", clock=clock)
        papers = PaperService(args.papers_db or state / "shared" / "papers.sqlite", clock=clock, online=False)
        registry = Registry(state / "registry.sqlite")

        def adapter(namespace: str) -> PaperAdapter:
            return PaperAdapter(state / "adaptations.sqlite", papers=papers, registry=registry, ledger=ResearchLedger(registry),
                                memory=PodMemory(state / "memory", namespace), context=ctx, clock=clock)

        queue = ResearchQueue(state / "queue.sqlite", clock=clock)
        if args.command == "adapt":
            review = json.loads(Path(args.review).read_text(encoding="utf-8"))
            ad = adapter(review.get("namespace", ""))
            result = ad.adapt(review)
            if args.submit:
                result["job_id"] = ad.submit_job(result["adaptation_id"], queue).job_id
        else:
            ns = _namespace_of(state, args.adaptation_id)
            ad = adapter(ns)
            if args.command == "submit":
                result = {"job_id": ad.submit_job(args.adaptation_id, queue).job_id}
            elif args.command == "execute":
                result = ad.execute_design(args.adaptation_id, executors)
            else:
                result = {"lineage": ad.lineage(args.adaptation_id)}
        out.write(json.dumps({"ok": True, "result": result}, indent=2, default=str) + "\n")
        return 0
    except (AdaptationRejected, PapersError, QueueError, ValueError, KeyError, OSError, sqlite3.Error) as exc:
        out.write(json.dumps({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)[:500]}}) + "\n")
        return 2
    finally:
        if registry is not None:
            registry.close()
        if papers is not None:
            papers.store.close()


def _namespace_of(state: Path, adaptation_id: Optional[str]) -> str:
    import sqlite3
    with sqlite3.connect(str(state / "adaptations.sqlite")) as conn:
        row = conn.execute("SELECT namespace FROM adaptations WHERE adaptation_id=?", (adaptation_id or "",)).fetchone()
    if row is None:
        raise KeyError(f"unknown adaptation {adaptation_id!r}")
    return row[0]


if __name__ == "__main__":
    sys.exit(main())
