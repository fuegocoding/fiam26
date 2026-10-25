"""Reviewed paper adaptation: adapt (register) / submit (queue worker job) / execute (design-only, in process) / lineage.

Settings are read from the matching section of final_v2/config.json.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.papers import PapersError, PaperService
from pod_platform.papers.adaptations import (  # noqa: E402
    AdaptationRejected,
    PaperAdapter,
)
from pod_platform.pods.runner import ExecutorRegistry
from pod_platform.research.builtins.executors import (
    register_executors,  # noqa: E402
)
from pod_platform.research.jobs import QueueError, ResearchQueue
from pod_platform.research.memory import PodMemory
from pod_platform.research.registration import ResearchLedger
from pod_platform.research.registry import Registry

ROOT = Path(__file__).resolve().parent.parent


SCRIPT_SETTINGS = load_config("_utility_schemas")["run_paper_adaptation"]


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    clock: Optional[Callable[[], datetime]] = None,
    out=None,
) -> int:
    out = out or sys.stdout
    args = load_script_settings("run_paper_adaptation", SCRIPT_SETTINGS, argv)
    clock = clock or (lambda: datetime.now(timezone.utc))
    state = Path(args.state_dir)
    registry = papers = None
    try:
        state.mkdir(parents=True, exist_ok=True)
        executors = ExecutorRegistry()
        ctx = register_executors(
            executors,
            manifest_path=args.manifest,
            state_dir=state / "builtin",
            clock=clock,
        )
        papers = PaperService(
            args.papers_db or state / "shared" / "papers.sqlite",
            clock=clock,
            online=False,
        )
        registry = Registry(state / "registry.sqlite")

        def adapter(namespace: str) -> PaperAdapter:
            return PaperAdapter(
                state / "adaptations.sqlite",
                papers=papers,
                registry=registry,
                ledger=ResearchLedger(registry),
                memory=PodMemory(state / "memory", namespace),
                context=ctx,
                clock=clock,
            )

        queue = ResearchQueue(state / "queue.sqlite", clock=clock)
        if args.command == "adapt":
            review = json.loads(Path(args.review).read_text(encoding="utf-8"))
            ad = adapter(review.get("namespace", ""))
            result = ad.adapt(review)
            if args.submit:
                result["job_id"] = ad.submit_job(
                    result["adaptation_id"], queue
                ).job_id
        else:
            ns = _namespace_of(state, args.adaptation_id)
            ad = adapter(ns)
            if args.command == "submit":
                result = {
                    "job_id": ad.submit_job(args.adaptation_id, queue).job_id
                }
            elif args.command == "execute":
                result = ad.execute_design(args.adaptation_id, executors)
            else:
                result = {"lineage": ad.lineage(args.adaptation_id)}
        out.write(
            json.dumps({"ok": True, "result": result}, indent=2, default=str)
            + "\n"
        )
        return 0
    except (
        AdaptationRejected,
        PapersError,
        QueueError,
        ValueError,
        KeyError,
        OSError,
        sqlite3.Error,
    ) as exc:
        out.write(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc)[:500],
                    },
                }
            )
            + "\n"
        )
        return 2
    finally:
        if registry is not None:
            registry.close()
        if papers is not None:
            papers.store.close()


def _namespace_of(state: Path, adaptation_id: Optional[str]) -> str:
    with sqlite3.connect(str(state / "adaptations.sqlite")) as conn:
        row = conn.execute(
            "SELECT namespace FROM adaptations WHERE adaptation_id=?",
            (adaptation_id or "",),
        ).fetchone()
    if row is None:
        raise KeyError(f"unknown adaptation {adaptation_id!r}")
    return row[0]


if __name__ == "__main__":
    sys.exit(main())
