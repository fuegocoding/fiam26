"""Recurring research allocator: configure / add-trigger / tick / status (explicit ticks only; no daemon, no model calls).

Settings are read from the matching section of final_v2/config.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.pods.charters import pod_spec
from pod_platform.research.jobs import QueueError, ResearchQueue
from pod_platform.research.scheduler import (  # noqa: E402
    PodPolicy,
    ResearchScheduler,
    SchedulerError,
)

ROOT = Path(__file__).resolve().parent.parent


SCRIPT_SETTINGS = load_config("_utility_schemas")["research_scheduler"]


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    clock: Optional[Callable[[], datetime]] = None,
    out=None,
) -> int:
    out = out or sys.stdout
    args = load_script_settings("research_scheduler", SCRIPT_SETTINGS, argv)
    clock = clock or (lambda: datetime.now(timezone.utc))
    state = Path(args.state_dir)
    try:
        state.mkdir(parents=True, exist_ok=True)
        sched = ResearchScheduler(
            state / "scheduler.sqlite",
            queue=ResearchQueue(state / "queue.sqlite", clock=clock),
            clock=clock,
            memory_root=str(state / "memory"),
        )
        ns = (
            pod_spec(args.pod).namespace if getattr(args, "pod", None) else None
        )
        if args.command == "configure":
            kwargs = {
                k: json.loads(v)
                for k, v in (
                    ("weekly_quota", args.weekly_quota),
                    ("monthly_quota", args.monthly_quota),
                )
                if v
            }
            pol = PodPolicy(
                ns,
                tuple(args.executor_id),
                cohort_maturity_days=args.cohort_maturity_days,
                max_failures_per_line=args.max_failures_per_line,
                **kwargs,
            )
            result = sched.configure_pod(pol).__dict__
        elif args.command == "add-trigger":
            payload = (
                json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
                if args.payload_file
                else {}
            )
            result = sched.add_trigger(
                ns,
                args.kind,
                args.subject,
                args.question,
                json.loads(args.scores),
                payload=payload,
                line_id=args.line_id,
                depends_on=args.depends_on,
                cohort_month=args.cohort_month,
                redundancy_key=args.redundancy_key,
            )
        elif args.command == "pause":
            sched.pause(ns, args.reason, args.line_id)
            result = {"paused": args.line_id or "*"}
        elif args.command == "status":
            result = sched.status(ns)
        else:
            result = sched.tick()
        out.write(
            json.dumps({"ok": True, "result": result}, indent=2, default=str)
            + "\n"
        )
        return 0
    except (SchedulerError, QueueError, ValueError, KeyError, OSError) as exc:
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


if __name__ == "__main__":
    sys.exit(main())
