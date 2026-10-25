"""Recurring paper watches: add / list / tick (explicit ticks only; no daemon, no model calls).

Settings are read from the matching section of final_v2/config.json.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.papers import (  # noqa: E402
    PapersError,
    PaperService,
    UrllibTransport,
)
from pod_platform.papers.watches import PaperWatches
from pod_platform.research.jobs import QueueError, ResearchQueue

ROOT = Path(__file__).resolve().parent.parent


SCRIPT_SETTINGS = load_config("_utility_schemas")["research_watch"]


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    transport=None,
    clock: Optional[Callable[[], datetime]] = None,
    out=None,
) -> int:
    out = out or sys.stdout
    args = load_script_settings("research_watch", SCRIPT_SETTINGS, argv)
    clock = clock or (lambda: datetime.now(timezone.utc))
    state = Path(args.state_dir)
    online = getattr(args, "online", False)
    papers = None
    try:
        state.mkdir(parents=True, exist_ok=True)
        papers = (
            PaperService(
                getattr(args, "papers_db", None)
                or state / "shared" / "papers.sqlite",
                clock=clock,
                online=online,
                timeout=getattr(args, "timeout", 20.0),
                transport=(transport or UrllibTransport()) if online else None,
                openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
                openalex_mailto=os.environ.get("OPENALEX_MAILTO") or None,
            )
            if args.command == "tick"
            else None
        )
        watches = PaperWatches(
            state / "watches.sqlite",
            queue=ResearchQueue(state / "queue.sqlite", clock=clock),
            papers=papers,
            clock=clock,
        )
        if args.command == "add":
            result = watches.add(
                args.pod,
                args.watch_id,
                args.topic,
                cadence_hours=args.cadence_hours,
                executors=args.executor_id,
                question=args.question,
                search_limit=args.search_limit,
                papers_per_job=args.papers_per_job,
                max_jobs=args.max_jobs,
            )
        elif args.command == "list":
            result = watches.list(args.pod)
        else:
            result = watches.tick(args.pod, args.watch_id)
        code = (
            1
            if any(
                r.get("status") == "failed"
                for r in result
                if isinstance(r, dict)
            )
            else 0
        )
        out.write(
            json.dumps(
                {"ok": code == 0, "online": online, "result": result},
                indent=2,
                default=str,
            )
            + "\n"
        )
        return code
    except (PapersError, QueueError, ValueError, KeyError) as exc:
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
        if papers is not None:
            papers.store.close()


if __name__ == "__main__":
    sys.exit(main())
