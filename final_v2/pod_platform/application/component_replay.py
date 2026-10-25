"""Offline replay of captured proposals/critics through today's research policy.

This is a host-side diagnostic, never new LLM research or a historical backtest.
Original evidence, hypotheses and critic text are retained without rewriting them.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from ..agents.provider import DeterministicMockProvider
from ..papers import PaperService, PaperStore
from ..research.memory import PodMemory
from ..research.progression import ProgressionPolicy
from .common import atomic_write_json, digest, file_sha256, now_utc, read_json
from .research_loop import AutonomousResearchLoop, RoleProviders


def replay_components(*, from_research, state_dir, manifest_path, papers_db):
    source, state = Path(from_research).resolve(), Path(state_dir).resolve()
    if state == source or source in state.parents or state in source.parents:
        raise ValueError("captured research and replay state must be separate")
    if (
        not (source / "runtime.sqlite").is_file()
        or not Path(papers_db).is_file()
    ):
        raise ValueError("existing captured runtime and paper store required")
    with sqlite3.connect(
        f"file:{source / 'runtime.sqlite'}?mode=ro", uri=True
    ) as conn:
        rows = conn.execute(
            "SELECT key,namespace,kind,payload,hash FROM runtime_events ORDER BY seq"
        ).fetchall()
    events = []
    for key, ns, kind, body, sha in rows:
        value = json.loads(body)
        if digest([key, ns, kind, value]) != sha:
            raise ValueError("captured runtime event hash mismatch")
        events.append((ns, kind, value))
    reviews = [(ns, value) for ns, kind, value in events if kind == "review"]
    records = []
    for ns, kind, critic in events:
        if kind != "critique":
            continue
        entry = PodMemory(source / "memory", ns).get(critic["draft_entry_id"])
        if entry is None or entry.kind != "draft_proposal":
            raise ValueError("captured critic has no verified draft")
        records.append((ns, entry, critic))
    if not records:
        raise ValueError("no captured proposal/critic pairs")
    state.mkdir(parents=True, exist_ok=True)
    binding = {
        "mode": "captured_output_replay",
        "from_research": str(source),
        "source_runtime_sha256": file_sha256(source / "runtime.sqlite"),
        "manifest_sha256": file_sha256(manifest_path),
        "papers_sha256": file_sha256(papers_db),
        "record_digests": [
            digest([ns, entry.content, critic]) for ns, entry, critic in records
        ],
    }
    binding_path = state / "replay_binding.json"
    if binding_path.exists():
        if read_json(binding_path) != binding:
            raise ValueError("replay inputs changed; use a new state directory")
    else:
        atomic_write_json(binding_path, binding)
    clock_path = state / "clock.json"
    if not clock_path.exists():
        atomic_write_json(clock_path, {"started_at": now_utc().isoformat()})
    at = dt.datetime.fromisoformat(read_json(clock_path)["started_at"])
    lookup = {
        digest(entry.content["proposal"]): critic["value"]
        for _, entry, critic in records
    }

    def replay_critic(request):
        return json.dumps(lookup[digest(json.loads(request.user)["proposal"])])

    def no_inference(_):
        raise RuntimeError("offline replay may not call lead/extractor models")

    challenger = DeterministicMockProvider(
        replay_critic, "captured-critic-replay", clock=lambda: at
    )
    unused = DeterministicMockProvider(
        no_inference, "disabled-offline-replay", clock=lambda: at
    )
    # Adapter provenance checks may append paper-spec artifacts. Keep them in a
    # private SQLite backup so a replay never modifies the captured source cache.
    replay_papers = state / "papers.sqlite"
    if not replay_papers.exists():
        with sqlite3.connect(
            f"file:{Path(papers_db).resolve()}?mode=ro", uri=True
        ) as src:
            with sqlite3.connect(replay_papers) as dest:
                src.backup(dest)
    loop = AutonomousResearchLoop(
        state / "research",
        providers=RoleProviders(unused, challenger, unused),
        manifest_path=manifest_path,
        papers=PaperService(PaperStore(replay_papers), clock=lambda: at),
        clock=lambda: at,
        run_id="captured_component_replay_v1",
        progression_policy=ProgressionPolicy(probe_only=True),
    )
    for ns, review in reviews:
        loop.review_method(ns, review)
    inputs = []
    for ns, entry, critic in records:
        key = "captured:" + entry.entry_id
        imported = loop.memory(ns).append(
            "draft_proposal", entry.content, idempotency_key=key
        )
        job = loop.queue.submit(
            ns,
            "propose",
            {"captured_entry_id": entry.entry_id},
            idempotency_key=key,
        )
        if job.state == "queued":
            claim = loop.queue.claim(
                "captured_replay", namespace=ns, kinds=["propose"]
            )
            if claim is None or claim.job_id != job.job_id:
                raise ValueError("unexpected replay queue contents")
            loop.queue.complete(
                claim.job_id,
                claim.lease_token,
                {"memory_entry_id": imported.entry_id},
            )
        inputs.append(
            {
                "namespace": ns,
                "original_entry_id": entry.entry_id,
                "original_critic": critic,
                "original_proposal": entry.content["proposal"],
                "job_id": job.job_id,
            }
        )
    # No scheduler tick: only these already captured proposals may be routed.
    loop._route_proposals()
    results = []
    for value in inputs:
        route = loop._get("route:" + value["job_id"])
        results.append({**value, "route": route})
    if (
        loop._events("adaptation")
        or loop._events("outcome")
        or loop._events("publication")
    ):
        raise ValueError(
            "component replay unexpectedly attempted integrated financial activity"
        )
    summary = {
        **binding,
        "started_at": at.isoformat(),
        "new_model_api_calls": 0,
        "admissions": 0,
        "protected_check_accesses": 0,
        "orders": 0,
        "limitations": [
            "Current-policy replay of saved contemporary research; not a point-in-time 2025 backtest.",
            "Canonical component diagnostics leave unmatched original paper/model mechanisms UNTESTED.",
            "DESIGN IC differences are descriptive comparisons, not incremental alpha or profitability proof.",
        ],
        "comparisons": results,
        "progression": loop.progression.snapshot(),
    }
    atomic_write_json(state / "summary.json", summary)
    return summary
