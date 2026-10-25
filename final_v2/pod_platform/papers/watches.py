"""Persistent recurring paper watches (spec 10.7). Explicit ``tick`` only; no daemon.

A watch is (pod namespace, watch id, search topic, research question, cadence).
``tick`` searches through an injected ``PaperService`` (cache-only unless that
service is online) and queues bounded ``propose`` jobs whose ``paper_cards``
are the papers actually returned. A (pod, watch, paper, version) is queued at
most once, so re-ticking never spends model tokens twice. No model is called
and nothing is admitted here. Search failures are recorded, never reported ok.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any, Callable, Optional

from ..contracts.base import require_aware
from ..pods.charters import pod_spec
from ..pods.schema import normalise_evidence
from ..research.jobs import IdempotencyConflict, ResearchQueue, open_sqlite

_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
RETRY_AFTER_FAILURE_S = 3600
_SCHEMA = """
CREATE TABLE IF NOT EXISTS watches (
    namespace TEXT NOT NULL, watch_id TEXT NOT NULL, query TEXT NOT NULL, question TEXT NOT NULL,
    cadence_s INTEGER NOT NULL, config_json TEXT NOT NULL, created_at TEXT NOT NULL,
    next_due_at TEXT NOT NULL, last_success_at TEXT, PRIMARY KEY (namespace, watch_id));
CREATE TABLE IF NOT EXISTS seen (
    namespace TEXT NOT NULL, watch_id TEXT NOT NULL, paper_id TEXT NOT NULL, version_key TEXT NOT NULL,
    job_id TEXT, outcome TEXT NOT NULL, at TEXT NOT NULL,
    PRIMARY KEY (namespace, watch_id, paper_id, version_key));
CREATE TABLE IF NOT EXISTS runs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT NOT NULL, watch_id TEXT NOT NULL,
    at TEXT NOT NULL, status TEXT NOT NULL, detail_json TEXT NOT NULL);
"""


def _iso(value: dt.datetime) -> str:
    return (
        require_aware(value, "timestamp")
        .astimezone(dt.timezone.utc)
        .isoformat()
    )


def paper_evidence(card) -> dict[str, Any]:
    """Evidence card dict from a real PaperCard; only fields the card actually has."""
    notes = (
        f"source={card.source}; version={card.version_key}; text_status={card.text_status}; "
        f"card_hash={card.card_hash}; authors={', '.join(card.authors[:6])}"
    )
    out = {
        "evidence_id": re.sub(
            r"[^A-Za-z0-9_.:\-]", "_", f"{card.paper_id}.{card.version_key}"
        )[:128],
        "title": card.title,
        "abstract": card.abstract,
        "claims": [c.text for c in card.claims][:10],
        "notes": notes,
    }
    if (
        card.version_date is not None
    ):  # undated cards stay undated (rejected by historical mode)
        out["available_at"] = _iso(card.version_date)
    return {k: v for k, v in out.items() if v}


class PaperWatches:
    def __init__(
        self,
        path,
        *,
        queue: ResearchQueue,
        papers,
        clock: Callable[[], dt.datetime],
    ):
        self.path, self.queue, self.papers, self._clock = (
            path,
            queue,
            papers,
            clock,
        )
        with open_sqlite(path) as conn:
            conn.executescript(_SCHEMA)

    def _now(self) -> dt.datetime:
        return require_aware(self._clock(), "clock")

    def add(
        self,
        pod: str,
        watch_id: str,
        query: str,
        *,
        cadence_hours: float,
        executors: list[str],
        question: Optional[str] = None,
        search_limit: int = 10,
        papers_per_job: int = 3,
        max_jobs: int = 1,
    ) -> dict:
        spec = pod_spec(pod)
        query = " ".join((query or "").split())
        executors = sorted(set(executors))
        question = (
            question
            or f"What does recent literature on '{query}' imply for the {spec.pod_id} "
            f"pod's mission: {spec.mission}"
        ).strip()
        if (
            not _IDENT.match(watch_id or "")
            or not 1 <= len(query) <= 300
            or not 1 <= len(question) <= 2000
        ):
            raise ValueError("invalid watch id, topic or question")
        if not executors or any(not _IDENT.match(e) for e in executors):
            raise ValueError("at least one valid executor id is required")
        if not 1 <= cadence_hours <= 24 * 366 or not (
            1 <= search_limit <= 50
            and 1 <= papers_per_job <= 10
            and 1 <= max_jobs <= 5
        ):
            raise ValueError("cadence or bounds out of range")
        config = {
            "executors": executors,
            "search_limit": search_limit,
            "papers_per_job": papers_per_job,
            "max_jobs": max_jobs,
        }
        row = (
            spec.namespace,
            watch_id,
            query,
            question,
            int(cadence_hours * 3600),
            json.dumps(config, sort_keys=True),
        )
        now = _iso(self._now())
        with open_sqlite(self.path, immediate=True) as conn:
            old = conn.execute(
                "SELECT query, question, cadence_s, config_json FROM watches WHERE "
                "namespace=? AND watch_id=?",
                row[:2],
            ).fetchone()
            if old is not None and tuple(old) != row[2:]:
                raise ValueError(
                    f"watch {watch_id!r} already exists with different settings"
                )
            if old is None:
                conn.execute(
                    "INSERT INTO watches VALUES (?,?,?,?,?,?,?,?,NULL)",
                    (*row, now, now),
                )
        return self._watch(spec.namespace, watch_id)

    def _watch(self, namespace: str, watch_id: str) -> dict:
        return next(
            w for w in self.list(namespace) if w["watch_id"] == watch_id
        )

    def list(self, pod: Optional[str] = None) -> list[dict]:
        ns = pod_spec(pod).namespace if pod else None
        sql = (
            "SELECT namespace, watch_id, query, question, cadence_s, config_json, next_due_at, "
            "last_success_at FROM watches"
            + (" WHERE namespace=?" if ns else "")
            + " ORDER BY namespace, watch_id"
        )
        with open_sqlite(self.path) as conn:
            rows = conn.execute(sql, (ns,) if ns else ()).fetchall()
        return [
            {
                "namespace": r[0],
                "watch_id": r[1],
                "query": r[2],
                "question": r[3],
                "cadence_s": r[4],
                **json.loads(r[5]),
                "next_due_at": r[6],
                "last_success_at": r[7],
            }
            for r in rows
        ]

    def runs(self, pod: str, watch_id: str, limit: int = 20) -> list[dict]:
        with open_sqlite(self.path) as conn:
            rows = conn.execute(
                "SELECT at, status, detail_json FROM runs WHERE namespace=? AND watch_id=? "
                "ORDER BY seq DESC LIMIT ?",
                (pod_spec(pod).namespace, watch_id, limit),
            ).fetchall()
        return [{"at": r[0], "status": r[1], **json.loads(r[2])} for r in rows]

    def tick(
        self, pod: Optional[str] = None, watch_id: Optional[str] = None
    ) -> list[dict]:
        """Run every due watch once; watches not yet due are untouched."""
        now = self._now()
        return [
            self._run(w, now)
            for w in self.list(pod)
            if (watch_id is None or w["watch_id"] == watch_id)
            and w["next_due_at"] <= _iso(now)
        ]

    def _seen(self, w: dict) -> set[tuple[str, str]]:
        with open_sqlite(self.path) as conn:
            rows = conn.execute(
                "SELECT paper_id, version_key FROM seen WHERE namespace=? AND watch_id=?",
                (w["namespace"], w["watch_id"]),
            ).fetchall()
        return {(r[0], r[1]) for r in rows}

    def _record(
        self,
        w: dict,
        now: dt.datetime,
        status: str,
        detail: dict,
        next_due: dt.datetime,
    ) -> dict:
        with open_sqlite(self.path, immediate=True) as conn:
            conn.execute(
                "INSERT INTO runs (namespace, watch_id, at, status, detail_json) VALUES (?,?,?,?,?)",
                (
                    w["namespace"],
                    w["watch_id"],
                    _iso(now),
                    status,
                    json.dumps(detail, sort_keys=True),
                ),
            )
            conn.execute(
                "UPDATE watches SET next_due_at=?, last_success_at=COALESCE(?, last_success_at) "
                "WHERE namespace=? AND watch_id=?",
                (
                    _iso(next_due),
                    _iso(now) if status == "ok" else None,
                    w["namespace"],
                    w["watch_id"],
                ),
            )
        return {
            "namespace": w["namespace"],
            "watch_id": w["watch_id"],
            "status": status,
            **detail,
        }

    def _mark(
        self,
        w: dict,
        card,
        job_id: Optional[str],
        outcome: str,
        now: dt.datetime,
    ) -> None:
        with open_sqlite(self.path, immediate=True) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO seen VALUES (?,?,?,?,?,?,?)",
                (
                    w["namespace"],
                    w["watch_id"],
                    card.paper_id,
                    card.version_key,
                    job_id,
                    outcome,
                    _iso(now),
                ),
            )

    def _run(self, w: dict, now: dt.datetime) -> dict:
        jobs: list[str] = []
        detail: dict[str, Any] = {"jobs": jobs, "skipped": []}
        try:
            result = self.papers.search(w["query"], limit=w["search_limit"])
            seen, fresh = self._seen(w), []
            for card in result.cards:
                key = (card.paper_id, card.version_key)
                if key not in seen and all(
                    key != (c.paper_id, c.version_key) for c in fresh
                ):
                    fresh.append(card)
            detail.update(
                origin=result.origin,
                returned=len(result.cards),
                new=len(fresh),
                warnings=list(result.warnings)[:5],
            )
            usable = []
            for card in fresh:
                try:
                    normalise_evidence([paper_evidence(card)], "paper_card")
                    usable.append(card)
                except ValueError as exc:  # never retried or sent to a model
                    self._mark(w, card, None, "invalid_evidence", now)
                    detail["skipped"].append(
                        {"paper_id": card.paper_id, "reason": str(exc)[:200]}
                    )
            size = w["papers_per_job"]
            for chunk in [
                usable[i : i + size] for i in range(0, len(usable), size)
            ][: w["max_jobs"]]:
                job = self._enqueue(w, chunk, now)
                jobs.append(job.job_id)
                for card in chunk:
                    self._mark(w, card, job.job_id, "queued", now)
            detail["deferred"] = max(
                0, len(usable) - w["papers_per_job"] * w["max_jobs"]
            )
            return self._record(
                w, now, "ok", detail, now + dt.timedelta(seconds=w["cadence_s"])
            )
        except (
            Exception
        ) as exc:  # recorded; an incomplete tick is never reported as ok
            detail["error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:300],
            }
            retry = min(w["cadence_s"], RETRY_AFTER_FAILURE_S)
            return self._record(
                w, now, "failed", detail, now + dt.timedelta(seconds=retry)
            )

    def _enqueue(self, w: dict, cards: list, now: dt.datetime):
        # research_asof is the tick's injected clock (when the research happens); paper dates stay on the
        # evidence. The key depends only on the cards, so a retry at a later clock reuses the queued job.
        digest = hashlib.sha256(
            "|".join(
                sorted(f"{c.paper_id}@{c.card_hash}" for c in cards)
            ).encode()
        ).hexdigest()
        payload = {
            "question": w["question"],
            "research_asof": _iso(now),
            "mode": "contemporary",
            "policy": "release_gated",
            "checkpoint_id": None,
            "executors": w["executors"],
            "paper_cards": [paper_evidence(c) for c in cards],
            "dataset_cards": [],
            "source_cards": [],
            "prior_work": [],
            "watch": {"watch_id": w["watch_id"], "query": w["query"]},
        }
        key = f"watch:{w['watch_id']}:{digest[:40]}"
        try:
            return self.queue.submit(
                w["namespace"], "propose", payload, idempotency_key=key
            )
        except IdempotencyConflict:  # same cards queued by an earlier, interrupted tick at an older clock
            for job in self.queue.list(namespace=w["namespace"], limit=1000):
                if job.idempotency_key == key:
                    return job
            raise
