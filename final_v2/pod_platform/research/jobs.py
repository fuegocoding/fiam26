"""Persistent research job queue (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 12.3-12.5, 20.9).

SQLite-backed, multi-process safe (``BEGIN IMMEDIATE`` on every mutation).
States: ``queued -> running -> succeeded | failed``. A claim is a lease with
an expiry and an unguessable token; a worker that dies leaves an expired
lease which the next ``claim`` recovers (back to ``queued`` while attempts
remain, otherwise ``failed``). Completed results and the event log are
append-only (enforced by triggers). Submissions are idempotent per
``(namespace, idempotency_key)`` and bound to the payload content hash.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from ..contracts.base import canonical_digest, require_aware

STATES = ("queued", "running", "succeeded", "failed")
NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_PAYLOAD_BYTES = 2_000_000
DEFAULT_MAX_VARIANTS = 8


class QueueError(RuntimeError):
    pass


class IdempotencyConflict(QueueError):
    """Same idempotency key submitted with different content."""


class LeaseError(QueueError):
    """Operation attempted without a live lease."""


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _ts(value: dt.datetime) -> str:
    # fixed-width UTC string: lexicographic order == chronological order
    return require_aware(value, "timestamp").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse(value: Optional[str]) -> Optional[dt.datetime]:
    if value is None:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=dt.timezone.utc)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@contextmanager
def open_sqlite(path: "str | Path", *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if immediate and conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            if immediate and conn.in_transaction:
                conn.execute("COMMIT")
    finally:
        conn.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('queued','running','succeeded','failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL CHECK (max_attempts BETWEEN 1 AND 10),
    lease_owner TEXT,
    lease_token TEXT,
    lease_expires_at TEXT,
    not_before TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error_json TEXT,
    UNIQUE (namespace, idempotency_key)
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state, not_before);
CREATE TRIGGER IF NOT EXISTS jobs_terminal_succeeded BEFORE UPDATE ON jobs
WHEN OLD.state = 'succeeded'
BEGIN SELECT RAISE(ABORT, 'succeeded jobs are immutable'); END;
CREATE TRIGGER IF NOT EXISTS jobs_no_delete BEFORE DELETE ON jobs
BEGIN SELECT RAISE(ABORT, 'jobs are never deleted'); END;
CREATE TABLE IF NOT EXISTS job_results (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id),
    result_json TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    completed_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS job_results_no_update BEFORE UPDATE ON job_results
BEGIN SELECT RAISE(ABORT, 'results are append-only'); END;
CREATE TRIGGER IF NOT EXISTS job_results_no_delete BEFORE DELETE ON job_results
BEGIN SELECT RAISE(ABORT, 'results are append-only'); END;
CREATE TABLE IF NOT EXISTS job_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    event TEXT NOT NULL,
    at TEXT NOT NULL,
    detail_json TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS job_events_no_update BEFORE UPDATE ON job_events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS job_events_no_delete BEFORE DELETE ON job_events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
"""

_COLUMNS = (
    "job_id, namespace, kind, payload_json, payload_hash, idempotency_key, state, attempts, "
    "max_attempts, lease_owner, lease_token, lease_expires_at, not_before, created_at, updated_at, error_json"
)


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    namespace: str
    kind: str
    payload: dict[str, Any]
    payload_hash: str
    idempotency_key: str
    state: str
    attempts: int
    max_attempts: int
    lease_owner: Optional[str]
    lease_token: Optional[str]
    lease_expires_at: Optional[dt.datetime]
    not_before: dt.datetime
    created_at: dt.datetime
    updated_at: dt.datetime
    error: Optional[dict[str, Any]]

    @property
    def terminal(self) -> bool:
        return self.state in ("succeeded", "failed")


def _row_to_job(row: Sequence[Any]) -> JobRecord:
    return JobRecord(
        job_id=row[0],
        namespace=row[1],
        kind=row[2],
        payload=json.loads(row[3]),
        payload_hash=row[4],
        idempotency_key=row[5],
        state=row[6],
        attempts=row[7],
        max_attempts=row[8],
        lease_owner=row[9],
        lease_token=row[10],
        lease_expires_at=_parse(row[11]),
        not_before=_parse(row[12]),  # type: ignore[arg-type]
        created_at=_parse(row[13]),  # type: ignore[arg-type]
        updated_at=_parse(row[14]),  # type: ignore[arg-type]
        error=json.loads(row[15]) if row[15] else None,
    )


class ResearchQueue:
    def __init__(
        self,
        path: "str | Path",
        *,
        clock: Callable[[], dt.datetime] = utc_now,
        max_variants: int = DEFAULT_MAX_VARIANTS,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self.max_variants = max_variants
        with open_sqlite(self.path) as conn:
            conn.executescript(_SCHEMA)

    def _now(self) -> dt.datetime:
        return require_aware(self._clock(), "clock")

    @staticmethod
    def _event(conn: sqlite3.Connection, job_id: str, event: str, at: dt.datetime, detail: Any = None) -> None:
        conn.execute(
            "INSERT INTO job_events (job_id, event, at, detail_json) VALUES (?,?,?,?)",
            (job_id, event, _ts(at), _json(detail or {})),
        )

    # -- submission --------------------------------------------------------

    def submit(
        self,
        namespace: str,
        kind: str,
        payload: dict[str, Any],
        *,
        idempotency_key: Optional[str] = None,
        max_attempts: int = 2,
        not_before: Optional[dt.datetime] = None,
    ) -> JobRecord:
        if not NAMESPACE_RE.match(namespace or ""):
            raise QueueError(f"invalid namespace {namespace!r}")
        if not re.match(r"^[a-z][a-z0-9_]{0,63}$", kind or ""):
            raise QueueError(f"invalid job kind {kind!r}")
        if not 1 <= max_attempts <= 10:
            raise QueueError("max_attempts must be in [1, 10]")
        variants = payload.get("variants")
        if variants is not None and (not isinstance(variants, list) or len(variants) > self.max_variants):
            raise QueueError(f"experiment variants are bounded at {self.max_variants}")
        try:
            blob = _json(payload)
        except (TypeError, ValueError) as exc:
            raise QueueError(f"payload is not finite JSON: {exc}") from None
        if len(blob.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise QueueError("payload too large")
        payload_hash = canonical_digest({"namespace": namespace, "kind": kind, "payload": payload})
        key = idempotency_key or payload_hash
        if not 1 <= len(key) <= 200:
            raise QueueError("idempotency key must be 1-200 characters")
        now = self._now()
        nb = require_aware(not_before, "not_before") if not_before else now
        with open_sqlite(self.path, immediate=True) as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE namespace = ? AND idempotency_key = ?",
                (namespace, key),
            ).fetchone()
            if row is not None:
                if row[4] != payload_hash:
                    raise IdempotencyConflict(
                        f"idempotency key {key!r} already used with different content"
                    )
                return _row_to_job(row)
            job_id = f"job_{uuid.uuid4().hex[:16]}"
            conn.execute(
                "INSERT INTO jobs (job_id, namespace, kind, payload_json, payload_hash, idempotency_key, state, "
                "attempts, max_attempts, not_before, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, namespace, kind, blob, payload_hash, key, "queued", 0, max_attempts, _ts(nb), _ts(now), _ts(now)),
            )
            self._event(conn, job_id, "submitted", now, {"kind": kind})
            row = conn.execute(f"SELECT {_COLUMNS} FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _row_to_job(row)

    # -- claiming ----------------------------------------------------------

    def _recover_expired(
        self, conn: sqlite3.Connection, now: dt.datetime, namespace: Optional[str] = None
    ) -> list[str]:
        recovered: list[str] = []
        sql = (
            "SELECT job_id, attempts, max_attempts, lease_owner FROM jobs "
            "WHERE state = 'running' AND lease_expires_at < ?"
        )
        args: list[Any] = [_ts(now)]
        if namespace is not None:  # a namespace-scoped caller never touches another pod's jobs
            sql += " AND namespace = ?"
            args.append(namespace)
        rows = conn.execute(sql, args).fetchall()
        for job_id, attempts, max_attempts, owner in rows:
            if attempts >= max_attempts:
                err = {"message": "lease expired and attempts exhausted", "retryable": False, "lease_owner": owner}
                conn.execute(
                    "UPDATE jobs SET state='failed', lease_owner=NULL, lease_token=NULL, lease_expires_at=NULL, "
                    "error_json=?, updated_at=? WHERE job_id=?",
                    (_json(err), _ts(now), job_id),
                )
                self._event(conn, job_id, "lease_expired_failed", now, err)
            else:
                conn.execute(
                    "UPDATE jobs SET state='queued', lease_owner=NULL, lease_token=NULL, lease_expires_at=NULL, "
                    "updated_at=? WHERE job_id=?",
                    (_ts(now), job_id),
                )
                self._event(
                    conn, job_id, "lease_expired_requeued", now, {"lease_owner": owner, "attempts": attempts}
                )
            recovered.append(job_id)
        return recovered

    def recover_expired(self, namespace: Optional[str] = None) -> list[str]:
        with open_sqlite(self.path, immediate=True) as conn:
            return self._recover_expired(conn, self._now(), namespace)

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 300.0,
        namespace: Optional[str] = None,
        kinds: Optional[Sequence[str]] = None,
    ) -> Optional[JobRecord]:
        if not worker_id or lease_seconds <= 0 or lease_seconds > 86400:
            raise QueueError("worker_id required and lease_seconds must be in (0, 86400]")
        now = self._now()
        with open_sqlite(self.path, immediate=True) as conn:
            self._recover_expired(conn, now, namespace)
            where =["state = 'queued'", "not_before <= ?"]
            args: list[Any] = [_ts(now)]
            if namespace is not None:
                where.append("namespace = ?")
                args.append(namespace)
            if kinds:
                where.append("kind IN (%s)" % ",".join("?" * len(kinds)))
                args.extend(kinds)
            row = conn.execute(
                f"SELECT job_id FROM jobs WHERE {' AND '.join(where)} ORDER BY created_at, rowid LIMIT 1", args
            ).fetchone()
            if row is None:
                return None
            token = secrets.token_hex(16)
            expires = now + dt.timedelta(seconds=lease_seconds)
            conn.execute(
                "UPDATE jobs SET state='running', attempts=attempts+1, lease_owner=?, lease_token=?, "
                "lease_expires_at=?, updated_at=? WHERE job_id=?",
                (worker_id, token, _ts(expires), _ts(now), row[0]),
            )
            self._event(conn, row[0], "claimed", now, {"worker": worker_id})
            full = conn.execute(f"SELECT {_COLUMNS} FROM jobs WHERE job_id = ?", (row[0],)).fetchone()
        return _row_to_job(full)

    def _require_lease(self, conn: sqlite3.Connection, job_id: str, token: str, now: dt.datetime) -> JobRecord:
        row = conn.execute(f"SELECT {_COLUMNS} FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise LeaseError(f"unknown job {job_id}")
        job = _row_to_job(row)
        if job.state != "running" or not token or job.lease_token != token:
            raise LeaseError(f"job {job_id} is not leased to this token (state={job.state})")
        if job.lease_expires_at is None or job.lease_expires_at < now:
            raise LeaseError(f"lease on {job_id} has expired")
        return job

    def heartbeat(self, job_id: str, lease_token: str, lease_seconds: float = 300.0) -> dt.datetime:
        if lease_seconds <= 0 or lease_seconds > 86400:
            raise QueueError("lease_seconds must be in (0, 86400]")
        now = self._now()
        with open_sqlite(self.path, immediate=True) as conn:
            self._require_lease(conn, job_id, lease_token, now)
            expires = now + dt.timedelta(seconds=lease_seconds)
            conn.execute(
                "UPDATE jobs SET lease_expires_at=?, updated_at=? WHERE job_id=?", (_ts(expires), _ts(now), job_id)
            )
            self._event(conn, job_id, "heartbeat", now)
        return expires

    # -- completion --------------------------------------------------------

    def complete(self, job_id: str, lease_token: str, result: dict[str, Any]) -> None:
        now = self._now()
        try:
            blob = _json(result)
        except (TypeError, ValueError) as exc:
            raise QueueError(f"result is not finite JSON: {exc}") from None
        with open_sqlite(self.path, immediate=True) as conn:
            self._require_lease(conn, job_id, lease_token, now)
            conn.execute(
                "INSERT INTO job_results (job_id, result_json, result_hash, completed_at) VALUES (?,?,?,?)",
                (job_id, blob, canonical_digest(result), _ts(now)),
            )
            conn.execute(
                "UPDATE jobs SET state='succeeded', lease_owner=NULL, lease_token=NULL, lease_expires_at=NULL, "
                "error_json=NULL, updated_at=? WHERE job_id=?",
                (_ts(now), job_id),
            )
            self._event(conn, job_id, "succeeded", now)

    def fail(
        self,
        job_id: str,
        lease_token: str,
        error: dict[str, Any],
        *,
        retryable: bool = False,
        retry_delay_s: float = 0.0,
    ) -> JobRecord:
        """Record a failure. Retry only if transient *and* attempts remain."""
        now = self._now()
        err = dict(error)
        err["retryable"] = bool(retryable)
        with open_sqlite(self.path, immediate=True) as conn:
            job = self._require_lease(conn, job_id, lease_token, now)
            if retryable and job.attempts < job.max_attempts:
                nb = now + dt.timedelta(seconds=max(0.0, retry_delay_s))
                conn.execute(
                    "UPDATE jobs SET state='queued', lease_owner=NULL, lease_token=NULL, lease_expires_at=NULL, "
                    "not_before=?, error_json=?, updated_at=? WHERE job_id=?",
                    (_ts(nb), _json(err), _ts(now), job_id),
                )
                self._event(conn, job_id, "failed_will_retry", now, err)
            else:
                conn.execute(
                    "UPDATE jobs SET state='failed', lease_owner=NULL, lease_token=NULL, lease_expires_at=NULL, "
                    "error_json=?, updated_at=? WHERE job_id=?",
                    (_json(err), _ts(now), job_id),
                )
                self._event(conn, job_id, "failed", now, err)
            row = conn.execute(f"SELECT {_COLUMNS} FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _row_to_job(row)

    # -- reads -------------------------------------------------------------

    def get(self, job_id: str) -> Optional[JobRecord]:
        with open_sqlite(self.path) as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list(
        self, *, namespace: Optional[str] = None, state: Optional[str] = None, limit: int = 100
    ) -> list[JobRecord]:
        if state is not None and state not in STATES:
            raise QueueError(f"unknown state {state!r}")
        where, args = ["1=1"], []
        if namespace is not None:
            where.append("namespace = ?")
            args.append(namespace)
        if state is not None:
            where.append("state = ?")
            args.append(state)
        with open_sqlite(self.path) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM jobs WHERE {' AND '.join(where)} ORDER BY created_at, rowid LIMIT ?",
                [*args, max(1, min(limit, 1000))],
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def result(self, job_id: str) -> Optional[dict[str, Any]]:
        with open_sqlite(self.path) as conn:
            row = conn.execute("SELECT result_json FROM job_results WHERE job_id = ?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def events(self, job_id: str) -> list[dict[str, Any]]:
        with open_sqlite(self.path) as conn:
            rows = conn.execute(
                "SELECT seq, event, at, detail_json FROM job_events WHERE job_id = ? ORDER BY seq", (job_id,)
            ).fetchall()
        return [{"seq": r[0], "event": r[1], "at": r[2], "detail": json.loads(r[3])} for r in rows]

    def counts(self, namespace: Optional[str] = None) -> dict[str, int]:
        out = {state: 0 for state in STATES}
        with open_sqlite(self.path) as conn:
            if namespace is None:
                rows = conn.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state").fetchall()
            else:
                rows = conn.execute(
                    "SELECT state, COUNT(*) FROM jobs WHERE namespace = ? GROUP BY state", (namespace,)
                ).fetchall()
        out.update({state: count for state, count in rows})
        return out
