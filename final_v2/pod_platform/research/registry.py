"""Append-only registry tables (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 22.3, 26.3).

Immutable records are insert-only: registration writes a sealed durable object and
no UPDATE or DELETE may mutate it. Trial counters and evaluation-access events are
append-only ledgers.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Iterable, Optional

from ..contracts.base import DurableModel
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse

IMMUTABLE_TABLES = frozenset(
    {
        "hypotheses",
        "experiment_specs",
        "model_versions",
        "admission_certificates",
        "forecast_publications",
        "target_books",
        "risk_reviews",
        "orders",
        "fills",
        "accounting_events",
        "decision_events",
    }
)

APPEND_ONLY_TABLES = frozenset(
    {
        "run_attempts",
        "evaluation_access",
        "trial_counts",
        "incidents",
        "research_registrations",
        "research_trials",
        "research_access",
        "research_audit",
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_specs (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_versions (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS admission_certificates (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_publications (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS target_books (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS risk_reviews (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS accounting_events (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decision_events (
    record_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_attempts (
    attempt_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    partition TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    artifact_ids TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluation_access (
    access_id TEXT PRIMARY KEY,
    partition TEXT NOT NULL,
    actor TEXT NOT NULL,
    purpose TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    record_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS trial_counts (
    counter_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    counted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_registrations (
    registration_id TEXT PRIMARY KEY,
    spec_hash TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_trials (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    registration_id TEXT,
    detail TEXT NOT NULL,
    counted_at TEXT NOT NULL,
    reservation_key TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS research_access (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    access_id TEXT NOT NULL UNIQUE,
    partition TEXT NOT NULL,
    actor TEXT NOT NULL,
    purpose TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    accessed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    subject TEXT NOT NULL,
    detail TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS research_registrations_no_update BEFORE UPDATE ON research_registrations BEGIN SELECT RAISE(ABORT, 'research registrations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS research_registrations_no_delete BEFORE DELETE ON research_registrations BEGIN SELECT RAISE(ABORT, 'research registrations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS research_trials_no_update BEFORE UPDATE ON research_trials BEGIN SELECT RAISE(ABORT, 'trial ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS research_trials_no_delete BEFORE DELETE ON research_trials BEGIN SELECT RAISE(ABORT, 'trial ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS research_access_no_update BEFORE UPDATE ON research_access BEGIN SELECT RAISE(ABORT, 'access ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS research_access_no_delete BEFORE DELETE ON research_access BEGIN SELECT RAISE(ABORT, 'access ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS research_audit_no_update BEFORE UPDATE ON research_audit BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS research_audit_no_delete BEFORE DELETE ON research_audit BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""

# Trial kinds that consume a registration's variant budget.
BUDGETED_TRIAL_KINDS = ("scored_variant", "hyperparameter_config", "seed_restart")

_PROTECTED_PARTITIONS = frozenset({"check", "final", "evaluation", "holdout"})

_DEPENDENCY_RULES = {
    "forecast_publications": (
        ("model_id", "model_versions"),
        ("admission_certificate_id", "admission_certificates"),
    ),
    "experiment_specs": (("identity.hypothesis_id", "hypotheses"),),
    "fills": (("order_id", "orders"),),
    "orders": (("target_book_id", "target_books"),),
}


class Registry:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # One Registry object per thread/process; separate objects on one file serialize via BEGIN IMMEDIATE.
        self._conn = sqlite3.connect(str(self._path), timeout=30.0)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def register(self, table: str, record: DurableModel) -> str:
        if table not in IMMUTABLE_TABLES:
            raise ValueError(f"{table} is not an immutable registry table")
        sealed = record.sealed()
        payload = sealed.model_dump(mode="json")
        self._check_dependencies(table, payload)
        try:
            self._conn.execute(
                f"INSERT INTO {table} (record_id, owner, created_at, parent_id, content_hash, payload)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sealed.stable_id,
                    sealed.owner,
                    sealed.created_at.isoformat(),
                    sealed.parent_id,
                    sealed.content_hash,
                    json.dumps(payload, sort_keys=True, default=str),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"record {sealed.stable_id} already registered in {table}; registered specifications are immutable",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    stop_condition="Register a new version with a new id instead of mutating.",
                )
            ) from exc
        self._conn.commit()
        return sealed.stable_id

    def get(self, table: str, record_id: str) -> Optional[dict]:
        row = self._conn.execute(
            f"SELECT payload FROM {table} WHERE record_id = ?", (record_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _check_dependencies(self, table: str, payload: dict) -> None:
        for dotted_key, target_table in _DEPENDENCY_RULES.get(table, ()):
            ref = payload
            for part in dotted_key.split("."):
                ref = ref.get(part) if isinstance(ref, dict) else None
            if not ref:
                raise PodPlatformError(
                    ToolResponse.failure(
                        summary=f"{table} record is missing dependency {dotted_key}",
                        code=ErrorCode.SCHEMA_MISMATCH,
                        stop_condition="Every referenced parent must exist before registration.",
                    )
                )
            if self.get(target_table, str(ref)) is None:
                raise PodPlatformError(
                    ToolResponse.failure(
                        summary=f"{table} references nonexistent {target_table} record {ref}",
                        code=ErrorCode.SCHEMA_MISMATCH,
                        stop_condition="Register the parent record first; dangling references are rejected.",
                    )
                )

    def count(self, table: str) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0])

    def record_run_attempt(
        self,
        attempt_id: str,
        idempotency_key: str,
        state: str,
        spec_hash: str,
        partition: str,
        artifact_ids: Iterable[str],
    ) -> dict:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        existing = self._conn.execute(
            "SELECT attempt_id, state, spec_hash, partition, artifact_ids FROM run_attempts"
            " WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing is not None:
            return {
                "attempt_id": existing[0],
                "state": existing[1],
                "spec_hash": existing[2],
                "partition": existing[3],
                "artifact_ids": json.loads(existing[4]),
                "idempotent_replay": True,
            }
        self._conn.execute(
            "INSERT INTO run_attempts (attempt_id, idempotency_key, state, spec_hash, partition,"
            " created_at, updated_at, artifact_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                attempt_id,
                idempotency_key,
                state,
                spec_hash,
                partition,
                now,
                now,
                json.dumps(sorted(artifact_ids)),
            ),
        )
        self._conn.commit()
        return {
            "attempt_id": attempt_id,
            "state": state,
            "spec_hash": spec_hash,
            "partition": partition,
            "artifact_ids": sorted(artifact_ids),
            "idempotent_replay": False,
        }

    def record_evaluation_access(
        self,
        access_id: str,
        partition: str,
        actor: str,
        purpose: str,
        record_count: int,
        *,
        authorized: bool = False,
    ) -> str:
        if partition in _PROTECTED_PARTITIONS and not authorized:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"partition {partition!r} is referee-controlled",
                    code=ErrorCode.PARTITION_ACCESS_DENIED,
                    stop_condition="Evaluation partitions are accessed only through Research Integrity with recorded authorization.",
                )
            )
        self._conn.execute(
            "INSERT INTO evaluation_access (access_id, partition, actor, purpose, accessed_at, record_count)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                access_id,
                partition,
                actor,
                purpose,
                dt.datetime.now(dt.timezone.utc).isoformat(),
                record_count,
            ),
        )
        self._conn.commit()
        return access_id

    def count_trial(self, counter_id: str, kind: str, spec_hash: str) -> str:
        self._conn.execute(
            "INSERT INTO trial_counts (counter_id, kind, spec_hash, counted_at) VALUES (?, ?, ?, ?)",
            (
                counter_id,
                kind,
                spec_hash,
                dt.datetime.now(dt.timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return counter_id

    def trial_count(self, kind: Optional[str] = None) -> int:
        if kind is None:
            row = self._conn.execute("SELECT COUNT(*) FROM trial_counts").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM trial_counts WHERE kind = ?", (kind,)
            ).fetchone()
        return int(row[0])

    def open_incident(self, incident_id: str, kind: str, payload: dict) -> str:
        self._conn.execute(
            "INSERT INTO incidents (incident_id, kind, opened_at, status, payload) VALUES (?, ?, ?, ?, ?)",
            (
                incident_id,
                kind,
                dt.datetime.now(dt.timezone.utc).isoformat(),
                "open",
                json.dumps(payload, sort_keys=True, default=str),
            ),
        )
        self._conn.commit()
        return incident_id

    # ---------------------------------------------------- durable research ledger
    @contextlib.contextmanager
    def _immediate(self):
        """Write transaction that takes the database write lock up front (atomic check-then-insert)."""
        if self._conn.in_transaction:  # an earlier failed statement (e.g. duplicate register) leaves an empty implicit transaction
            self._conn.commit()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.rollback()
            raise
        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def _insert_trial(self, conn, kind, spec_hash, detail, counted_at, registration_id, reservation_key) -> str:
        trial_id = f"trial_{uuid.uuid4().hex}"
        conn.execute(
            "INSERT INTO research_trials (trial_id, kind, spec_hash, registration_id, detail, counted_at, reservation_key)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (trial_id, kind, spec_hash, registration_id, detail, counted_at, reservation_key),
        )
        # mirror into the legacy counter table under the same unique id (no shared in-process counter)
        conn.execute(
            "INSERT INTO trial_counts (counter_id, kind, spec_hash, counted_at) VALUES (?, ?, ?, ?)",
            (trial_id, kind, spec_hash, self._now()),
        )
        return trial_id

    def research_register(self, registration_id: str, spec_hash: str, registered_at: str, payload: dict, trial: dict) -> None:
        """Insert a registration and its ``registered_specification`` trial in ONE transaction."""
        try:
            with self._immediate() as conn:
                conn.execute(
                    "INSERT INTO research_registrations (registration_id, spec_hash, registered_at, payload) VALUES (?, ?, ?, ?)",
                    (registration_id, spec_hash, registered_at, json.dumps(payload, sort_keys=True, default=str)),
                )
                self._insert_trial(conn, trial["kind"], trial["spec_hash"], trial["detail"], trial["counted_at"], registration_id, None)
        except sqlite3.IntegrityError as exc:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"registration {registration_id} already exists and is immutable",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    stop_condition="Amendments create new IDs linked to parents.",
                )
            ) from exc

    def research_registration(self, registration_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT payload FROM research_registrations WHERE registration_id = ?", (registration_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def research_add_trial(self, kind: str, spec_hash: str, detail: str, counted_at: str, registration_id: Optional[str] = None) -> str:
        with self._immediate() as conn:
            return self._insert_trial(conn, kind, spec_hash, detail, counted_at, registration_id, None)

    def research_reserve(
        self,
        registration_id: str,
        spec_hash: str,
        variant_budget: int,
        kind: str,
        detail: str,
        counted_at: str,
        reservation_key: str,
        *,
        extra_used: int = 0,
    ) -> dict:
        """Atomically spend one trial before execution. The same ``reservation_key`` never spends twice.

        Budget check and insert share one write transaction, so concurrent callers cannot both take the
        last slot. A refused reservation is recorded in ``research_audit`` before the error is raised.
        """
        denied = None
        with self._immediate() as conn:
            prior = conn.execute(
                "SELECT trial_id FROM research_trials WHERE reservation_key = ?", (reservation_key,)
            ).fetchone()
            if prior:
                return {"trial_id": prior[0], "replayed": True}
            if kind in BUDGETED_TRIAL_KINDS:
                marks = ",".join("?" * len(BUDGETED_TRIAL_KINDS))
                used = conn.execute(
                    f"SELECT COUNT(*) FROM research_trials WHERE spec_hash = ? AND kind IN ({marks})",
                    (spec_hash, *BUDGETED_TRIAL_KINDS),
                ).fetchone()[0] + extra_used
                if used >= variant_budget:
                    denied = f"{kind} {detail!r}: {used}/{variant_budget} variants already spent"
                    conn.execute(
                        "INSERT INTO research_audit (event, subject, detail, at) VALUES (?, ?, ?, ?)",
                        ("reservation_denied", registration_id, denied, self._now()),
                    )
            if denied is None:
                trial_id = self._insert_trial(conn, kind, spec_hash, detail, counted_at, registration_id, reservation_key)
        if denied is not None:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"variant budget exhausted for {registration_id}",
                    code=ErrorCode.PROVIDER_BUDGET_EXCEEDED,
                    stop_condition="A new registered variant is required; silent retries do not reset the budget.",
                )
            )
        return {"trial_id": trial_id, "replayed": False}

    def research_budgeted_used(self, spec_hash: str) -> int:
        marks = ",".join("?" * len(BUDGETED_TRIAL_KINDS))
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM research_trials WHERE spec_hash = ? AND kind IN ({marks})", (spec_hash, *BUDGETED_TRIAL_KINDS)
        ).fetchone()
        return int(row[0])

    def research_trial_count(self, kind: Optional[str] = None) -> int:
        if kind is None:
            row = self._conn.execute("SELECT COUNT(*) FROM research_trials").fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM research_trials WHERE kind = ?", (kind,)).fetchone()
        return int(row[0])

    def research_record_access(
        self, access_id: str, partition: str, actor: str, purpose: str, record_count: int, *, authorized: bool = False
    ) -> dict:
        """Durable outcome-access event; an identical repeat is a replay, a conflicting id or an unauthorized
        protected access is refused and logged in ``research_audit``."""
        problem = None
        code = ErrorCode.SCHEMA_MISMATCH
        with self._immediate() as conn:
            prior = conn.execute(
                "SELECT partition, actor, purpose, record_count FROM research_access WHERE access_id = ?", (access_id,)
            ).fetchone()
            if partition in _PROTECTED_PARTITIONS and not authorized:
                problem, code = f"partition {partition!r} is referee-controlled", ErrorCode.PARTITION_ACCESS_DENIED
            elif prior and tuple(prior) == (partition, actor, purpose, record_count):
                return {"access_id": access_id, "replayed": True}
            elif prior:
                problem = f"access id {access_id} already used for a different access"
            elif conn.execute("SELECT 1 FROM evaluation_access WHERE access_id = ?", (access_id,)).fetchone():
                problem = f"access id {access_id} already used in the evaluation access log"
            if problem is not None:
                conn.execute(
                    "INSERT INTO research_audit (event, subject, detail, at) VALUES (?, ?, ?, ?)",
                    ("access_refused", access_id, f"{actor}/{partition}: {problem}", self._now()),
                )
            else:
                now = self._now()
                conn.execute(
                    "INSERT INTO research_access (access_id, partition, actor, purpose, record_count, accessed_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (access_id, partition, actor, purpose, record_count, now),
                )
                conn.execute(
                    "INSERT INTO evaluation_access (access_id, partition, actor, purpose, accessed_at, record_count)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (access_id, partition, actor, purpose, now, record_count),
                )
        if problem is not None:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=problem,
                    code=code,
                    stop_condition="Evaluation partitions are accessed only through Research Integrity with recorded authorization.",
                )
            )
        return {"access_id": access_id, "replayed": False}

    def research_accesses(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT access_id, partition, actor, purpose, record_count FROM research_access ORDER BY seq"
        ).fetchall()
        return [dict(access_id=r[0], partition=r[1], actor=r[2], purpose=r[3], record_count=r[4]) for r in rows]

    def research_audit(self, event: str, subject: str, detail: str) -> None:
        with self._immediate() as conn:
            conn.execute(
                "INSERT INTO research_audit (event, subject, detail, at) VALUES (?, ?, ?, ?)",
                (event, subject, detail[:500], self._now()),
            )

    def research_audit_log(self, event: Optional[str] = None) -> list[dict]:
        query = "SELECT seq, event, subject, detail, at FROM research_audit"
        rows = self._conn.execute(query + (" WHERE event = ? ORDER BY seq" if event else " ORDER BY seq"), (event,) if event else ()).fetchall()
        return [dict(seq=r[0], event=r[1], subject=r[2], detail=r[3], at=r[4]) for r in rows]

    def verify_immutability(self) -> None:
        for table in sorted(IMMUTABLE_TABLES | APPEND_ONLY_TABLES):
            self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
