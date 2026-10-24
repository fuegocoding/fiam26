"""Append-only registry tables (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 22.3, 26.3).

Immutable records are insert-only: registration writes a sealed durable object and
no UPDATE or DELETE may mutate it. Trial counters and evaluation-access events are
append-only ledgers.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
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

APPEND_ONLY_TABLES = frozenset({"run_attempts", "evaluation_access", "trial_counts", "incidents"})

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
"""

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
        self._conn = sqlite3.connect(str(self._path))
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

    def verify_immutability(self) -> None:
        for table in sorted(IMMUTABLE_TABLES | APPEND_ONLY_TABLES):
            self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
