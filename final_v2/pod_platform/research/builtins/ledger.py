"""Resumable variant-budget and trial ledger for the built-in executors (SQLite, append-only semantics).

* A *variant slot* is keyed by ``(registration_id, variant_name)`` and bound to the spec hash. Re-running the
  same variant with the same spec returns the stored result without spending budget; re-using the name with a
  different spec is refused; an interrupted slot is resumed and the interruption is counted as a failed
  infrastructure attempt (nothing is silently forgotten).
* Trial counts are separate append-only rows by kind (spec 13.8) and are never reset.
* Design-partition accesses are logged (spec 13.1); protected partitions are not reachable from here.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from .errors import VariantBudgetExhausted

_SCHEMA = """
CREATE TABLE IF NOT EXISTS variant_slots (
    registration_id TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('started','succeeded','failed_evidence')),
    result_json TEXT,
    report_json TEXT,
    failure_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (registration_id, variant_name)
);
CREATE TRIGGER IF NOT EXISTS slots_final_immutable BEFORE UPDATE ON variant_slots
WHEN OLD.state IN ('succeeded','failed_evidence')
BEGIN SELECT RAISE(ABORT, 'finished variant slots are immutable'); END;
CREATE TRIGGER IF NOT EXISTS slots_no_delete BEFORE DELETE ON variant_slots
BEGIN SELECT RAISE(ABORT, 'variant slots are never deleted'); END;
CREATE TABLE IF NOT EXISTS trials (
    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
    registration_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    detail TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    counted_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS trials_no_update BEFORE UPDATE ON trials
BEGIN SELECT RAISE(ABORT, 'trials are append-only'); END;
CREATE TRIGGER IF NOT EXISTS trials_no_delete BEFORE DELETE ON trials
BEGIN SELECT RAISE(ABORT, 'trials are append-only'); END;
CREATE TABLE IF NOT EXISTS access_log (
    access_id INTEGER PRIMARY KEY AUTOINCREMENT,
    registration_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    partition_name TEXT NOT NULL,
    purpose TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    accessed_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS access_no_update BEFORE UPDATE ON access_log
BEGIN SELECT RAISE(ABORT, 'access log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS access_no_delete BEFORE DELETE ON access_log
BEGIN SELECT RAISE(ABORT, 'access log is append-only'); END;
"""

TRIAL_KINDS = (
    "scored_variant",
    "hyperparameter_config",
    "fold_evaluation",
    "failed_infrastructure_attempt",
    "design_partition_access",
)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Admission:
    """Outcome of ``BuiltinLedger.admit``."""

    def __init__(
        self,
        state: str,
        result: Optional[dict] = None,
        report: Optional[dict] = None,
        failure: Optional[dict] = None,
    ):
        self.state = state  # new | resumed | cached | failed
        self.result = result
        self.report = report
        self.failure = failure


class BuiltinLedger:
    def __init__(
        self,
        state_dir: "str | Path | None" = None,
        *,
        clock: Callable[[], dt.datetime] = _utc_now,
    ):
        self._clock = clock
        self._lock = threading.RLock()
        if state_dir is None:
            self.path: Optional[Path] = None
            target = ":memory:"
        else:
            root = Path(state_dir)
            root.mkdir(parents=True, exist_ok=True)
            self.path = root / "builtin_research.sqlite"
            target = str(self.path)
        self._conn = sqlite3.connect(
            target, timeout=30, isolation_level=None, check_same_thread=False
        )
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _now(self) -> str:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("ledger clock must be timezone-aware")
        return now.astimezone(dt.timezone.utc).isoformat()

    def _tx(self):
        return _Tx(self._conn, self._lock)

    # -- admission ---------------------------------------------------------

    def admit(
        self,
        registration_id: str,
        variant_name: str,
        spec_hash: str,
        *,
        budget: int,
    ) -> Admission:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT spec_hash, state, result_json, report_json, failure_json FROM variant_slots"
                " WHERE registration_id=? AND variant_name=?",
                (registration_id, variant_name),
            ).fetchone()
            if row is not None:
                if row[0] != spec_hash:
                    raise VariantBudgetExhausted(
                        f"variant {variant_name!r} of {registration_id} was already registered with a different specification; "
                        "registered specifications are immutable (use a new registration or variant)"
                    )
                if row[1] == "succeeded":
                    return Admission(
                        "cached",
                        result=json.loads(row[2]),
                        report=json.loads(row[3]),
                    )
                if row[1] == "failed_evidence":
                    return Admission("failed", failure=json.loads(row[4]))
                self._trial(
                    conn,
                    registration_id,
                    "failed_infrastructure_attempt",
                    spec_hash,
                    "interrupted attempt resumed",
                    1,
                )
                return Admission("resumed")
            used = conn.execute(
                "SELECT COUNT(*) FROM variant_slots WHERE registration_id=?",
                (registration_id,),
            ).fetchone()[0]
            if used >= budget:
                raise VariantBudgetExhausted(
                    f"registered variant budget ({budget}) exhausted for {registration_id}; a new registration is required"
                )
            now = self._now()
            conn.execute(
                "INSERT INTO variant_slots (registration_id, variant_name, spec_hash, state, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?)",
                (registration_id, variant_name, spec_hash, "started", now, now),
            )
            self._trial(
                conn,
                registration_id,
                "scored_variant",
                spec_hash,
                f"variant {variant_name}",
                1,
            )
            return Admission("new")

    def _trial(
        self,
        conn: sqlite3.Connection,
        registration_id: str,
        kind: str,
        spec_hash: str,
        detail: str,
        quantity: int,
    ) -> None:
        if kind not in TRIAL_KINDS:
            raise ValueError(f"unknown trial kind {kind!r}")
        conn.execute(
            "INSERT INTO trials (registration_id, kind, spec_hash, detail, quantity, counted_at) VALUES (?,?,?,?,?,?)",
            (
                registration_id,
                kind,
                spec_hash,
                detail,
                int(quantity),
                self._now(),
            ),
        )

    def record_trial(
        self,
        registration_id: str,
        kind: str,
        spec_hash: str,
        detail: str,
        quantity: int = 1,
    ) -> None:
        with self._tx() as conn:
            self._trial(
                conn, registration_id, kind, spec_hash, detail, quantity
            )

    def record_access(
        self,
        registration_id: str,
        dataset_id: str,
        partition: str,
        purpose: str,
        record_count: int,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO access_log (registration_id, dataset_id, partition_name, purpose, record_count, accessed_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    registration_id,
                    dataset_id,
                    partition,
                    purpose,
                    int(record_count),
                    self._now(),
                ),
            )
            self._trial(
                conn,
                registration_id,
                "design_partition_access",
                "0" * 64,
                f"{dataset_id}:{purpose}",
                1,
            )

    # -- completion --------------------------------------------------------

    def record_success(
        self,
        registration_id: str,
        variant_name: str,
        spec_hash: str,
        result: dict,
        report: dict,
        *,
        fits: int,
        ridge: bool,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE variant_slots SET state='succeeded', result_json=?, report_json=?, updated_at=?"
                " WHERE registration_id=? AND variant_name=? AND spec_hash=? AND state='started'",
                (
                    json.dumps(result, sort_keys=True, allow_nan=False),
                    json.dumps(report, sort_keys=True, allow_nan=False),
                    self._now(),
                    registration_id,
                    variant_name,
                    spec_hash,
                ),
            )
            if conn.execute("SELECT changes()").fetchone()[0] != 1:
                raise RuntimeError("variant slot was not in the started state")
            if ridge:
                self._trial(
                    conn,
                    registration_id,
                    "hyperparameter_config",
                    spec_hash,
                    f"ridge configuration of {variant_name}",
                    1,
                )
            if fits:
                self._trial(
                    conn,
                    registration_id,
                    "fold_evaluation",
                    spec_hash,
                    f"walk-forward refits of {variant_name}",
                    fits,
                )

    def record_evidence_failure(
        self,
        registration_id: str,
        variant_name: str,
        spec_hash: str,
        failure: dict,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE variant_slots SET state='failed_evidence', failure_json=?, updated_at=?"
                " WHERE registration_id=? AND variant_name=? AND spec_hash=? AND state='started'",
                (
                    json.dumps(failure, sort_keys=True, allow_nan=False),
                    self._now(),
                    registration_id,
                    variant_name,
                    spec_hash,
                ),
            )

    def record_infrastructure_failure(
        self, registration_id: str, spec_hash: str, detail: str
    ) -> None:
        self.record_trial(
            registration_id,
            "failed_infrastructure_attempt",
            spec_hash,
            detail[:300],
            1,
        )

    # -- reads -------------------------------------------------------------

    def report(self, registration_id: str, variant_name: str) -> Optional[dict]:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT report_json FROM variant_slots WHERE registration_id=? AND variant_name=?",
                (registration_id, variant_name),
            ).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def slot(
        self, registration_id: str, variant_name: str
    ) -> Optional[dict[str, Any]]:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT spec_hash, state, failure_json FROM variant_slots WHERE registration_id=? AND variant_name=?",
                (registration_id, variant_name),
            ).fetchone()
        if row is None:
            return None
        return {
            "spec_hash": row[0],
            "state": row[1],
            "failure": json.loads(row[2]) if row[2] else None,
        }

    def trial_counts(
        self, registration_id: Optional[str] = None
    ) -> dict[str, int]:
        sql = "SELECT kind, COALESCE(SUM(quantity),0) FROM trials"
        args: tuple = ()
        if registration_id is not None:
            sql += " WHERE registration_id=?"
            args = (registration_id,)
        with self._tx() as conn:
            rows = conn.execute(sql + " GROUP BY kind", args).fetchall()
        return {kind: int(total) for kind, total in sorted(rows)}

    def variants_used(self, registration_id: str) -> int:
        with self._tx() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM variant_slots WHERE registration_id=?",
                    (registration_id,),
                ).fetchone()[0]
            )


class _Tx:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._conn, self._lock = conn, lock

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
        except BaseException:
            self._lock.release()
            raise
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self._conn.execute("COMMIT")
            elif self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
        finally:
            self._lock.release()
