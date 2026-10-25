"""Private, namespace-isolated, append-only pod research memory
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 12.5, 20.2, 20.4).

Each namespace owns its own SQLite file under ``root``; there is no API that
reads another namespace, and an existing file bound to a different namespace
is refused. Entries are never updated or deleted (triggers). Drafts are
stored with ``evaluated=0``; only a registered-executor result can be an
``evaluated_result``. Memory content is model/executor output and is treated
as untrusted data when fed back into prompts.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..contracts.base import canonical_digest, require_aware
from .jobs import NAMESPACE_RE, IdempotencyConflict, QueueError, open_sqlite, utc_now

KINDS = ("draft_proposal", "failure_note", "evaluated_result", "note")
EVALUATED_REQUIRED = ("registration_id", "executor_id", "metrics")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entries (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('draft_proposal','failure_note','evaluated_result','note')),
    evaluated INTEGER NOT NULL CHECK (evaluated IN (0,1)),
    idempotency_key TEXT NOT NULL UNIQUE,
    content_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK ((kind = 'evaluated_result') = (evaluated = 1))
);
CREATE TRIGGER IF NOT EXISTS entries_no_update BEFORE UPDATE ON entries
BEGIN SELECT RAISE(ABORT, 'memory is append-only'); END;
CREATE TRIGGER IF NOT EXISTS entries_no_delete BEFORE DELETE ON entries
BEGIN SELECT RAISE(ABORT, 'memory is append-only'); END;
"""


@dataclass(frozen=True)
class MemoryEntry:
    seq: int
    entry_id: str
    namespace: str
    kind: str
    evaluated: bool
    idempotency_key: str
    content: dict[str, Any]
    content_hash: str
    created_at: dt.datetime


class PodMemory:
    def __init__(
        self,
        root: "str | Path",
        namespace: str,
        *,
        clock: Callable[[], dt.datetime] = utc_now,
    ):
        if not NAMESPACE_RE.match(namespace or ""):
            raise QueueError(f"invalid namespace {namespace!r}")
        self.namespace = namespace
        self.root = Path(root)
        self.path = self.root / f"{namespace}.sqlite"
        if self.path.resolve().parent != self.root.resolve():
            raise QueueError("memory path escapes its root")
        self.root.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        with open_sqlite(self.path) as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT v FROM meta WHERE k = 'namespace'").fetchone()
            if row is None:
                conn.execute("INSERT OR IGNORE INTO meta VALUES ('namespace', ?)", (namespace,))
                row = conn.execute("SELECT v FROM meta WHERE k = 'namespace'").fetchone()
            if row[0] != namespace:
                raise QueueError(f"memory file is bound to namespace {row[0]!r}, not {namespace!r}")

    @staticmethod
    def _row(namespace: str, row: tuple) -> MemoryEntry:
        content = json.loads(row[5])
        if canonical_digest({"kind": row[2], "content": content}) != row[6]:
            raise QueueError(f"memory entry {row[1]} failed its content hash")
        return MemoryEntry(
            seq=row[0],
            entry_id=row[1],
            namespace=namespace,
            kind=row[2],
            evaluated=bool(row[3]),
            idempotency_key=row[4],
            content=content,
            content_hash=row[6],
            created_at=dt.datetime.strptime(row[7], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=dt.timezone.utc),
        )

    _COLS = "seq, entry_id, kind, evaluated, idempotency_key, content_json, content_hash, created_at"

    def append(
        self,
        kind: str,
        content: dict[str, Any],
        *,
        idempotency_key: Optional[str] = None,
    ) -> MemoryEntry:
        if kind not in KINDS:
            raise QueueError(f"unknown memory kind {kind!r}")
        if kind == "evaluated_result":
            missing = [name for name in EVALUATED_REQUIRED if not content.get(name)]
            if missing:
                raise QueueError(f"evaluated_result requires registered-executor fields: {missing}")
            metrics = content["metrics"]
            if not isinstance(metrics, dict) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in metrics.values()
            ):
                raise QueueError("evaluated_result metrics must be finite numbers")
            if content.get("admission", "not_admitted") != "not_admitted":
                raise QueueError("memory cannot record an admission; admission needs a promotion certificate")
            draft = self.get(str(content.get("draft_entry_id") or ""))
            if draft is None or draft.kind != "draft_proposal":
                raise QueueError("evaluated_result must derive from a stored draft_proposal in this namespace")
        try:
            blob = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise QueueError(f"memory content is not finite JSON: {exc}") from None
        # hash the JSON round trip so the digest re-verifies identically on every read
        content_hash = canonical_digest({"kind": kind, "content": json.loads(blob)})
        key = idempotency_key or content_hash
        entry_id = "mem_" + canonical_digest({"ns": self.namespace, "key": key})[:20]
        now = require_aware(self._clock(), "clock").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        with open_sqlite(self.path, immediate=True) as conn:
            row = conn.execute(
                f"SELECT {self._COLS} FROM entries WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if row is not None:
                if row[6] != content_hash:
                    raise IdempotencyConflict(f"memory key {key!r} already holds different content")
                return self._row(self.namespace, row)
            conn.execute(
                "INSERT INTO entries (entry_id, kind, evaluated, idempotency_key, content_json, content_hash, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (entry_id, kind, 1 if kind == "evaluated_result" else 0, key, blob, content_hash, now),
            )
            row = conn.execute(f"SELECT {self._COLS} FROM entries WHERE entry_id = ?", (entry_id,)).fetchone()
        return self._row(self.namespace, row)

    def get_by_key(self, idempotency_key: str) -> Optional[MemoryEntry]:
        with open_sqlite(self.path) as conn:
            row = conn.execute(
                f"SELECT {self._COLS} FROM entries WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        return self._row(self.namespace, row) if row else None

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        with open_sqlite(self.path) as conn:
            row = conn.execute(f"SELECT {self._COLS} FROM entries WHERE entry_id = ?", (entry_id,)).fetchone()
        return self._row(self.namespace, row) if row else None

    def list(self, kind: Optional[str] = None, limit: int = 200) -> list[MemoryEntry]:
        sql = f"SELECT {self._COLS} FROM entries"
        args: list[Any] = []
        if kind is not None:
            if kind not in KINDS:
                raise QueueError(f"unknown memory kind {kind!r}")
            sql += " WHERE kind = ?"
            args.append(kind)
        sql += " ORDER BY seq LIMIT ?"
        args.append(max(1, min(limit, 1000)))
        with open_sqlite(self.path) as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._row(self.namespace, r) for r in rows]

    def evaluated_results(self) -> list[MemoryEntry]:
        return self.list("evaluated_result")

    def count(self) -> int:
        with open_sqlite(self.path) as conn:
            return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def prior_work_summaries(self, limit: int = 5, max_chars: int = 240) -> list[dict[str, Any]]:
        """Short, labelled digests for prompts. Drafts are marked unevaluated."""
        out: list[dict[str, Any]] = []
        for entry in reversed(self.list(limit=1000)):
            if entry.kind not in ("draft_proposal", "evaluated_result", "failure_note"):
                continue
            text = entry.content.get("title") or entry.content.get("summary") or entry.content.get("error") or ""
            out.append(
                {
                    "entry_id": entry.entry_id,
                    "kind": entry.kind,
                    "evaluated": entry.evaluated,
                    "summary": str(text)[:max_chars],
                }
            )
            if len(out) >= limit:
                break
        return out
