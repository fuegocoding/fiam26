"""Tiny content-hash artifact index (existence receipts for evidence references).

Forecast evidence and uncertainty references must resolve to a real, hashed
artifact (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.2, 21.6). The index stores
metadata only; private payloads stay in the owning namespace's workspace.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ..contracts.errors import ErrorCode
from .common import UTC, digest, fail, now_utc

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    namespace TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    meta_json TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS artifacts_no_update BEFORE UPDATE ON artifacts
BEGIN SELECT RAISE(ABORT, 'artifact receipts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_no_delete BEFORE DELETE ON artifacts
BEGIN SELECT RAISE(ABORT, 'artifact receipts are never deleted'); END;
"""


class ArtifactIndex:
    def __init__(self, path: "str | Path", clock=now_utc):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), timeout=30)

    def register(self, artifact_id: str, kind: str, namespace: str, sha256: str, meta: Optional[dict[str, Any]] = None) -> str:
        """Idempotent for identical content; changed content under the same id fails."""
        meta_json = json.dumps(meta or {}, sort_keys=True, default=str)
        conn = self._connect()
        try:
            row = conn.execute("SELECT sha256, kind, namespace FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
            if row is not None:
                if row[0] == sha256 and row[1] == kind and row[2] == namespace:
                    return artifact_id
                fail(ErrorCode.SCHEMA_MISMATCH, f"artifact {artifact_id} already registered with different content",
                     "Artifacts are immutable; register a new id.")
            with conn:
                conn.execute(
                    "INSERT INTO artifacts (artifact_id, kind, namespace, sha256, created_at, meta_json) VALUES (?,?,?,?,?,?)",
                    (artifact_id, kind, namespace, sha256, self._clock().astimezone(UTC).isoformat(), meta_json),
                )
        finally:
            conn.close()
        return artifact_id

    def register_payload(self, artifact_id: str, kind: str, namespace: str, payload: Any) -> str:
        return self.register(artifact_id, kind, namespace, digest(payload), meta=payload if isinstance(payload, dict) else None)

    def exists(self, artifact_id: str) -> bool:
        return self.get(artifact_id) is not None

    def get(self, artifact_id: str) -> Optional[dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT artifact_id, kind, namespace, sha256, created_at, meta_json FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            "artifact_id": row[0],
            "kind": row[1],
            "namespace": row[2],
            "sha256": row[3],
            "created_at": row[4],
            "meta": json.loads(row[5]),
        }

    def list(self, kind: Optional[str] = None) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            if kind is None:
                rows = conn.execute("SELECT artifact_id FROM artifacts ORDER BY artifact_id").fetchall()
            else:
                rows = conn.execute("SELECT artifact_id FROM artifacts WHERE kind = ? ORDER BY artifact_id", (kind,)).fetchall()
        finally:
            conn.close()
        return [self.get(r[0]) for r in rows]  # type: ignore[misc]
