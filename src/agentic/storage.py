"""SQLite persistence for idempotent council calls and local identity joins."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator


class CouncilStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_map (
                document_id TEXT PRIMARY KEY,
                opaque_id TEXT NOT NULL,
                permno INTEGER,
                filing_date TEXT,
                eom TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS verdicts (
                document_id TEXT NOT NULL,
                permno INTEGER,
                eom TEXT,
                role TEXT NOT NULL,
                model_key TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                redaction_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (document_id, role, model_key, prompt_version, redaction_version)
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS failures (
                document_id TEXT NOT NULL,
                role TEXT NOT NULL,
                model_key TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                redaction_version TEXT NOT NULL,
                error TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (document_id, role, model_key, prompt_version, redaction_version)
            )
            """
        )
        self.con.commit()

    def save_identity(
        self, document_id: str, opaque_id: str, permno: int | None, filing_date: str, eom: str
    ) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO identity_map
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (document_id, opaque_id, permno, filing_date, eom, time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self.con.commit()

    def cached_roles(
        self,
        document_id: str,
        model_key: str,
        prompt_version: str,
        redaction_version: str,
        retry_failed: bool = False,
    ) -> set[str]:
        rows = self.con.execute(
            """
            SELECT role FROM verdicts
            WHERE document_id = ? AND model_key = ? AND prompt_version = ? AND redaction_version = ?
            """,
            (document_id, model_key, prompt_version, redaction_version),
        ).fetchall()
        roles = {row[0] for row in rows}
        if not retry_failed:
            failed = self.con.execute(
                """
                SELECT role FROM failures
                WHERE document_id = ? AND model_key = ? AND prompt_version = ? AND redaction_version = ?
                """,
                (document_id, model_key, prompt_version, redaction_version),
            ).fetchall()
            roles.update(row[0] for row in failed)
        return roles

    def save_verdict(
        self,
        document_id: str,
        permno: int | None,
        eom: str,
        role: str,
        model_key: str,
        prompt_version: str,
        redaction_version: str,
        payload: dict[str, Any],
    ) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO verdicts
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                permno,
                eom,
                role,
                model_key,
                prompt_version,
                redaction_version,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )
        self.con.commit()

    def save_failure(
        self,
        document_id: str,
        role: str,
        model_key: str,
        prompt_version: str,
        redaction_version: str,
        error: str,
    ) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO failures
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                role,
                model_key,
                prompt_version,
                redaction_version,
                error,
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )
        self.con.commit()

    def iter_verdicts(self, model_key: str, prompt_version: str, redaction_version: str) -> Iterator[dict[str, Any]]:
        rows = self.con.execute(
            """
            SELECT document_id, permno, eom, role, payload_json
            FROM verdicts
            WHERE model_key = ? AND prompt_version = ? AND redaction_version = ?
            """,
            (model_key, prompt_version, redaction_version),
        )
        for document_id, permno, eom, role, payload_json in rows:
            payload = json.loads(payload_json)
            yield {
                "document_id": document_id,
                "permno": permno,
                "eom": eom,
                "role": role,
                **payload,
            }

    def close(self) -> None:
        self.con.close()
