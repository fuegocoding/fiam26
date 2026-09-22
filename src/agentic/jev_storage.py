"""SQLite persistence for idempotent Jev scoring.

Separate store from the archived LLM council: keyed on
``(document_id, model_key, prompt_version, redaction_version)`` so a prompt or
model change cleanly invalidates cached rows. Full answer distributions are
stored so confidence/category thresholds can be re-tuned without re-calling
the API.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator


class JevStore:
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
            CREATE TABLE IF NOT EXISTS answers (
                document_id TEXT NOT NULL,
                permno INTEGER,
                eom TEXT,
                model_key TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                redaction_version TEXT NOT NULL,
                answers_json TEXT NOT NULL,
                encoded_json TEXT NOT NULL,
                usage_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (document_id, model_key, prompt_version, redaction_version)
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS failures (
                document_id TEXT NOT NULL,
                model_key TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                redaction_version TEXT NOT NULL,
                error TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (document_id, model_key, prompt_version, redaction_version)
            )
            """
        )
        self.con.commit()

    def save_identity(
        self, document_id: str, opaque_id: str, permno: int | None, filing_date: str, eom: str
    ) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO identity_map VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, opaque_id, permno, filing_date, eom, _now()),
        )
        self.con.commit()

    def is_cached(
        self, document_id: str, model_key: str, prompt_version: str, redaction_version: str
    ) -> bool:
        row = self.con.execute(
            """
            SELECT 1 FROM answers
            WHERE document_id = ? AND model_key = ? AND prompt_version = ? AND redaction_version = ?
            """,
            (document_id, model_key, prompt_version, redaction_version),
        ).fetchone()
        if row:
            return True
        row = self.con.execute(
            """
            SELECT 1 FROM failures
            WHERE document_id = ? AND model_key = ? AND prompt_version = ? AND redaction_version = ?
            """,
            (document_id, model_key, prompt_version, redaction_version),
        ).fetchone()
        return bool(row)

    def save_answer(
        self,
        document_id: str,
        permno: int | None,
        eom: str,
        model_key: str,
        prompt_version: str,
        redaction_version: str,
        answers: dict[str, Any],
        encoded: dict[str, Any],
        usage: dict[str, Any],
    ) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document_id,
                permno,
                eom,
                model_key,
                prompt_version,
                redaction_version,
                json.dumps(answers, ensure_ascii=False, sort_keys=True),
                json.dumps(encoded, ensure_ascii=False, sort_keys=True),
                json.dumps(usage, ensure_ascii=False, sort_keys=True),
                _now(),
            ),
        )
        self.con.execute(
            """
            DELETE FROM failures
            WHERE document_id = ? AND model_key = ? AND prompt_version = ? AND redaction_version = ?
            """,
            (document_id, model_key, prompt_version, redaction_version),
        )
        self.con.commit()

    def save_failure(
        self,
        document_id: str,
        model_key: str,
        prompt_version: str,
        redaction_version: str,
        error: str,
    ) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO failures VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, model_key, prompt_version, redaction_version, error, _now()),
        )
        self.con.commit()

    def iter_answers(
        self, model_key: str, prompt_version: str, redaction_version: str
    ) -> Iterator[dict[str, Any]]:
        rows = self.con.execute(
            """
            SELECT document_id, permno, eom, answers_json, encoded_json
            FROM answers
            WHERE model_key = ? AND prompt_version = ? AND redaction_version = ?
            """,
            (model_key, prompt_version, redaction_version),
        )
        for document_id, permno, eom, answers_json, encoded_json in rows:
            encoded = json.loads(encoded_json)
            yield {
                "document_id": document_id,
                "permno": permno,
                "eom": eom,
                **encoded,
            }

    def close(self) -> None:
        self.con.close()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
