"""Persistent SQLite store: version rows (metadata cards), content, query cache, artifacts.

Rows are keyed ``(paper_id, version_key)``; identifiers are data, never file paths.
Alias rows map DOI / arXiv / OpenAlex ids to one canonical ``paper_id`` so a
duplicate DOI across providers collapses to one paper while versions stay apart.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import PaperCard, PapersError, canonical_json, ensure_aware, sha256_text

_SCHEMA = """
CREATE TABLE IF NOT EXISTS versions(
  paper_id TEXT NOT NULL, version_key TEXT NOT NULL, version INTEGER,
  version_date TEXT, retrieved_at TEXT NOT NULL, card_json TEXT NOT NULL,
  PRIMARY KEY(paper_id, version_key));
CREATE TABLE IF NOT EXISTS aliases(alias TEXT PRIMARY KEY, paper_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS contents(
  content_hash TEXT PRIMARY KEY, text TEXT NOT NULL, original_hash TEXT NOT NULL,
  truncated INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS queries(
  qkey TEXT PRIMARY KEY, results_json TEXT NOT NULL, fetched_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts(
  kind TEXT NOT NULL, artifact_id TEXT NOT NULL, body_json TEXT NOT NULL,
  created_at TEXT NOT NULL, PRIMARY KEY(kind, artifact_id));
"""
_RANK = {"arxiv": 0, "doi": 1, "openalex": 2, "local": 3}


def _rank(pid: str) -> int:
    return _RANK.get(pid.split(":", 1)[0], 9)


class PaperStore:
    def __init__(self, path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    # ----------------------------------------------------------- aliases
    def resolve(self, alias: str) -> Optional[str]:
        r = self._db.execute("SELECT paper_id FROM aliases WHERE alias=?", (alias,)).fetchone()
        return r[0] if r else None

    def _merge_into(self, old: str, new: str) -> None:
        for (key, js) in self._db.execute(
                "SELECT version_key, card_json FROM versions WHERE paper_id=?", (old,)).fetchall():
            exists = self._db.execute("SELECT 1 FROM versions WHERE paper_id=? AND version_key=?",
                                      (new, key)).fetchone()
            if not exists:
                d = json.loads(js)
                d["paper_id"] = new
                card = PaperCard.model_validate(d)
                self._db.execute(
                    "UPDATE versions SET paper_id=?, card_json=? WHERE paper_id=? AND version_key=?",
                    (new, card.sealed().model_dump_json(), old, key))
        self._db.execute("DELETE FROM versions WHERE paper_id=?", (old,))
        self._db.execute("UPDATE aliases SET paper_id=? WHERE paper_id=?", (new, old))

    def register_aliases(self, card: PaperCard) -> str:
        """Link all ids of ``card``; merge previously separate papers. Return canonical id."""
        aliases = {card.paper_id}
        if card.doi:
            aliases.add(f"doi:{card.doi}")
        if card.arxiv_id:
            aliases.add(f"arxiv:{card.arxiv_id}")
        if card.openalex_id:
            aliases.add(f"openalex:{card.openalex_id}")
        owners = {self.resolve(a) or a for a in aliases}
        canon = sorted(owners, key=lambda p: (_rank(p), p))[0]
        for o in owners - {canon}:
            self._merge_into(o, canon)
        for a in aliases | owners:
            self._db.execute("INSERT OR REPLACE INTO aliases(alias, paper_id) VALUES(?,?)", (a, canon))
        return canon

    # ------------------------------------------------------------- cards
    def upsert_card(self, card: PaperCard) -> PaperCard:
        """Store ``card``; keep any claims/content already attached to the same row."""
        with self._db:
            canon = self.register_aliases(card)
            card = card.model_copy(update={"paper_id": canon})
            old = self.get_row(canon, card.version_key)
            if old is not None:
                keep = {"latest_version": card.latest_version or old.latest_version}
                if old.abstract_hash == card.abstract_hash:
                    keep.update(claims=old.claims, content_hash=old.content_hash,
                                original_hash=old.original_hash, content_chars=old.content_chars,
                                content_truncated=old.content_truncated,
                                text_status=old.text_status, access_status=old.access_status)
                card = card.model_copy(update=keep)
            self._write(card)
        return card.sealed()

    def _write(self, card: PaperCard) -> None:
        card = card.sealed()
        self._db.execute(
            "INSERT OR REPLACE INTO versions VALUES(?,?,?,?,?,?)",
            (card.paper_id, card.version_key, card.version,
             card.version_date.isoformat() if card.version_date else None,
             card.retrieved_at.isoformat(), card.model_dump_json()))

    def save_card(self, card: PaperCard) -> PaperCard:
        with self._db:
            self._write(card)
        return card.sealed()

    @staticmethod
    def _load(js: str) -> PaperCard:
        """Parse a stored card and re-verify its seal (a damaged row must not become evidence)."""
        card = PaperCard.model_validate_json(js)
        if card.card_hash != card.compute_hash():
            raise PapersError(f"stored card {card.paper_id}/{card.version_key} failed its card hash")
        return card

    def get_row(self, paper_id: str, version_key: str) -> Optional[PaperCard]:
        r = self._db.execute("SELECT card_json FROM versions WHERE paper_id=? AND version_key=?",
                             (paper_id, version_key)).fetchone()
        return self._load(r[0]) if r else None

    def rows(self, paper_id: str) -> list[PaperCard]:
        return [self._load(r[0]) for r in self._db.execute(
            "SELECT card_json FROM versions WHERE paper_id=?", (paper_id,)).fetchall()]

    def all_cards(self, limit: int = 20000) -> list[PaperCard]:
        return [self._load(r[0]) for r in self._db.execute(
            "SELECT card_json FROM versions LIMIT ?", (limit,)).fetchall()]

    # ----------------------------------------------------------- content
    def put_content(self, text: str, content_hash: str, original_hash: str, truncated: bool) -> None:
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO contents VALUES(?,?,?,?)",
                             (content_hash, text, original_hash, int(truncated)))

    def get_content(self, content_hash: str) -> Optional[str]:
        r = self._db.execute("SELECT text FROM contents WHERE content_hash=?", (content_hash,)).fetchone()
        if r is None:
            return None
        if sha256_text(r[0]) != content_hash:
            raise PapersError("stored content failed its content hash")
        return r[0]

    # ----------------------------------------------------------- queries
    def put_query(self, qkey: str, results: list[dict], fetched_at: datetime) -> None:
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO queries VALUES(?,?,?)",
                             (qkey, canonical_json(results), ensure_aware(fetched_at).isoformat()))

    def get_query(self, qkey: str) -> Optional[list[dict]]:
        r = self._db.execute("SELECT results_json FROM queries WHERE qkey=?", (qkey,)).fetchone()
        return json.loads(r[0]) if r else None

    # --------------------------------------------------------- artifacts
    def put_artifact(self, kind: str, artifact_id: str, body: dict, created_at: datetime) -> None:
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?)",
                             (kind, artifact_id, canonical_json(body),
                              ensure_aware(created_at).isoformat()))

    def get_artifact(self, kind: str, artifact_id: str) -> Optional[dict]:
        r = self._db.execute("SELECT body_json FROM artifacts WHERE kind=? AND artifact_id=?",
                             (kind, artifact_id)).fetchone()
        return json.loads(r[0]) if r else None


__all__ = ["PaperStore", "PapersError"]
