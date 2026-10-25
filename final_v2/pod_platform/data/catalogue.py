"""Catalogue import and manifest (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 8.4, 22.4; W02).

The catalogue is documentation and samples. Importing it never establishes
entitlement or historical coverage; that is the source card's job.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Iterable

from ..contracts.base import canonical_digest
from ..contracts.sources import SourceCard


class CatalogueManifest:
    def __init__(self) -> None:
        self._cards: dict[str, SourceCard] = {}
        self._raw_hashes: dict[str, str] = {}

    def import_card_bytes(self, raw: bytes) -> SourceCard:
        payload = json.loads(raw.decode("utf-8"))
        card = SourceCard.model_validate(payload)
        return self.import_card(
            card, raw_digest=hashlib.sha256(raw).hexdigest()
        )

    def import_card(
        self, card: SourceCard, *, raw_digest: str | None = None
    ) -> SourceCard:
        if card.source_id in self._cards:
            raise ValueError(f"duplicate source_id {card.source_id}")
        self._cards[card.source_id] = card
        self._raw_hashes[card.source_id] = raw_digest or canonical_digest(
            card.model_dump(mode="json", exclude={"content_hash"})
        )
        return card

    def import_directory(self, directory: str | Path) -> int:
        count = 0
        for path in sorted(Path(directory).glob("*.json")):
            self.import_card_bytes(path.read_bytes())
            count += 1
        return count

    def card(self, source_id: str) -> SourceCard:
        return self._cards[source_id]

    def source_ids(self) -> list[str]:
        return sorted(self._cards)

    def topic_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for card in self._cards.values():
            counts[card.topic] = counts.get(card.topic, 0) + 1
        return dict(sorted(counts.items()))

    def content_hashes(self) -> dict[str, str]:
        return dict(sorted(self._raw_hashes.items()))

    def manifest(self, snapshot_id: str) -> dict:
        return {
            "snapshot_id": snapshot_id,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "entry_count": len(self._cards),
            "topic_counts": self.topic_counts(),
            "content_hashes": self.content_hashes(),
            "formation_eligible": sorted(
                card.source_id
                for card in self._cards.values()
                if card.formation_eligible
            ),
        }

    def __len__(self) -> int:
        return len(self._cards)


def summarize_entries(entries: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        topic = entry.get("topic", "unknown")
        counts[topic] = counts.get(topic, 0) + 1
    return dict(sorted(counts.items()))
