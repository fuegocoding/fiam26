"""Schema-constrained extractor with response caching
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 20.6, 20.10).

Extraction is short instruction-following output validated against a
schema. Every material response is persisted keyed by model/checkpoint,
prompt hash, input hashes, research date and generation parameters;
a replay uses cached decisions, and temperature zero alone is not
sufficient for reproduction.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..contracts.base import require_aware


def cache_key(
    *,
    model_id: str,
    checkpoint_revision: str,
    prompt_hash: str,
    input_hashes: dict[str, str],
    research_asof: dt.datetime,
    generation_params: dict[str, Any],
    role: str,
) -> str:
    payload = {
        "model_id": model_id,
        "checkpoint_revision": checkpoint_revision,
        "prompt_hash": prompt_hash,
        "input_hashes": dict(sorted(input_hashes.items())),
        "research_asof": require_aware(
            research_asof, "research_asof"
        ).isoformat(),
        "generation_params": dict(sorted(generation_params.items())),
        "role": role,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


@dataclass
class ExtractionRecord:
    key: str
    fields: dict[str, Any]
    created_at: dt.datetime
    from_cache: bool = False


class ExtractorCache:
    def __init__(self) -> None:
        self._store: dict[str, ExtractionRecord] = {}

    def get(self, key: str) -> Optional[ExtractionRecord]:
        return self._store.get(key)

    def put(self, key: str, record: ExtractionRecord) -> None:
        self._store[key] = record

    def __len__(self) -> int:
        return len(self._store)


class SchemaExtractor:
    def __init__(
        self,
        cache: ExtractorCache,
        model_call: Callable[[dict[str, Any], str], dict[str, Any]],
        required_fields: tuple[str, ...],
    ):
        self._cache = cache
        self._model_call = model_call
        self._required_fields = required_fields

    def extract(
        self,
        source_payload: dict[str, Any],
        *,
        prompt_hash: str,
        model_id: str,
        checkpoint_revision: str,
        research_asof: dt.datetime,
        generation_params: Optional[dict[str, Any]] = None,
        role: str = "extractor",
    ) -> ExtractionRecord:
        generation_params = generation_params or {"temperature": 0.0}
        input_hashes = {
            key: hashlib.sha256(
                json.dumps(value, sort_keys=True, default=str).encode()
            ).hexdigest()
            for key, value in source_payload.items()
        }
        key = cache_key(
            model_id=model_id,
            checkpoint_revision=checkpoint_revision,
            prompt_hash=prompt_hash,
            input_hashes=input_hashes,
            research_asof=research_asof,
            generation_params=generation_params,
            role=role,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return ExtractionRecord(
                key=key,
                fields=dict(cached.fields),
                created_at=cached.created_at,
                from_cache=True,
            )
        raw = self._model_call(source_payload, prompt_hash)
        missing = [name for name in self._required_fields if name not in raw]
        if missing:
            raise ValueError(
                f"extraction missing required fields: {', '.join(missing)}"
            )
        record = ExtractionRecord(
            key=key,
            fields={name: raw[name] for name in self._required_fields},
            created_at=require_aware(research_asof, "research_asof"),
            from_cache=False,
        )
        self._cache.put(key, record)
        return record

    def replay_uses_cache(self) -> bool:
        return True
