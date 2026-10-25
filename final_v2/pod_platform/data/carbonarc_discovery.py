"""Read-only Carbon Arc catalogue adapter for the bounded discovery service.

The host supplies an authenticated ``data_library`` transport. No analytical
tool, raw data entitlement, historical vintage or source approval is implied.
The adapter never receives credentials or an agent-selected tool name/URL.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any, Callable, Mapping


def _payload(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping) or value.get("isError"):
        raise ValueError("catalogue discovery failed")
    if isinstance(value.get("structuredContent"), Mapping):
        value = value["structuredContent"]
    elif "content" in value:
        texts = [
            item.get("text", "")
            for item in value["content"]
            if item.get("type") == "text"
        ]
        value = json.loads("\n".join(texts))
    if not isinstance(value, dict) or value.get("mode") not in {
        "search",
        "describe",
    }:
        raise ValueError("unexpected catalogue response")
    return value


def provider_reference(tearsheet_name: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", tearsheet_name.lower()).strip("_")[:40]
    return (
        "ca_"
        + stem
        + "_"
        + hashlib.sha256(tearsheet_name.encode()).hexdigest()[:12]
    )


class CarbonArcDiscoveryConnector:
    provider_id = "carbonarc"

    def __init__(self, data_library: Callable[[dict], Any]):
        self._call = data_library

    def discover(self, query: str, *, as_of: dt.datetime, limit: int):
        payload = _payload(self._call({"query": query}))
        if payload["mode"] != "search" or not isinstance(
            payload.get("data"), list
        ):
            raise ValueError("expected catalogue search results")
        rows = []
        for item in payload["data"][:limit]:
            if not isinstance(item, dict):
                continue
            name = item.get("tearsheet_name")
            if not isinstance(name, str) or not name or len(name) > 200:
                continue
            rows.append(
                {
                    "provider_ref": provider_reference(name),
                    "title": name[:200],
                    "description": str(item.get("overview", ""))[:600],
                    "concepts": [str(item["ecosystem"])[:80]]
                    if item.get("ecosystem")
                    else [],
                    "fields": [],
                    "entity_grain": [],
                    "dataset_hint": name[:100],
                    # A vendor's '2019-Present' claim does not establish usable vintages.
                    "history_start": None,
                    "availability": "unresolved_history",
                }
            )
        return rows

    def sample(self, provider_ref: str, *, as_of: dt.datetime, max_rows: int):
        raise PermissionError(
            "catalogue discovery does not grant raw-data sampling"
        )

    def describe(self, tearsheet_name: str) -> dict:
        """Trusted host detail lookup; not an unbudgeted strategy tool."""
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", tearsheet_name):
            raise ValueError("invalid dataset identifier")
        result = _payload(self._call({"dataset_id": tearsheet_name}))
        if result["mode"] != "describe":
            raise ValueError("expected catalogue description")
        return {
            "catalogue_documentation": result["data"],
            "formation_eligible": False,
            "historical_coverage_established": False,
            "raw_data_access_confirmed": False,
        }
