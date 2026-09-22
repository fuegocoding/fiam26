"""Read-only tools over one already-redacted filing."""
from __future__ import annotations

from typing import Any

from ..contracts import RedactedFiling


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_filing",
            "description": "Find relevant passages in the supplied redacted filing text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Case-insensitive text to find."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_filing_excerpt",
            "description": "Read a bounded character excerpt from the supplied redacted filing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                },
                "required": ["start", "end"],
                "additionalProperties": False,
            },
        },
    },
]


class RedactedFilingTools:
    """Tool dispatcher whose only source is the redacted filing body."""

    def __init__(self, filing: RedactedFiling):
        self.text = filing.text

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_filing":
            return self.search_filing(
                str(arguments.get("query", "")), int(arguments.get("max_results", 5))
            )
        if name == "get_filing_excerpt":
            return self.get_filing_excerpt(
                int(arguments.get("start", 0)), int(arguments.get("end", 0))
            )
        raise ValueError(f"unknown filing tool: {name}")

    def search_filing(self, query: str, max_results: int = 5) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"matches": []}
        max_results = max(1, min(max_results, 5))
        haystack = self.text.casefold()
        needle = query.casefold()
        matches = []
        cursor = 0
        while len(matches) < max_results:
            start = haystack.find(needle, cursor)
            if start < 0:
                break
            end = start + len(query)
            excerpt_start = max(0, start - 180)
            excerpt_end = min(len(self.text), end + 420)
            matches.append(
                {"start": start, "end": end, "excerpt": self.text[excerpt_start:excerpt_end]}
            )
            cursor = end
        return {"query": query, "matches": matches}

    def get_filing_excerpt(self, start: int, end: int) -> dict[str, Any]:
        start = max(0, min(start, len(self.text)))
        end = max(start, min(end, start + 1200, len(self.text)))
        return {"start": start, "end": end, "excerpt": self.text[start:end]}
