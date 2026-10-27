"""Explicit providers for the runtime's three research roles."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class RoleProviders:
    lead: object
    challenger: object
    extractor: object

    def identities(self):
        result = {}
        for role in ("lead", "challenger", "extractor"):
            p = getattr(self, role)
            if (
                not p
                or not getattr(p, "name", None)
                or not getattr(p, "model", None)
                or not callable(getattr(p, "complete", None))
            ):
                raise ValueError(
                    "BLOCKED_BACKEND_UNAVAILABLE: explicit provider required for "
                    + role
                )
            result[role] = {
                "provider": p.name,
                "model": p.model,
                "mock": bool(getattr(p, "is_mock", False)),
            }
        return result
