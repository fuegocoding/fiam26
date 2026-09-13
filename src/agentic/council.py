"""Parallel independent council execution for one redacted filing."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable

from .contracts import AgentVerdict, ROLE_NAMES, RedactedFiling, validate_verdict
from .roles import build_role_messages


@dataclass(frozen=True)
class RoleFailure:
    role: str
    error: str


@dataclass(frozen=True)
class CouncilResult:
    verdicts: tuple[AgentVerdict, ...]
    failures: tuple[RoleFailure, ...]


class CouncilRunner:
    def __init__(self, llm: Any, roles: Iterable[str] = ROLE_NAMES):
        self.llm = llm
        self.roles = tuple(roles)
        invalid = set(self.roles) - set(ROLE_NAMES)
        if invalid:
            raise ValueError(f"unknown council roles: {sorted(invalid)}")

    def _score_role(self, role: str, filing: RedactedFiling) -> AgentVerdict:
        system_message, user_message = build_role_messages(role, filing)
        request_id = f"council-{filing.opaque_id.lower()}-{role}"
        result = self.llm.complete_json(
            system_message["content"], user_message["content"], request_id=request_id
        )
        try:
            return validate_verdict(result.content, role=role)
        except ValueError as first_error:
            repair_user = (
                user_message["content"]
                + "\n\nYour previous response was invalid. Correct it and return only the exact JSON contract. "
                + str(first_error)
            )
            repaired = self.llm.complete_json(
                system_message["content"], repair_user, request_id=request_id + "-repair"
            )
            return validate_verdict(repaired.content, role=role)

    def score_filing(self, filing: RedactedFiling) -> CouncilResult:
        verdicts: list[AgentVerdict] = []
        failures: list[RoleFailure] = []
        with ThreadPoolExecutor(max_workers=max(len(self.roles), 1)) as pool:
            futures = {pool.submit(self._score_role, role, filing): role for role in self.roles}
            for future in as_completed(futures):
                role = futures[future]
                try:
                    verdicts.append(future.result())
                except Exception as exc:
                    failures.append(RoleFailure(role=role, error=f"{type(exc).__name__}: {exc}"))
        verdicts.sort(key=lambda verdict: verdict.role)
        failures.sort(key=lambda failure: failure.role)
        return CouncilResult(tuple(verdicts), tuple(failures))
