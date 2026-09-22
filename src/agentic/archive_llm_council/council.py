"""Parallel independent council execution for one redacted filing."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable

from ..contracts import AgentVerdict, ROLE_NAMES, RedactedFiling, validate_verdict
from .roles import build_role_messages
from .tools import RedactedFilingTools, TOOL_SPECS


@dataclass(frozen=True)
class RoleFailure:
    role: str
    error: str


@dataclass(frozen=True)
class CouncilResult:
    verdicts: tuple[AgentVerdict, ...]
    failures: tuple[RoleFailure, ...]


class CouncilRunner:
    def __init__(self, llm: Any, roles: Iterable[str] = ROLE_NAMES, *, use_tools: bool = False):
        self.llm = llm
        self.roles = tuple(roles)
        self.use_tools = use_tools
        invalid = set(self.roles) - set(ROLE_NAMES)
        if invalid:
            raise ValueError(f"unknown council roles: {sorted(invalid)}")

    def _score_role(self, role: str, filing: RedactedFiling) -> AgentVerdict:
        system_message, user_message = build_role_messages(
            role, filing, tools_enabled=self.use_tools
        )
        request_id = f"council-{filing.opaque_id.lower()}-{role}"
        if self.use_tools:
            filing_tools = RedactedFilingTools(filing)
            result = self.llm.complete_json_with_tools(
                system_message["content"],
                user_message["content"],
                request_id=request_id,
                tools=TOOL_SPECS,
                dispatch=filing_tools.dispatch,
                max_tool_calls=4,
            )
        else:
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
            if self.use_tools:
                repaired = self.llm.complete_json_with_tools(
                    system_message["content"],
                    repair_user,
                    request_id=request_id + "-repair",
                    tools=TOOL_SPECS,
                    dispatch=RedactedFilingTools(filing).dispatch,
                    max_tool_calls=4,
                )
            else:
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
