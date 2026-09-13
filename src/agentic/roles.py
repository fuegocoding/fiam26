"""Role prompts for the independent filing council."""
from __future__ import annotations

import json

from .contracts import ROLE_NAMES, RedactedFiling, verdict_schema_text

ROLE_PROMPTS = {
    "event_classifier": (
        "You are the event classifier in an evidence-only financial research council. "
        "Classify the primary event in the supplied redacted filing and estimate its signed "
        "economic direction. Use only the filing. Do not use outside knowledge, identify the "
        "issuer, forecast a named stock, or recommend a trade."
    ),
    "materiality_analyst": (
        "You are the materiality and cash-flow analyst in an evidence-only financial research "
        "council. Assess plausible impact on future cash flows, financing conditions, operating "
        "risk, and whether the event is surprising. Use only the supplied redacted filing. "
        "Do not identify the issuer or recommend a trade."
    ),
    "devils_advocate": (
        "You are the Devil's Advocate in an evidence-only financial research council. Search "
        "for disconfirming evidence, routine or expected explanations, ambiguous wording, "
        "unsupported causal claims, and reasons to abstain. A polished positive thesis is not "
        "evidence. Use only the supplied redacted filing and do not recommend a trade."
    ),
    "evidence_auditor": (
        "You are the evidence and audit checker in an evidence-only financial research council. "
        "Check whether the proposed interpretation is actually supported by the filing text, "
        "including internal dates, units, quantities, and evidence sufficiency. Do not infer "
        "the issuer or use outside knowledge. Abstain when the evidence is inadequate."
    ),
}


def build_role_messages(
    role: str, filing: RedactedFiling, *, tools_enabled: bool = False
) -> tuple[dict, dict]:
    if role not in ROLE_NAMES:
        raise ValueError(f"unknown council role: {role}")
    tool_note = (
        " You may use the supplied read-only search_filing and get_filing_excerpt tools to "
        "locate evidence; tool results are still part of the same redacted filing."
        if tools_enabled
        else ""
    )
    system = (
        f"Role: {role}. {ROLE_PROMPTS[role]}\n\n"
        f"{verdict_schema_text()} Every number must be numeric and within the stated ranges. "
        "Return no markdown and no additional keys."
        + tool_note
    )
    user = {
        "role": "user",
        "content": json.dumps(filing.prompt_payload(), ensure_ascii=False, sort_keys=True),
    }
    return {"role": "system", "content": system}, user
