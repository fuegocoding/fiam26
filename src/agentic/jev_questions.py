"""Frozen Jev System One question set for 8-K event scoring (``jev-v1``).

Design rules (TypeSafe guidance + jev-1.13 jaggedness notes):
- One atomic judgment per question. No personas, no fused judgments.
- Literal wording: boundary cases live in the criteria, not in implied intent.
- Jev never counts, does math, compares dates, or generates text — all
  combination logic (signed products, OR-gates, counts) lives in
  :mod:`src.agentic.jev_encode` and :mod:`src.agentic.jev_aggregate`.
- ``state`` is the redacted filing only; detected SEC item codes are passed
  as explicitly *unverified hints*.

Request shape follows the TypeSafe System One API, served here through
OpenRouter ``POST {base}/api/v1/systemone`` with ``{"model", "state",
"questions"}``. Do not change ``JEV_PROMPT_VERSION`` without invalidating
cached rows (the store keys on it).
"""
from __future__ import annotations

from .contracts import EVENT_CATEGORIES, RedactedFiling

JEV_PROMPT_VERSION = "jev-v1"

#: Input truncation. Covers ~98% of filings fully (median 3.9k chars, p99 20k).
CHAR_BUDGET = 12_000

#: Model pinned for production. Never use the ``jev-latest`` alias here:
#: confidence thresholds couple to one version's probability distribution.
JEV_MODEL_PIN = "jev-1.13"

EVENT_CATEGORY_CRITERIA: dict[str, str] = {
    "earnings_results": "Quarterly or annual results: revenues, earnings per share, earnings release or earnings call.",
    "guidance_update": "Outlook or forecast: guidance raised, lowered, reaffirmed, or initiated for a future period.",
    "material_agreement": "Entry into, amendment, or termination of a material definitive agreement.",
    "financing_credit": "New credit facility, loan, note offering, or financing commitment received.",
    "capital_structure_debt": "Debt restructuring, covenant waiver or amendment, default, or debt payoff.",
    "share_issuance_buyback": "Stock issuance, public offering, buyback authorization, or share repurchase.",
    "dividend": "Dividend declared, increased, decreased, suspended, or a special dividend.",
    "mna": "Merger, acquisition, sale of the company, or a merger agreement.",
    "asset_disposal": "Sale or purchase of assets or a business unit (not the whole company).",
    "restructuring_impairment": "Layoffs, restructuring charges, asset impairments, or facility closure.",
    "management_change": "Appointment, resignation, or retirement of a CEO, CFO, officer, or director.",
    "auditor_change": "Auditor dismissed, resigned, declined to stand for re-election, or a new auditor engaged.",
    "accounting_nonreliance": "Previously issued financial statements can no longer be relied upon, or will be restated.",
    "litigation_regulatory": "Lawsuit, settlement, government investigation, or regulatory enforcement action.",
    "delisting_compliance": "Exchange listing notice, deficiency letter, or compliance plan.",
    "shareholder_vote": "Shareholder meeting results or proxy vote outcomes.",
    "product_operations": "Product launch, recall, clinical or test results, or an operations update.",
    "contract_partnership": "Customer contract win or loss, partnership, or joint venture.",
    "investor_conference": "Conference presentation, investor day, or other routine investor-relations event.",
    "other": "A material business event not described by any option above.",
    "unreadable": "No company-specific substance: empty, boilerplate-only, or incomprehensible text.",
}

assert set(EVENT_CATEGORY_CRITERIA) == set(EVENT_CATEGORIES)
assert len(EVENT_CATEGORY_CRITERIA) <= 255

MATERIALITY_LEVELS = [
    "Pure boilerplate: cover page, signatures, or checkbox disclosure with no business event.",
    "Routine event: scheduled meeting, conference attendance, or minor administrative update.",
    "Meaningful but contained event: a single contract, personnel change, or small transaction unlikely to move the stock.",
    "Significant event: earnings surprise, large financing, major agreement, or guidance change that could plausibly move the stock.",
    "Company-defining event: merger, restatement, auditor loss, going-concern warning, or delisting notice.",
]

SURPRISE_LEVELS = [
    "Fully anticipated or routine: scheduled earnings, regular dividend, annual meeting.",
    "Mostly expected: flagged in advance, ordinary-course agreement or hire.",
    "Somewhat unexpected: unscheduled transaction, surprise departure, or off-cycle update.",
    "Genuinely unexpected one-off: auditor resignation, restatement, regulator action, or sudden deal.",
]

FORWARD_LOOKING_LEVELS = [
    "Purely backward-looking: reports what already happened, no outlook.",
    "Mostly backward-looking with a passing mention of future plans.",
    "Balanced: material forward guidance alongside reported results.",
    "Dominated by guidance and outlook about future periods.",
]

CASHFLOW_MAGNITUDE_LEVELS = [
    "No stated or clearly inferrable near-term cash effect.",
    "Small cash effect: fees, minor contract, small cost item.",
    "Meaningful cash effect: material contract, offering proceeds, dividend change.",
    "Large cash effect: merger consideration, major financing, large impairment charge.",
]


def _noul(question_id: str, instructions: str, true: str, false: str) -> dict:
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {"true": true, "false": false},
    }


def build_jev_questions() -> dict[str, dict]:
    """Return the frozen ``jev-v1`` question map (12 questions, one request)."""
    return {
        "event_category": {
            "type": "choice",
            "instructions": (
                "Classify the single most material event disclosed in the filing body text. "
                "Judge from the body text only: the detected item codes are unverified hints, "
                "and cover-page boilerplate is not an event. "
                "Priority rules: if the body states prior financials can no longer be relied upon, "
                "choose accounting_nonreliance. If the body discloses an auditor dismissal, "
                "resignation, or new engagement, choose auditor_change. "
                "A routine conference or investor presentation is investor_conference. "
                "If several events appear, choose the one most likely to move the stock. "
                "If the text has no company-specific substance, choose unreadable."
            ),
            "criteria": dict(EVENT_CATEGORY_CRITERIA),
        },
        "going_concern": _noul(
            "going_concern",
            "Does the filing state going-concern doubt, a covenant default or breach, "
            "or severe liquidity stress?",
            "The body explicitly states going-concern doubt, a covenant default or breach, "
            "or that the company may be unable to meet its obligations.",
            "No such statement appears in the body.",
        ),
        "auditor_issue": _noul(
            "auditor_issue",
            "Does the filing disclose an auditor dismissal, resignation, decision not to "
            "stand for re-election, or a disagreement with the auditor?",
            "The body discloses an auditor dismissal, resignation, refusal to stand for "
            "re-election, or a disagreement.",
            "No auditor change or disagreement is disclosed.",
        ),
        "nonreliance": _noul(
            "nonreliance",
            "Does the filing state that previously issued financial statements can no "
            "longer be relied upon or will be restated?",
            "The body states non-reliance on prior financials or an upcoming restatement.",
            "No non-reliance or restatement statement appears.",
        ),
        "is_positive": _noul(
            "is_positive",
            "Does the single most material event improve expected shareholder value?",
            "The most material event is good for shareholders: higher earnings, new funding, "
            "a valuable agreement, a resolved risk.",
            "The most material event is bad for or neutral to shareholders.",
        ),
        "improves_cashflow": _noul(
            "improves_cashflow",
            "Does the filing state or clearly imply a near-term increase in cash inflows "
            "or decrease in outflows?",
            "The body states new revenue, cost cuts, offering proceeds, or financing received.",
            "No near-term cash improvement is stated or clearly implied.",
        ),
        "materiality": {
            "type": "score",
            "instructions": "How material is the most material event in this filing?",
            "criteria": list(MATERIALITY_LEVELS),
        },
        "surprise": {
            "type": "score",
            "instructions": "How surprising is the most material event relative to what was anticipated?",
            "criteria": list(SURPRISE_LEVELS),
        },
        "forward_looking": {
            "type": "score",
            "instructions": "How much of this filing is forward-looking guidance versus backward-looking reporting?",
            "criteria": list(FORWARD_LOOKING_LEVELS),
        },
        "cashflow_magnitude": {
            "type": "score",
            "instructions": "How large is the near-term cash effect stated or clearly implied by this filing, "
            "regardless of whether it is positive or negative?",
            "criteria": list(CASHFLOW_MAGNITUDE_LEVELS),
        },
        "secondary_mna": _noul(
            "secondary_mna",
            "Does the filing ALSO disclose a merger or acquisition event that is not the "
            "most material event?",
            "A merger or acquisition is disclosed in addition to a more material event.",
            "No additional merger or acquisition is disclosed.",
        ),
        "secondary_guidance": _noul(
            "secondary_guidance",
            "Does the filing ALSO disclose guidance or an outlook that is not the most "
            "material event?",
            "Guidance or an outlook is disclosed in addition to a more material event.",
            "No additional guidance or outlook is disclosed.",
        ),
    }


def build_state(filing: RedactedFiling) -> dict:
    """Build the Jev ``state`` from a redacted filing (identifiers already masked)."""
    meta = dict(filing.visible_metadata)
    return {
        "form_type": meta.get("form_type", "unknown"),
        "items_unverified": list(meta.get("items", [])),
        "is_amendment": bool(meta.get("is_amendment", False)),
        "filing_text": filing.text[:CHAR_BUDGET],
    }
