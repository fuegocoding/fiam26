"""Prompt builders for the 8-K text scoring engine."""
from __future__ import annotations

import json

EVENT_CATEGORIES = (
    "earnings_results",
    "guidance_update",
    "material_agreement",
    "financing_credit",
    "capital_structure_debt",
    "share_issuance_buyback",
    "dividend",
    "mna",
    "asset_disposal",
    "restructuring_impairment",
    "management_change",
    "auditor_change",
    "accounting_nonreliance",
    "litigation_regulatory",
    "delisting_compliance",
    "shareholder_vote",
    "product_operations",
    "contract_partnership",
    "investor_conference",
    "other",
    "unreadable",
)

CATEGORY_ALIASES = {
    "earnings_release": "earnings_results",
    "earnings_announcement": "earnings_results",
    "earnings": "earnings_results",
    "earnings_report": "earnings_results",
    "conference_participation": "investor_conference",
    "conference_announcement": "investor_conference",
    "conference_presentation": "investor_conference",
    "investor_meeting": "investor_conference",
    "guidance": "guidance_update",
    "merger_acquisition": "mna",
    "merger": "mna",
    "acquisition": "mna",
    "m_a_announcement": "mna",
    "acquisition_announcement": "mna",
    "impairment": "restructuring_impairment",
    "restructuring": "restructuring_impairment",
    "buyback": "share_issuance_buyback",
    "share_repurchase": "share_issuance_buyback",
    "auditor_resignation": "auditor_change",
    "non_reliance": "accounting_nonreliance",
    "officer_change": "management_change",
    "executive_compensation": "management_change",
    "legal_settlement": "litigation_regulatory",
    "covenant": "capital_structure_debt",
    "credit_agreement": "financing_credit",
}

SIGNED_KEYS = ("sentiment", "cash_flow_impact", "conviction")
UNIT_KEYS = ("materiality", "surprise", "forward_looking", "distress_signal")

SYSTEM_PROMPT = """You are a buy-side financial analyst scoring a single Form 8-K current report \
filed by a U.S. public company. You will read the filing text and return a strict JSON object \
scoring the SINGLE most material event disclosed in the body.

Scoring instructions:
- Judge the event from the filing BODY TEXT. The detected item codes are unverified hints only; \
if the body contradicts them, trust the body.
- Ignore SEC cover-page boilerplate (registrant addresses, checkbox blocks, signatures).
- If the filing contains several events, score the one most likely to move the stock, and list \
up to 2 others in secondary_categories.
- If the text is empty, truncated beyond comprehension, or contains no company-specific substance, \
use event_category "unreadable" with all scores at 0 and an empty summary.
- event_category MUST be one of the enum values below, spelled exactly. If Item 4.02 content \
(financials can no longer be relied upon) is present, use "accounting_nonreliance". If an \
auditor change or resignation is present, use "auditor_change". These are the \
highest-priority classifications. Announcements of investor/conference presentations are \
routine: use "investor_conference" with materiality 0.1 or below.
- Anchor points: sentiment +1 = clearly value-positive for shareholders, 0 = neutral or no view, \
-1 = clearly value-negative. materiality 0 = pure boilerplate, 0.5 = meaningful but contained, \
0.8+ = event that plausibly moves the stock. surprise 0 = fully anticipated or routine, 1 = \
genuinely unexpected one-off. forward_looking 0 = purely backward-looking disclosure, 1 = \
dominated by guidance/outlook. distress_signal 0 = none, 1 = explicit going-concern, auditor \
resignation, restatement/non-reliance, or severe covenant impairment. cash_flow_impact: stated \
or clearly inferrable near-term cash-flow effect, signed. conviction: your single signed verdict \
(direction times confidence), 0 = no directional view.
- You see ONLY this document and its metadata. Do not assume anything not stated in it. Do not \
use any knowledge of what happened to the company after this filing date.

Return ONLY a JSON object with EXACTLY these keys:
event_category (string, enum), secondary_categories (array of 0-2 enums), sentiment (float, \
-1.0..1.0), materiality (float, 0.0..1.0), surprise (float, 0.0..1.0), forward_looking (float, \
0.0..1.0), distress_signal (float, 0.0..1.0), cash_flow_impact (float, -1.0..1.0), conviction \
(float, -1.0..1.0), summary (string, max 25 words).
No markdown, no code fences, no keys beyond these, numbers not strings, two decimals."""


def build_user_message(
    ticker: str,
    company_name: str,
    filing_date: str,
    event_date_hint: str | None,
    form_type: str,
    items: list[str],
    text: str,
    char_budget: int = 12_000,
) -> str:
    items_str = ", ".join(items) if items else "(none detected)"
    hint = event_date_hint if event_date_hint else "(not extracted)"
    body = text[:char_budget]
    payload = {
        "ticker": ticker,
        "company_name": company_name,
        "form_type": form_type,
        "filing_date": filing_date,
        "event_date_from_header": hint,
        "detected_items_unverified": items_str,
        "filing_text": body,
    }
    return json.dumps(payload, ensure_ascii=False)


def validate_response(raw: str) -> dict:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:]
        txt = txt.strip()
    obj = json.loads(txt)
    required = {
        "event_category",
        "secondary_categories",
        *SIGNED_KEYS,
        *UNIT_KEYS,
        "summary",
    }
    missing = required - set(obj)
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    if obj["event_category"] not in EVENT_CATEGORIES:
        alias = CATEGORY_ALIASES.get(str(obj["event_category"]).strip().lower())
        if alias:
            obj["event_category"] = alias
        else:
            raise ValueError(f"bad event_category: {obj['event_category']!r}")
    sec = []
    for k in obj["secondary_categories"]:
        if k not in EVENT_CATEGORIES:
            alias = CATEGORY_ALIASES.get(str(k))
            if not alias:
                raise ValueError(f"bad secondary category: {k!r}")
            sec.append(alias)
        else:
            sec.append(k)
    obj["secondary_categories"] = sec
    for k in SIGNED_KEYS:
        v = float(obj[k])
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"{k} out of range: {v}")
    for k in UNIT_KEYS:
        v = float(obj[k])
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{k} out of range: {v}")
    obj["summary"] = str(obj["summary"])[:200]
    return {k: obj[k] for k in sorted(obj)}
