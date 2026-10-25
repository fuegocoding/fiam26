"""Text utilities: PDF extraction (optional pypdf), page/section lookup, heuristic claims.

Pages in extracted text are separated by form feed (``\\f``); page = count of
preceding form feeds + 1. Heuristic claim extraction is purely extractive
(exact sentences from the text, flagged ``extraction="heuristic"`` for review);
it never generates or paraphrases content.
"""

from __future__ import annotations

import io
import re
from typing import Optional

from .models import PapersError

MAX_PDF_PAGES = 400


class PdfUnavailable(PapersError):
    """pypdf missing or the file could not be decoded; text status becomes incomplete."""


def extract_pdf_text(data: bytes) -> str:
    if not data.startswith(b"%PDF-"):
        raise PdfUnavailable("not_a_pdf")
    try:
        import pypdf  # optional dependency
    except ImportError as exc:
        raise PdfUnavailable("pypdf_unavailable") from exc
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages):
            if i >= MAX_PDF_PAGES:
                break
            pages.append(page.extract_text() or "")
    except Exception as exc:  # pypdf raises many types on malformed input
        raise PdfUnavailable(f"pdf_decode_failed:{type(exc).__name__}") from exc
    text = "\f".join(pages)
    if not text.replace("\f", "").strip():
        raise PdfUnavailable("pdf_has_no_extractable_text")
    return text


def page_of(text: str, offset: int) -> int:
    return text.count("\f", 0, offset) + 1


_HEAD = re.compile(
    r"^[ \t\f]{0,3}(?:#{1,4}[ \t]+(?P<md>[^\n]{1,120})"
    r"|(?P<num>(?:\d+(?:\.\d+)*\.?|[A-Z]\.|Appendix(?:[ \t]+[A-Z0-9]+)?[.:]?)[ \t]+[A-Z][^\n]{0,100}))[ \t]*$",
    re.M)


def section_of(text: str, offset: int) -> Optional[str]:
    found = None
    for m in _HEAD.finditer(text, 0, offset):
        label = (m.group("md") or m.group("num")).strip()
        if m.group("num") and label.endswith(".") and not label.startswith("Appendix"):
            continue  # a numbered sentence, not a heading
        found = label[:120]
    return found


_CUES = (
    ("economic_claim", r"\bwe (?:find|show|document|report)\b|\bresults? (?:show|indicate|suggest)\b"),
    ("mechanism", r"\b(?:mechanism|because|explanation|consistent with)\b"),
    ("universe", r"\b(?:sample|universe)\b.{0,40}\b(?:consists?|includes?|comprises?|contains?)\b"),
    ("period", r"\b(?:19|20)\d{2}\s*(?:to|-|–|through|until)\s*(?:19|20)\d{2}\b"),
    ("horizon", r"\b(?:horizon|holding period|days? ahead|months? ahead|one-month|next[- ]month)\b"),
    ("variable_definition", r"\bis defined as\b|\bwe define\b|\bis computed as\b"),
    ("timing", r"\blagged\b|\bend of (?:the )?(?:month|quarter|year)\b|\bavailable at\b"),
    ("statistical_design", r"\bregress|\bFama[- ]MacBeth\b|\bt-stat|\bstandard errors?\b|\bportfolio sort"),
    ("baseline", r"\bbenchmark\b|\bbaseline\b|\bcompared (?:to|with)\b"),
    ("limitation", r"\blimitation|\bcaveat\b|\bdoes not hold\b"),
)
_SENT = re.compile(r"[^.!?\n][^.!?]{19,599}[.!?]")


def heuristic_spans(text: str, *, max_total: int = 40, per_kind: int = 5):
    """Yield ``(kind, start, end)`` for sentences matching cue patterns."""
    counts: dict[str, int] = {}
    out = []
    compiled = [(k, re.compile(p, re.I)) for k, p in _CUES]
    for m in _SENT.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        sent = text[s:e]
        for kind, rx in compiled:
            if counts.get(kind, 0) < per_kind and rx.search(sent):
                counts[kind] = counts.get(kind, 0) + 1
                out.append((kind, s, e))
                break
        if len(out) >= max_total:
            break
    return out
