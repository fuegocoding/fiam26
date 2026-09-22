"""Minimal transport for Jev System One decisions via OpenRouter.

Endpoint: ``POST {base_url}`` where the default base URL is the full
``https://openrouter.ai/api/v1/systemone`` path (TypeSafe-compatible request
and response shapes). Auth is the OpenRouter Bearer key — the same
``LLM_API_KEY`` value already in ``.env``; ``JEV_API_KEY`` overrides it.

Stdlib only (urllib): no new dependencies for the scoring path.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class JevError(RuntimeError):
    """Raised for transport failures and malformed Jev responses."""


@dataclass(frozen=True)
class JevConfig:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1/systemone"
    model: str = "jev-1.13"
    timeout_s: int = 120
    max_retries: int = 4


@dataclass(frozen=True)
class JevResponse:
    answers: dict
    model: str
    usage: dict = field(default_factory=dict)
    cost: float | None = None
    request_id: str | None = None


RETRY_BACKOFF_S = [2, 5, 12, 30]


def _validate_answers(questions: dict, answers: dict) -> dict:
    if not isinstance(answers, dict):
        raise JevError("response 'answers' is not an object")
    for qid, question in questions.items():
        if qid not in answers:
            raise JevError(f"missing answer for question {qid!r}")
        answer = answers[qid]
        qtype = question.get("type")
        if not isinstance(answer, dict) or answer.get("type") != qtype:
            raise JevError(f"answer {qid!r} has wrong type (expected {qtype!r})")
        if qtype == "noul":
            p = answer.get("noul")
            if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0.0 <= p <= 1.0:
                raise JevError(f"answer {qid!r} has invalid noul probability: {p!r}")
        elif qtype == "choice":
            criteria = question.get("criteria", {})
            if answer.get("choice") not in criteria:
                raise JevError(f"answer {qid!r} choice not in criteria: {answer.get('choice')!r}")
            probs = answer.get("probabilities", {})
            total = 0.0
            for opt in criteria:
                p = probs.get(opt)
                if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0.0 <= p <= 1.0:
                    raise JevError(f"answer {qid!r} has invalid probability for {opt!r}")
                total += p
            if not 0.95 <= total <= 1.05:
                raise JevError(f"answer {qid!r} probabilities sum to {total:.3f}")
            conf = answer.get("confidence")
            if conf is not None and (
                not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0.0 <= conf <= 1.0
            ):
                raise JevError(f"answer {qid!r} has invalid confidence: {conf!r}")
        elif qtype == "score":
            levels = question.get("criteria", [])
            probs = answer.get("probabilities", {})
            if len(probs) != len(levels):
                raise JevError(f"answer {qid!r} probability count != level count")
            for i in range(len(levels)):
                p = probs.get(str(i), probs.get(i))
                if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0.0 <= p <= 1.0:
                    raise JevError(f"answer {qid!r} has invalid probability for level {i}")
            score = answer.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise JevError(f"answer {qid!r} has invalid score: {score!r}")
            if not -0.5 <= score <= len(levels) - 0.5:
                raise JevError(f"answer {qid!r} score out of range: {score!r}")
        else:
            raise JevError(f"unknown question type {qtype!r} for {qid!r}")
    return answers


class JevClient:
    """Thin client for one-shot System One evaluations."""

    def __init__(self, config: JevConfig):
        if not config.api_key:
            raise JevError("missing Jev/OpenRouter API key")
        self.config = config

    def system_one(self, state: dict | str | list, questions: dict) -> JevResponse:
        payload = json.dumps(
            {"model": self.config.model, "state": state, "questions": questions}
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(
                self.config.base_url,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                    request_id = resp.headers.get("x-request-id")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code in (408, 425, 429) or 500 <= exc.code <= 599:
                    last_error = JevError(f"HTTP {exc.code}: {detail}")
                    retry_after = exc.headers.get("retry-after") if exc.headers else None
                    delay = _backoff_delay(attempt, retry_after)
                    time.sleep(delay)
                    continue
                raise JevError(f"HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = JevError(f"transport error: {exc}")
                time.sleep(_backoff_delay(attempt, None))
                continue
            try:
                envelope = json.loads(body)
            except json.JSONDecodeError as exc:
                raise JevError(f"response is not JSON: {body[:200]!r}") from exc
            answers = _validate_answers(questions, envelope.get("answers"))
            usage = envelope.get("usage", {}) or {}
            cost = usage.get("cost")
            return JevResponse(
                answers=answers,
                model=str(envelope.get("model", self.config.model)),
                usage=usage if isinstance(usage, dict) else {},
                cost=float(cost) if isinstance(cost, (int, float)) else None,
                request_id=request_id,
            )
        raise last_error or JevError("Jev request failed without a captured error")


def _backoff_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 60.0)
        except ValueError:
            pass
    return RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
