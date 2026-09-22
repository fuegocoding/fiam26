"""Encode one filing's Jev answers into a numeric feature row.

All combination logic lives here, in code — never in the model:
- Score expectations/spreads from the returned distributions.
- Signed direction/cash-flow as products of Noul probabilities and magnitudes.
- Distress as a max over atomic Nouls.
- Abstention from the Choice confidence.

Threshold defaults below are starting points. Final values must be tuned on
validation folds only (never on the OOS window) and recorded in the strategy
memo; the store keeps full probability distributions so re-thresholding never
requires re-calling the API.
"""
from __future__ import annotations

import math

# Starting thresholds (tune on validation; record frozen values in the memo).
TAU_ABSTAIN_CONF = 0.5
TAU_MATERIAL = 0.5
TAU_DISTRESS = 0.5


def _score_stats(answer: dict, n_levels: int) -> tuple[float, float]:
    """Return (mean, sd) of a Score distribution, normalized to [0, 1]."""
    probs = answer.get("probabilities", {})
    mean = sum(i * float(probs.get(str(i), probs.get(i, 0.0))) for i in range(n_levels))
    var = sum(
        (i - mean) ** 2 * float(probs.get(str(i), probs.get(i, 0.0)))
        for i in range(n_levels)
    )
    span = max(n_levels - 1, 1)
    return mean / span, math.sqrt(max(var, 0.0)) / span


def _entropy(probs: dict) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in probs.values() if p > 0) / math.log(
        max(len(probs), 2)
    )


def encode_filing_answers(answers: dict, questions: dict) -> dict:
    """Flatten validated Jev answers into a numeric filing-level row."""
    choice = answers["event_category"]
    choice_probs = {k: float(v) for k, v in choice["probabilities"].items()}
    ranked = sorted(choice_probs.items(), key=lambda kv: -kv[1])
    top_prob = ranked[0][1]
    margin = top_prob - (ranked[1][1] if len(ranked) > 1 else 0.0)

    def noul(qid: str) -> float:
        return float(answers[qid]["noul"])

    def score(qid: str) -> tuple[float, float, float]:
        answer = answers[qid]
        mean, spread = _score_stats(answer, len(questions[qid]["criteria"]))
        return mean, spread, float(answer.get("confidence", 1.0 - spread))

    materiality, materiality_spread, materiality_conf = score("materiality")
    surprise, surprise_spread, _ = score("surprise")
    forward_looking, forward_spread, _ = score("forward_looking")
    cashflow_mag, cashflow_spread, _ = score("cashflow_magnitude")

    p_positive = noul("is_positive")
    p_improves = noul("improves_cashflow")
    p_going = noul("going_concern")
    p_auditor = noul("auditor_issue")
    p_nonreliance = noul("nonreliance")
    distress_p = max(p_going, p_auditor, p_nonreliance)

    conf = float(choice.get("confidence", top_prob))
    return {
        "event_category": str(choice["choice"]),
        "conf_event": conf,
        "prob_event_top": top_prob,
        "event_margin": margin,
        "event_entropy": _entropy(choice_probs),
        "p_going_concern": p_going,
        "p_auditor_issue": p_auditor,
        "p_nonreliance": p_nonreliance,
        "distress_p": distress_p,
        "distress_any": int(distress_p >= TAU_DISTRESS),
        "p_positive": p_positive,
        "p_improves_cashflow": p_improves,
        "materiality": materiality,
        "materiality_spread": materiality_spread,
        "materiality_conf": materiality_conf,
        "surprise": surprise,
        "surprise_spread": surprise_spread,
        "forward_looking": forward_looking,
        "forward_spread": forward_spread,
        "cashflow_magnitude": cashflow_mag,
        "cashflow_spread": cashflow_spread,
        # Signed combinations (code-owned): direction × conviction, cash sign × size.
        "signed_direction": (2.0 * p_positive - 1.0) * materiality,
        "signed_cashflow": (2.0 * p_improves - 1.0) * cashflow_mag,
        "secondary_mna_p": noul("secondary_mna"),
        "secondary_guidance_p": noul("secondary_guidance"),
        "abstain": int(conf < TAU_ABSTAIN_CONF),
        "is_material": int(materiality >= TAU_MATERIAL),
    }
