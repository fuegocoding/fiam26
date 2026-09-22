import unittest

from src.agentic.jev_encode import encode_filing_answers
from src.agentic.jev_questions import build_jev_questions


def _answers(**overrides):
    questions = build_jev_questions()
    n_cat = len(questions["event_category"]["criteria"])
    probs = {cat: (0.7 if cat == "earnings_results" else 0.3 / (n_cat - 1)) for cat in questions["event_category"]["criteria"]}

    def score(mean_level, n):
        p = [0.0] * n
        lo, frac = int(mean_level), mean_level - int(mean_level)
        p[lo] = 1.0 - frac
        if frac and lo + 1 < n:
            p[lo] = 1.0 - frac
            p[lo + 1] = frac
        return {"type": "score", "score": mean_level, "confidence": 0.8,
                "probabilities": {str(i): p[i] for i in range(n)}}

    base = {
        "event_category": {"type": "choice", "choice": "earnings_results",
                           "confidence": 0.6, "probabilities": probs},
        "going_concern": {"type": "noul", "noul": 0.05},
        "auditor_issue": {"type": "noul", "noul": 0.02},
        "nonreliance": {"type": "noul", "noul": 0.01},
        "is_positive": {"type": "noul", "noul": 0.9},
        "improves_cashflow": {"type": "noul", "noul": 0.8},
        "materiality": score(3.0, 5),
        "surprise": score(1.0, 4),
        "forward_looking": score(2.0, 4),
        "cashflow_magnitude": score(2.0, 4),
        "secondary_mna": {"type": "noul", "noul": 0.1},
        "secondary_guidance": {"type": "noul", "noul": 0.7},
    }
    base.update(overrides)
    return questions, base


class JevEncodeTests(unittest.TestCase):
    def test_signed_direction_combines_sign_and_magnitude(self):
        questions, answers = _answers()
        row = encode_filing_answers(answers, questions)
        self.assertEqual(row["event_category"], "earnings_results")
        # (2*0.9-1) * (3/4) = 0.8 * 0.75
        self.assertAlmostEqual(row["signed_direction"], 0.6)
        # (2*0.8-1) * (2/3)
        self.assertAlmostEqual(row["signed_cashflow"], 0.4)
        self.assertEqual(row["distress_any"], 0)
        self.assertEqual(row["abstain"], 0)
        self.assertEqual(row["is_material"], 1)

    def test_distress_is_max_over_atomic_nouls(self):
        questions, answers = _answers(nonreliance={"type": "noul", "noul": 0.9})
        row = encode_filing_answers(answers, questions)
        self.assertAlmostEqual(row["distress_p"], 0.9)
        self.assertEqual(row["distress_any"], 1)

    def test_low_confidence_abstains(self):
        questions, answers = _answers()
        answers["event_category"] = dict(answers["event_category"], confidence=0.2)
        row = encode_filing_answers(answers, questions)
        self.assertEqual(row["abstain"], 1)

    def test_negative_event_gives_negative_direction(self):
        questions, answers = _answers(is_positive={"type": "noul", "noul": 0.1})
        row = encode_filing_answers(answers, questions)
        self.assertLess(row["signed_direction"], 0)


if __name__ == "__main__":
    unittest.main()
