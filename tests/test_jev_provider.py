import unittest

from src.agentic.jev_provider import JevError, _validate_answers


def _questions():
    return {
        "cat": {"type": "choice", "instructions": "i",
                "criteria": {"a": "first", "b": "second"}},
        "urgent": {"type": "noul", "instructions": "i"},
        "sev": {"type": "score", "instructions": "i", "criteria": ["low", "high"]},
    }


def _answers():
    return {
        "cat": {"type": "choice", "choice": "a", "confidence": 0.6,
                "probabilities": {"a": 0.8, "b": 0.2}},
        "urgent": {"type": "noul", "noul": 0.9},
        "sev": {"type": "score", "score": 1.0, "confidence": 1.0,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.0, "1": 1.0}},
    }


class JevValidationTests(unittest.TestCase):
    def test_valid_response_passes(self):
        out = _validate_answers(_questions(), _answers())
        self.assertEqual(out["cat"]["choice"], "a")

    def test_unknown_choice_rejected(self):
        answers = _answers()
        answers["cat"] = dict(answers["cat"], choice="zzz")
        with self.assertRaises(JevError):
            _validate_answers(_questions(), answers)

    def test_unnormalized_probabilities_rejected(self):
        answers = _answers()
        answers["cat"] = dict(answers["cat"], probabilities={"a": 0.5, "b": 0.1})
        with self.assertRaises(JevError):
            _validate_answers(_questions(), answers)

    def test_out_of_range_noul_rejected(self):
        answers = _answers()
        answers["urgent"] = {"type": "noul", "noul": 1.5}
        with self.assertRaises(JevError):
            _validate_answers(_questions(), answers)

    def test_wrong_answer_type_rejected(self):
        answers = _answers()
        answers["sev"] = {"type": "noul", "noul": 0.5}
        with self.assertRaises(JevError):
            _validate_answers(_questions(), answers)

    def test_missing_answer_rejected(self):
        answers = _answers()
        del answers["urgent"]
        with self.assertRaises(JevError):
            _validate_answers(_questions(), answers)


if __name__ == "__main__":
    unittest.main()
