import unittest

from src.agentic.contracts import EVENT_CATEGORIES
from src.agentic.jev_questions import (
    CHAR_BUDGET,
    JEV_MODEL_PIN,
    JEV_PROMPT_VERSION,
    build_jev_questions,
    build_state,
)
from src.agentic.contracts import RedactedFiling


def _filing(text="Body text about earnings."):
    return RedactedFiling(
        document_id="doc-1",
        opaque_id="DOC_ABC",
        permno=1,
        filing_date="2021-01-15",
        eom="2021-01-31",
        text=text,
        visible_metadata={"form_type": "8-K", "items": ["2.02"], "is_amendment": False},
    )


class JevQuestionsTests(unittest.TestCase):
    def setUp(self):
        self.questions = build_jev_questions()

    def test_prompt_version_is_pinned(self):
        self.assertEqual(JEV_PROMPT_VERSION, "jev-v1")
        self.assertEqual(JEV_MODEL_PIN, "jev-1.13")

    def test_event_category_covers_closed_enum_with_escape_hatches(self):
        criteria = self.questions["event_category"]["criteria"]
        self.assertEqual(set(criteria), set(EVENT_CATEGORIES))
        self.assertIn("other", criteria)
        self.assertIn("unreadable", criteria)
        self.assertLessEqual(len(criteria), 255)

    def test_score_questions_use_ordered_levels_within_limits(self):
        for qid, question in self.questions.items():
            if question["type"] != "score":
                continue
            levels = question["criteria"]
            self.assertGreaterEqual(len(levels), 2, qid)
            self.assertLessEqual(len(levels), 10, qid)
            for level in levels:
                self.assertIsInstance(level, str)
                self.assertTrue(level.strip(), qid)

    def test_noul_questions_carry_true_false_criteria(self):
        nouls = [q for q in self.questions.values() if q["type"] == "noul"]
        self.assertGreaterEqual(len(nouls), 5)
        for question in nouls:
            self.assertIn("true", question["criteria"])
            self.assertIn("false", question["criteria"])

    def test_every_question_has_literal_instructions(self):
        for qid, question in self.questions.items():
            self.assertTrue(question["instructions"].strip(), qid)
            self.assertIn(question["type"], ("choice", "score", "noul"), qid)

    def test_state_excludes_identity_and_truncates(self):
        state = build_state(_filing(text="x" * (CHAR_BUDGET + 5000)))
        self.assertNotIn("ticker", str(state))
        self.assertNotIn("document_id", str(state))
        self.assertLessEqual(len(state["filing_text"]), CHAR_BUDGET)
        self.assertEqual(state["items_unverified"], ["2.02"])


if __name__ == "__main__":
    unittest.main()
