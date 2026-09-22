import tempfile
import unittest
from pathlib import Path

from src.agentic.jev_storage import JevStore


class JevStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = JevStore(Path(self.tmp.name) / "jev_scores.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_answer_roundtrip_is_idempotent(self):
        self.store.save_identity("d1", "DOC_X", 1, "2021-01-15", "2021-01-31")
        self.assertFalse(self.store.is_cached("d1", "m", "jev-v1", "redaction-v2"))
        self.store.save_answer(
            "d1", 1, "2021-01-31", "m", "jev-v1", "redaction-v2",
            answers={"event_category": {"type": "choice"}},
            encoded={"event_category": "other", "signed_direction": 0.1},
            usage={"input_tokens": 10},
        )
        self.assertTrue(self.store.is_cached("d1", "m", "jev-v1", "redaction-v2"))
        rows = list(self.store.iter_answers("m", "jev-v1", "redaction-v2"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_category"], "other")

    def test_prompt_version_change_invalidates_cache(self):
        self.store.save_answer(
            "d1", 1, "2021-01-31", "m", "jev-v1", "redaction-v2",
            answers={}, encoded={"event_category": "other"}, usage={},
        )
        self.assertFalse(self.store.is_cached("d1", "m", "jev-v2", "redaction-v2"))

    def test_failure_marks_cached(self):
        self.store.save_failure("d9", "m", "jev-v1", "redaction-v2", "boom")
        self.assertTrue(self.store.is_cached("d9", "m", "jev-v1", "redaction-v2"))


if __name__ == "__main__":
    unittest.main()
