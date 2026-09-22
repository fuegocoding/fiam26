import json
import unittest
from types import SimpleNamespace

from src.agentic.contracts import ROLE_NAMES, RedactedFiling
from src.agentic.archive_llm_council.council import CouncilRunner
from src.agentic.archive_llm_council.roles import build_role_messages
from src.agentic.archive_llm_council.run_council import PROMPT_VERSION
from src.agentic.redaction import REDACTION_VERSION


def verdict_payload(category="other"):
    return {
        "event_category": category,
        "direction": 0.2,
        "materiality": 0.4,
        "surprise": 0.3,
        "forward_looking": 0.5,
        "distress_signal": 0.0,
        "cash_flow_impact": 0.1,
        "confidence": 0.6,
        "contradiction_score": 0.2,
        "evidence_quality": 0.8,
        "abstain": 0,
        "evidence_count": 1,
        "summary": "The redacted filing contains one supported event.",
    }


class FakeLLM:
    def __init__(self, failing_role=None):
        self.failing_role = failing_role
        self.calls = []

    def complete_json(self, system, user, request_id):
        self.calls.append((system, user, request_id))
        role = next(name for name in ROLE_NAMES if name in system)
        if role == self.failing_role:
            raise RuntimeError("temporary provider failure")
        return SimpleNamespace(content=json.dumps(verdict_payload()))


class CouncilTests(unittest.TestCase):
    def setUp(self):
        self.filing = RedactedFiling(
            document_id="doc-1",
            opaque_id="DOC_ABC",
            permno=1,
            filing_date="2021-01-01",
            eom="2021-01-31",
            text="<IDENTIFIER> reported revenue of $100 million.",
            visible_metadata={"form_type": "8-K", "items": ["2.02"]},
        )

    def test_all_roles_get_the_same_redacted_user_payload(self):
        messages = [build_role_messages(role, self.filing) for role in ROLE_NAMES]
        users = [user["content"] for _, user in messages]
        payload = json.loads(users[0])

        self.assertEqual(len(set(users)), 1)
        self.assertNotIn("document_id", payload)
        self.assertNotIn("opaque_id", payload)
        self.assertNotIn("filing_date", payload)
        devil_system = messages[2][0]["content"]
        self.assertIn("disconfirm", devil_system.lower())
        self.assertIn("abstain", devil_system.lower())
        self.assertIn("direction and cash_flow_impact are in [-1.0, 1.0]", devil_system)

    def test_failed_role_does_not_discard_valid_verdicts(self):
        llm = FakeLLM(failing_role="devils_advocate")
        result = CouncilRunner(llm).score_filing(self.filing)

        self.assertEqual(len(result.verdicts), 3)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].role, "devils_advocate")

    def test_invalid_json_gets_one_repair_attempt(self):
        class RepairLLM(FakeLLM):
            def __init__(self):
                super().__init__()
                self.n = 0

            def complete_json(self, system, user, request_id):
                self.n += 1
                if self.n == 1:
                    return SimpleNamespace(content="not json")
                return SimpleNamespace(content=json.dumps(verdict_payload()))

        result = CouncilRunner(RepairLLM(), roles=(ROLE_NAMES[0],)).score_filing(self.filing)

        self.assertEqual(len(result.verdicts), 1)
        self.assertEqual(len(result.failures), 0)

    def test_cache_versions_are_bumped_after_firewall_and_prompt_changes(self):
        self.assertEqual(REDACTION_VERSION, "redaction-v2")
        self.assertEqual(PROMPT_VERSION, "council-v3")


if __name__ == "__main__":
    unittest.main()
