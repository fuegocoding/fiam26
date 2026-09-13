import json
import unittest

from src.agentic.contracts import ROLE_NAMES, validate_verdict
from src.agentic.redaction import redact_filing, stable_opaque_id


class RedactionTests(unittest.TestCase):
    def setUp(self):
        self.filing = {
            "document_id": "doc-123",
            "permno": 10001,
            "filing_date": "2021-03-17",
            "ticker": "ACME",
            "company_name": "Acme Holdings Inc.",
            "cik": "0000123456",
            "cusip": "123456789",
            "isin": "US1234567890",
            "figi": "BBG000ABC123",
            "gvkey": "001234",
            "iid": "01",
            "sector": "Technology",
            "industry": "Software",
            "report_url": "https://example.com/filing/123",
            "form_type": "8-K",
            "items": ["2.02"],
            "is_amendment": False,
            "text": (
                "Acme Holdings Inc. (NASDAQ: ACME) reported revenue of $125.4 million "
                "on March 17, 2021. The company entered a $50 million facility."
            ),
        }

    def test_redaction_removes_identity_and_date_cues_but_keeps_economics(self):
        result = redact_filing(self.filing, secret="test-secret")

        self.assertNotIn("Acme Holdings", result.text)
        self.assertNotIn("ACME", result.text)
        self.assertNotIn("123456789", result.text)
        self.assertNotIn("March 17, 2021", result.text)
        self.assertNotIn("example.com", result.text)
        self.assertIn("$125.4 million", result.text)
        self.assertIn("$50 million", result.text)
        self.assertEqual(result.visible_metadata["form_type"], "8-K")
        self.assertEqual(result.visible_metadata["items"], ["2.02"])
        self.assertNotIn("company_name", result.visible_metadata)
        self.assertNotIn("sector", result.visible_metadata)

    def test_opaque_id_is_stable_and_secret_dependent(self):
        first = stable_opaque_id("doc-123", "test-secret")
        second = stable_opaque_id("doc-123", "test-secret")
        other = stable_opaque_id("doc-123", "other-secret")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertTrue(first.startswith("DOC_"))

    def test_empty_secret_is_rejected(self):
        with self.assertRaises(ValueError):
            stable_opaque_id("doc-123", "")


class ContractTests(unittest.TestCase):
    def test_valid_verdict_is_parsed(self):
        payload = {
            "event_category": "guidance_update",
            "direction": 0.5,
            "materiality": 0.8,
            "surprise": 0.4,
            "forward_looking": 1.0,
            "distress_signal": 0.0,
            "cash_flow_impact": 0.6,
            "confidence": 0.7,
            "contradiction_score": 0.1,
            "evidence_quality": 0.9,
            "abstain": 0,
            "evidence_count": 2,
            "summary": "The filing raises forward guidance based on the disclosed operating outlook.",
        }

        verdict = validate_verdict(json.dumps(payload), role=ROLE_NAMES[0])

        self.assertEqual(verdict.role, ROLE_NAMES[0])
        self.assertEqual(verdict.event_category, "guidance_update")
        self.assertEqual(verdict.evidence_count, 2)

    def test_invalid_range_and_unknown_keys_are_rejected(self):
        payload = {
            "event_category": "guidance_update",
            "direction": 2,
            "materiality": 0.8,
            "surprise": 0.4,
            "forward_looking": 1.0,
            "distress_signal": 0.0,
            "cash_flow_impact": 0.6,
            "confidence": 0.7,
            "contradiction_score": 0.1,
            "evidence_quality": 0.9,
            "abstain": 0,
            "evidence_count": 2,
            "summary": "bad",
            "trade": "long",
        }

        with self.assertRaises(ValueError):
            validate_verdict(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
