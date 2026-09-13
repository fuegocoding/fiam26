import tempfile
import unittest
from pathlib import Path

from src.agentic.aggregate import aggregate_verdicts
from src.agentic.storage import CouncilStore


def row(document_id, role, direction, materiality=0.5, category="other", abstain=0):
    return {
        "document_id": document_id,
        "permno": 10,
        "eom": "2021-01-31",
        "role": role,
        "event_category": category,
        "direction": direction,
        "materiality": materiality,
        "surprise": 0.3,
        "forward_looking": 0.4,
        "distress_signal": 0.1,
        "cash_flow_impact": direction / 2,
        "confidence": 0.8,
        "contradiction_score": 0.2,
        "evidence_quality": 0.9,
        "abstain": abstain,
        "evidence_count": 2,
    }


class AggregationTests(unittest.TestCase):
    def test_aggregates_roles_without_counting_roles_as_filings(self):
        rows = [
            row("doc-a", "event_classifier", 1.0, 0.8, "guidance_update"),
            row("doc-a", "materiality_analyst", 0.5, 0.7, "guidance_update"),
            row("doc-a", "devils_advocate", -0.5, 0.6, "other", abstain=1),
            row("doc-b", "event_classifier", -1.0, 0.2, "auditor_change"),
        ]

        features = aggregate_verdicts(rows)
        result = features.iloc[0]

        self.assertEqual(len(features), 1)
        self.assertEqual(result["council_n_filings"], 2)
        self.assertAlmostEqual(result["council_role_coverage"], 0.5)
        self.assertAlmostEqual(result["council_direction_median"], -0.25)
        self.assertEqual(result["council_event_guidance_update"], 1)
        self.assertEqual(result["council_event_auditor_change"], 1)
        self.assertGreater(result["council_direction_disagreement"], 0.0)

    def test_empty_input_has_contract_columns(self):
        features = aggregate_verdicts([])

        self.assertEqual(len(features), 0)
        self.assertIn("council_direction_median", features.columns)
        self.assertIn("council_n_filings", features.columns)


class StorageTests(unittest.TestCase):
    def test_verdict_cache_is_idempotent_and_keyed_by_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CouncilStore(Path(tmp) / "council.db")
            store.save_verdict(
                document_id="doc-a",
                permno=10,
                eom="2021-01-31",
                role="event_classifier",
                model_key="model-a",
                prompt_version="prompt-1",
                redaction_version="redaction-v1",
                payload={"direction": 1.0},
            )

            cached = store.cached_roles(
                "doc-a", "model-a", "prompt-1", "redaction-v1"
            )
            other_prompt = store.cached_roles(
                "doc-a", "model-a", "prompt-2", "redaction-v1"
            )

            self.assertEqual(cached, {"event_classifier"})
            self.assertEqual(other_prompt, set())
            store.save_failure(
                "doc-a",
                "devils_advocate",
                "model-a",
                "prompt-1",
                "redaction-v1",
                "provider unavailable",
            )
            self.assertIn(
                "devils_advocate",
                store.cached_roles("doc-a", "model-a", "prompt-1", "redaction-v1"),
            )
            self.assertNotIn(
                "devils_advocate",
                store.cached_roles(
                    "doc-a", "model-a", "prompt-1", "redaction-v1", retry_failed=True
                ),
            )
            store.close()


if __name__ == "__main__":
    unittest.main()
