import unittest

import pandas as pd

from src.agentic.jev_aggregate import JEV_AGGREGATE_COLUMNS, aggregate_jev_rows


def _row(doc, permno, eom, category="earnings_results", direction=0.5, materiality=0.7,
         distress=0.1, abstain=0, conf=0.8):
    return {
        "document_id": doc,
        "permno": permno,
        "eom": eom,
        "event_category": category,
        "conf_event": conf,
        "signed_direction": direction,
        "materiality": materiality,
        "materiality_spread": 0.1,
        "surprise": 0.3,
        "forward_looking": 0.2,
        "distress_p": distress,
        "signed_cashflow": 0.2,
        "event_entropy": 0.4,
        "abstain": abstain,
    }


class JevAggregateTests(unittest.TestCase):
    def test_empty_input_returns_schema(self):
        out = aggregate_jev_rows([])
        self.assertEqual(list(out.columns), JEV_AGGREGATE_COLUMNS)
        self.assertEqual(len(out), 0)

    def test_month_aggregation_uses_medians_and_counts(self):
        rows = [
            _row("d1", 1, "2021-01-31", direction=0.5, materiality=0.7),
            _row("d2", 1, "2021-01-31", category="guidance_update", direction=0.9, materiality=0.3),
        ]
        out = aggregate_jev_rows(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "jev_n_filings"], 2)
        self.assertAlmostEqual(out.loc[0, "jev_direction_median"], 0.7)
        self.assertAlmostEqual(out.loc[0, "jev_direction_disagreement"], 0.2)
        self.assertEqual(out.loc[0, "jev_n_material"], 1)
        self.assertEqual(out.loc[0, "jev_event_earnings_results"], 1)
        self.assertEqual(out.loc[0, "jev_event_guidance_update"], 1)
        self.assertEqual(out.loc[0, "jev_role_coverage"], 1.0)

    def test_distress_and_abstention_propagate(self):
        rows = [_row("d1", 1, "2021-01-31", distress=0.9, abstain=1, conf=0.2)]
        out = aggregate_jev_rows(rows)
        self.assertEqual(out.loc[0, "jev_distress_any"], 1)
        self.assertAlmostEqual(out.loc[0, "jev_distress_p_median"], 0.9)
        self.assertAlmostEqual(out.loc[0, "jev_abstain_rate"], 1.0)
        self.assertGreater(out.loc[0, "jev_contradiction_rate"], 0.5)

    def test_missing_columns_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_jev_rows([{"document_id": "d1"}])

    def test_stocks_do_not_leak_across_permnos(self):
        rows = [_row("d1", 1, "2021-01-31"), _row("d2", 2, "2021-01-31")]
        out = aggregate_jev_rows(rows)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out["permno"]), {1, 2})


if __name__ == "__main__":
    unittest.main()
