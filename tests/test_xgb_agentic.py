import unittest

import pandas as pd

from src.agentic.features import merge_council_features
from src.agentic.filings import filing_month_eom


class FeatureJoinTests(unittest.TestCase):
    def test_filing_month_cutoff_is_the_month_end_before_next_month_target(self):
        self.assertEqual(filing_month_eom("2021-01-17"), "2021-01-31")
        self.assertEqual(filing_month_eom("2021-02-01"), "2021-02-28")

    def test_left_join_preserves_panel_and_excludes_target_columns(self):
        table = pd.DataFrame(
            {
                "permno": [1, 2],
                "eom": ["2021-01-31", "2021-01-31"],
                "ret_exc_lead1m": [0.1, -0.2],
                "factor": [0.3, 0.4],
            }
        )
        council = pd.DataFrame(
            {
                "permno": [1],
                "eom": ["2021-01-31"],
                "council_direction_median": [0.5],
                "ret_exc_lead1m": [999.0],
            }
        )

        merged, features = merge_council_features(table, council)

        self.assertEqual(len(merged), len(table))
        self.assertEqual(features, ["council_direction_median"])
        self.assertEqual(merged.loc[0, "council_direction_median"], 0.5)
        self.assertTrue(pd.isna(merged.loc[1, "council_direction_median"]))
        self.assertEqual(list(merged.columns).count("ret_exc_lead1m"), 1)

    def test_january_filing_features_join_to_february_target_row(self):
        table = pd.DataFrame(
            {
                "permno": [1],
                "eom": ["2021-01-31"],
                "target_month": ["2021-02-28"],
                "ret_exc_lead1m": [0.1],
            }
        )
        council = pd.DataFrame(
            {
                "permno": [1],
                "eom": ["2021-01-31"],
                "council_direction_median": [0.7],
            }
        )

        merged, _ = merge_council_features(table, council)

        self.assertEqual(merged.loc[0, "target_month"], "2021-02-28")
        self.assertEqual(merged.loc[0, "council_direction_median"], 0.7)

    def test_duplicate_council_keys_are_rejected(self):
        table = pd.DataFrame({"permno": [1], "eom": ["2021-01-31"]})
        council = pd.DataFrame(
            {
                "permno": [1, 1],
                "eom": ["2021-01-31", "2021-01-31"],
                "council_direction_median": [0.5, 0.6],
            }
        )

        with self.assertRaises(ValueError):
            merge_council_features(table, council)


if __name__ == "__main__":
    unittest.main()
