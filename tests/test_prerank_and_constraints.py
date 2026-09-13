import unittest

import pandas as pd

from src.agentic.prerank import select_uncertain_document_ids, with_uncertainty_score
from src.portfolio_constraints import apply_investability_screen


class PrerankTests(unittest.TestCase):
    def setUp(self):
        self.scores = pd.DataFrame(
            {
                "document_id": ["certain", "ambiguous", "disagreeing"],
                "prob_positive": [0.98, 1 / 3, 0.49],
                "prob_negative": [0.01, 1 / 3, 0.49],
                "prob_neutral": [0.01, 1 / 3, 0.02],
                "sentiment_min_chunk": [0.0, 0.0, -0.9],
                "sentiment_max_chunk": [0.0, 0.0, 0.9],
            }
        )

    def test_entropy_and_chunk_disagreement_rank_uncertain_filings(self):
        scored = with_uncertainty_score(self.scores)
        self.assertGreater(
            scored.loc[scored.document_id == "ambiguous", "prerank_uncertainty"].iloc[0],
            scored.loc[scored.document_id == "certain", "prerank_uncertainty"].iloc[0],
        )
        self.assertEqual(select_uncertain_document_ids(scored, 2), ["ambiguous", "disagreeing"])

    def test_duplicate_document_ids_are_rejected(self):
        duplicate = pd.concat([self.scores, self.scores.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            with_uncertainty_score(duplicate)


class PortfolioConstraintTests(unittest.TestCase):
    def test_screen_uses_monthly_fields_and_excludes_small_groups(self):
        frame = pd.DataFrame(
            {
                "eom": ["2021-01-31"] * 4,
                "permno": [1, 2, 3, 4],
                "me": [10.0, 20.0, 30.0, 40.0],
                "dolvol": [100.0, 200.0, 300.0, 400.0],
                "size_grp": ["nano", "small", "large", "mega"],
            }
        )
        screened = apply_investability_screen(
            frame,
            min_me_percentile=0.5,
            min_dolvol_percentile=0.5,
            excluded_size_groups=("nano", "small"),
        )
        self.assertEqual(screened.permno.tolist(), [3, 4])

    def test_screen_is_opt_in(self):
        frame = pd.DataFrame({"eom": ["2021-01-31"], "permno": [1]})
        result = apply_investability_screen(frame)
        self.assertEqual(result.permno.tolist(), [1])
