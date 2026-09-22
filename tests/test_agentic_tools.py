import unittest

from src.agentic.contracts import RedactedFiling
from src.agentic.archive_llm_council.tools import RedactedFilingTools


class FilingToolsTests(unittest.TestCase):
    def setUp(self):
        self.tools = RedactedFilingTools(
            RedactedFiling(
                document_id="internal-doc",
                opaque_id="DOC_ABC",
                permno=1,
                filing_date="2021-01-01",
                eom="2021-01-31",
                text="Revenue increased.\nCash flow remained positive.\nRevenue guidance was raised.",
                visible_metadata={"form_type": "8-K"},
            )
        )

    def test_search_is_case_insensitive_and_bounded(self):
        result = self.tools.dispatch("search_filing", {"query": "REVENUE", "max_results": 1})

        self.assertEqual(len(result["matches"]), 1)
        self.assertIn("Revenue", result["matches"][0]["excerpt"])

    def test_excerpt_is_capped(self):
        result = self.tools.dispatch("get_filing_excerpt", {"start": 0, "end": 5000})

        self.assertLessEqual(len(result["excerpt"]), 1200)

    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            self.tools.dispatch("read_web", {})


if __name__ == "__main__":
    unittest.main()
