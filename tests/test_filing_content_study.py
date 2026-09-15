import unittest
import numpy as np
from src.filing_content_study import RX
from src.filing_momentum_study import stats


class FilingContentChecks(unittest.TestCase):
    def test_actual_and_hypothetical_distress(self):
        self.assertIsNotNone(RX['distress'].search('The Company has failed to pay interest on its notes.'))
        self.assertIsNone(RX['distress'].search('Customary events of default include failure to pay interest or principal.'))

    def test_actual_dividend_cut(self):
        self.assertIsNotNone(RX['dividend_cut'].search('The board suspended its quarterly cash dividend.'))
        self.assertIsNone(RX['dividend_cut'].search('The executive will receive dividend equivalents.'))
        self.assertIsNone(RX['dividend_cut'].search('We may reduce debt, pay dividends, and repurchase stock.'))
        self.assertIsNone(RX['dividend_cut'].search('We may suspend the dividend at our discretion.'))

    def test_explicit_guidance(self):
        self.assertIsNotNone(RX['guidance_up'].search('The Company raised its earnings guidance.'))
        self.assertIsNotNone(RX['guidance_down'].search('The Company reduced its annual guidance.'))

    def test_non_reliance(self):
        self.assertIsNotNone(RX['non_reliance'].search('The financial statements should no longer be relied upon.'))

    def test_hac_preserves_missing_calendar_months(self):
        a=np.array([1.,2.,3.,4.,2.,1.,3.,2.])
        gapped=np.full(16,np.nan);gapped[::2]=a
        self.assertEqual(stats(gapped)[0],8)
        self.assertAlmostEqual(stats(gapped)[1],stats(a)[1])
        self.assertNotEqual(stats(gapped)[2],stats(a)[2])


if __name__=='__main__':unittest.main()
