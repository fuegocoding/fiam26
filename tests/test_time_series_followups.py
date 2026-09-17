import pandas as pd
from src.time_series_followups import future_same_quarter


def test_future_labels_require_exact_next_month_and_same_fiscal_quarter():
    p = pd.DataFrame({
        'permno': [1]*5,
        'eom': pd.to_datetime(['2020-01-31','2020-02-29','2020-03-31','2020-05-31','2020-06-30']),
        'target_fpe': pd.to_datetime(['2020-03-31','2020-03-31','2020-06-30','2020-06-30','2020-06-30']),
        'MEANEST': [1.,2.,3.,5.,6.], 'STDEV': [.1,.2,.3,.5,.6],
        'STATPERS': pd.to_datetime(['2020-01-25','2020-02-25','2020-03-25','2020-05-25','2020-06-25'])})
    labels = future_same_quarter(p)
    assert labels.next_estimate.iloc[0] == 2
    assert labels.next_std.iloc[0] == .2
    assert pd.isna(labels.next_estimate.iloc[1])  # Fiscal rollover cannot become a revision.
    assert pd.isna(labels.next_estimate.iloc[2])  # Missing April cannot be bridged to May.
    assert labels.next_estimate.iloc[3] == 6
    assert pd.isna(labels.next_estimate.iloc[4])
    assert 'next_estimate' not in p.columns  # Labels do not mutate the feature panel.
