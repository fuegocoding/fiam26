"""Counts official 8-K events available at each formation cutoff."""

import pathlib

import pandas as pd

import config
import official

# These item families measure disclosed events, not unavailable SEC form counts.
FAMILIES = {
    'results': 'Item 2.02',
    'agreement': 'Item 1.01',
    'termination': 'Item 1.02',
    'bankruptcy': 'Item 1.03',
    'acquisition': 'Item 2.01',
    'debt': 'Item 2.03',
    'exit_costs': 'Item 2.05',
    'impairment': 'Item 2.06',
    'delisting': 'Item 3.01',
    'equity_sale': 'Item 3.02',
    'auditor': 'Item 4.01',
    'nonreliance': 'Item 4.02',
    'control': 'Item 5.01',
    'management': 'Item 5.02',
    'reg_fd': 'Item 7.01',
    'other': 'Item 8.01',
}
WINDOWS = {'1m': 1, '3m': 3, '12m': 12}
FEATURE_COLUMNS = [f'{family}_{window}' for family in [*FAMILIES, 'filing', 'amend']
                   for window in WINDOWS]


def build(panel_path: pathlib.Path, filings_path: pathlib.Path,
          work_dir: pathlib.Path) -> pathlib.Path:
    """Retains rolling event counts; unsupported form and IPO features are dropped."""
    output = work_dir / 'features.parquet'
    if output.exists():
        return output
    work_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(filings_path, columns=['permno', 'filing_date', 'items', 'form_type'])
    events['filing_date'] = pd.to_datetime(events.filing_date)
    events['available_date'] = official.next_trading_day(events.filing_date)
    for name, item in FAMILIES.items():
        events[name] = events['items'].map(lambda items: int(official.has_item(items, item)))
    events['filing'] = 1
    events['amend'] = events.form_type.eq('8-K/A').astype(int)
    names = [*FAMILIES, 'filing', 'amend']
    parts = []
    for month in pd.period_range('2015-02', config.LAST_TARGET.to_period('M'), freq='M'):
        cutoff = (month - 1).to_timestamp('M')
        known = events.loc[events.filing_date.le(cutoff) & events.available_date.le(cutoff)]
        windows = []
        for label, length in WINDOWS.items():
            recent = known.loc[known.filing_date.ge((month - length).start_time)]
            counts = recent.groupby('permno')[names].sum().add_suffix('_' + label)
            windows.append(counts)
        part = pd.concat(windows, axis=1).fillna(0).reset_index()
        part['return_month'] = month.to_timestamp('M')
        parts.append(part)
    pd.concat(parts, ignore_index=True)[['permno', 'return_month', *FEATURE_COLUMNS]].to_parquet(
        output, index=False)
    return output
