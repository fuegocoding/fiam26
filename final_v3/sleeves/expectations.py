"""Official filing events and panel-based expectation proxies."""

import pandas as pd

import config
import official


def month_end(values):
    return pd.to_datetime(values) + pd.offsets.MonthEnd(0)


def _panel(columns):
    frame = pd.read_parquet(config.DATA_DIR / config.PANEL_FILE,
                            columns=list(dict.fromkeys(['permno', 'eom'] + columns)))
    frame['eom'] = pd.to_datetime(frame.eom).astype('datetime64[ns]')
    # Compatibility key: a permanent security ID, never a historical ticker guess.
    frame['TICKER'] = frame.permno.astype(str)
    return frame


def load_link():
    frame = _panel([]).groupby(['permno', 'TICKER']).eom.agg(['min', 'max']).reset_index()
    frame = frame.rename(columns={'min': 'sdate', 'max': 'edate'})
    frame['sdate'] -= pd.offsets.MonthBegin(1)
    frame['SCORE'] = 0
    return frame


def attach_ticker(grid, link):
    result = grid[['permno', 'eom']].copy()
    result['TICKER'] = result.permno.astype(str)
    return result


def load_prices():
    frame = _panel(['prc']).rename(columns={'eom': 'STATPERS', 'prc': 'PRICE'})
    frame['PRICE'] = frame.PRICE.abs()
    return frame[['TICKER', 'STATPERS', 'PRICE']]


def load_consensus():
    """Compatibility snapshots: annual earnings yield expressed per share.

    No analyst count, dispersion, fiscal target or future EPS is invented.
    """
    frame = _panel(['ni_me', 'prc']).rename(columns={'eom': 'STATPERS'})
    frame['MEANEST'] = frame.ni_me * frame.prc.abs()
    frame['snap_month'] = frame.STATPERS.dt.to_period('M')
    return frame[['TICKER', 'STATPERS', 'MEANEST', 'snap_month']]


def load_surprises():
    """Item 2.02 filings with surprise first observable at/after the filing."""
    filings = pd.read_parquet(config.DATA_DIR / config.FILINGS_FILE,
                               columns=['permno', 'filing_date', 'items', 'form_type'])
    filings = filings.loc[filings['items'].map(lambda x: official.has_item(x, '2.02'))].copy()
    filings['anndats'] = pd.to_datetime(filings.filing_date).astype('datetime64[ns]')
    filings = filings.drop_duplicates(['permno', 'anndats'])
    filings['usable_date'] = official.next_trading_day(filings.anndats)
    panel = _panel(['niq_su', 'saleq_su'])
    events = pd.merge_asof(filings.sort_values('anndats'), panel.sort_values('eom'),
                           by='permno', left_on='anndats', right_on='eom', direction='forward')
    events['available_date'] = events[['eom', 'usable_date']].max(axis=1)
    events.loc[events.usable_date.isna() | events.eom.isna(), 'available_date'] = pd.NaT
    events['fpe'] = events.eom  # observation month, not a claimed fiscal quarter
    events['suescore'] = events.niq_su
    # Multiple filings pointing at one panel observation are one surprise update.
    events = events.sort_values('anndats').drop_duplicates(['permno', 'fpe'])
    return events[['permno', 'TICKER', 'fpe', 'anndats', 'available_date',
                   'suescore', 'saleq_su']].reset_index(drop=True)


def build_expectation_state(grid):
    """Formation snapshot plus next filing's labels and their availability clock."""
    state = attach_ticker(grid, load_link())
    state['eom'] = month_end(state.eom).astype('datetime64[ns]')
    state = state.merge(load_prices().rename(columns={'STATPERS': 'eom'}),
                        on=['TICKER', 'eom'], how='left')
    events = load_surprises().rename(columns={'fpe': 'target_fpe', 'anndats': 'target_ann',
        'available_date': 'target_available', 'suescore': 'target_surprise_p'})
    state = pd.merge_asof(state.sort_values('eom'),
        events.drop(columns=['permno', 'saleq_su']).sort_values('target_ann'),
        by='TICKER', left_on='eom', right_on='target_ann', direction='forward',
        allow_exact_matches=False, tolerance=pd.Timedelta(days=120))
    state['STATPERS'] = state.eom
    return state.sort_values(['permno', 'eom']).reset_index(drop=True)
