"""Shared official-data clocks and market returns."""

from functools import lru_cache

import numpy as np
import pandas as pd

import config


@lru_cache(maxsize=2)
def trading_days(data_dir=None):
    """Trading dates observed in the supplied daily file, read in chunks."""
    dates = set()
    for chunk in pd.read_csv((data_dir or config.DATA_DIR) / 'dsf_mini_from_2014.csv',
                             usecols=['date'], chunksize=250000):
        dates.update(chunk.date.unique())
    return pd.DatetimeIndex(pd.to_datetime(sorted(dates), format='%Y%m%d'))


def next_trading_day(dates, calendar=None):
    """Strictly next supplied trading day; unavailable beyond the calendar."""
    calendar = trading_days() if calendar is None else calendar
    positions = calendar.searchsorted(pd.to_datetime(dates), side='right')
    values = np.full(len(positions), np.datetime64('NaT'), dtype='datetime64[ns]')
    valid = positions < len(calendar)
    values[valid] = calendar.values[positions[valid]]
    return pd.Series(values, index=dates.index)


def market_panel(data_dir=None):
    frame = pd.read_parquet((data_dir or config.DATA_DIR) / config.PANEL_FILE,
                            columns=['permno', 'eom', 'me', 'ret', 'ret_exc'])
    frame['month'] = pd.to_datetime(frame.eom).dt.to_period('M')
    return frame


def market_weights(frame):
    """Top 500 selected at the preceding calendar month-end."""
    weights = frame.loc[frame.me.gt(0), ['permno', 'month', 'me']].copy()
    weights = weights.sort_values(['month', 'me', 'permno'], ascending=[True, False, True])
    weights = weights.groupby('month').head(500).copy()
    weights['weight'] = weights.me / weights.groupby('month').me.transform('sum')
    weights['month'] += 1
    return weights[['permno', 'month', 'weight']]


def monthly_market(frame=None):
    frame = market_panel() if frame is None else frame
    rf = (frame.ret - frame.ret_exc).groupby(frame.month).median()
    matched = market_weights(frame).merge(frame[['permno', 'month', 'ret']],
                                         on=['permno', 'month'], how='left')
    # Missing constituent returns remain missing; normalize observed weights.
    matched = matched.dropna(subset=['ret'])
    numerator = (matched.weight * matched.ret).groupby(matched.month).sum()
    proxy = numerator / matched.groupby('month').weight.sum()
    return pd.DataFrame({'rf': rf, 'spx': proxy})


def daily_market(data_dir, first, last):
    """Fixed prior-month weights and geometrically spread monthly risk-free rate."""
    from evaluate import load_dsf

    panel = market_panel(data_dir)
    weights = market_weights(panel)
    weights = weights.loc[weights.month.between(first.to_period('M'), last.to_period('M'))]
    daily = load_dsf(data_dir, set(weights.permno), first, last)
    daily['month'] = daily.day.dt.to_period('M')
    matched = daily.merge(weights, on=['permno', 'month']).dropna(subset=['daily_return'])
    numerator = (matched.weight * matched.daily_return).groupby(matched.day).sum()
    proxy = numerator / matched.groupby('day').weight.sum()
    calendar = trading_days(data_dir)
    months = calendar.to_period('M')
    counts = pd.Series(1, index=calendar).groupby(months).transform('sum')
    rf = monthly_market(panel).rf.reindex(months).to_numpy()
    out = pd.DataFrame({'rf': np.power(1 + rf, 1 / counts.to_numpy()) - 1}, index=calendar)
    out['spx'] = proxy
    return out.loc[first:last]


def has_item(items, item):
    """Accept both numeric corpus labels and labels prefixed with Item."""
    if items is None:
        return False
    wanted = item.lower().removeprefix('item ').strip()
    return any(str(value).lower().removeprefix('item ').strip() == wanted
               for value in items)
