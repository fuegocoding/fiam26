"""Formation-time feature catalogue and bounded source loaders."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import config
from sleeves.filing_events import edgar

KEYS = ['permno', 'eom']
DAILY = {
    'return_5d': 'Compounded return over the last 5 observed trading days.',
    'return_21d': 'Compounded return over the last 21 observed trading days.',
    'max_return': 'Maximum daily return in the formation month.',
    'vol_21d': 'Sample standard deviation of the last 21 daily returns.',
    'up_share': 'Share of positive returns in the last 21 trading days.',
    'skew_21d': 'Sample skewness of the last 21 daily returns.',
}
OWNED = ['credit_card_spend', 'credit_card_transactions', 'credit_card_users', 'foot_traffic']


def fingerprint(paths):
    """Invalidate derived features when inputs or loader code change."""
    from pathlib import Path
    paths = [*paths, Path(__file__), Path(edgar.__file__)]
    return hashlib.sha256(json.dumps([(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                                     for p in paths]).encode()).hexdigest()[:20]


def daily_months(frame):
    """Compute trailing features before retaining the last row of each month."""
    frame = frame.sort_values(['permno', 'date']).copy()
    frame['eom'] = pd.to_datetime(frame.date.astype(str)).dt.to_period('M').dt.to_timestamp('M')
    groups = frame.groupby('permno', sort=False).daily_return
    for days in (5, 21):
        frame[f'return_{days}d'] = groups.transform(
            lambda x: (1 + x).rolling(days).apply(np.prod, raw=True) - 1)
    frame['vol_21d'] = groups.transform(lambda x: x.rolling(21).std())
    frame['skew_21d'] = groups.transform(lambda x: x.rolling(21).skew())
    frame['up_share'] = groups.transform(lambda x: x.gt(0).where(x.notna()).rolling(21).mean())
    frame['max_return'] = frame.groupby(KEYS).daily_return.transform('max')
    return frame.groupby(KEYS, sort=False).tail(1)[[*KEYS, *DAILY]]


def build_daily(path, output):
    """Stream the PERMNO-sorted official CSV, retaining boundary securities."""
    carry, parts, previous = pd.DataFrame(), [], -1
    for chunk in pd.read_csv(path, usecols=['permno', 'date', 'daily_return'], chunksize=150000):
        if not chunk.permno.is_monotonic_increasing or chunk.permno.iloc[0] < previous:
            raise ValueError('Daily CSV must be sorted by PERMNO')
        previous = chunk.permno.iloc[-1]
        chunk = pd.concat([carry, chunk], ignore_index=True)
        carry = chunk.loc[chunk.permno.eq(previous)]
        complete = chunk.loc[chunk.permno.ne(previous)]
        if not complete.empty:
            parts.append(daily_months(complete))
    if not carry.empty:
        parts.append(daily_months(carry))
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(KEYS).any():
        raise ValueError('Duplicate daily formation keys')
    result.to_parquet(output, index=False)


def map_owned(raw, labels, features, lag_months=1):
    """Map the observation-month identity before delaying availability."""
    raw = raw[['ticker', 'date', *features]].copy()
    raw['eom'] = pd.to_datetime(raw.pop('date')).dt.to_period('M').dt.to_timestamp('M')
    reference = pd.to_datetime(labels.ticker_name_reference_date)
    valid = (reference.notna() & reference.le(labels.eom)
             & labels.ticker_name_status.ne('no_verified_label_for_observation_period'))
    links = labels.loc[valid, [*KEYS, 'ticker']].dropna()
    # Ambiguous ticker-month mappings cannot identify a security safely.
    links = links.loc[~links.duplicated(['ticker', 'eom'], keep=False)]
    if raw.duplicated(['ticker', 'eom']).any():
        raise ValueError('Duplicate owned ticker-month values')
    mapped = links.merge(raw, on=['ticker', 'eom'], validate='one_to_one')[[*KEYS, *features]]
    mapped['eom'] = (mapped.eom.dt.to_period('M') + lag_months).dt.to_timestamp('M')
    return mapped


class Sources:
    """Lazy column loading; all joins use formation keys, never future labels."""

    def __init__(self, data_dir=None, cache_dir=None):
        self.data_dir = data_dir or config.DATA_DIR
        self.cache_dir = cache_dir or config.FINAL / 'discovery_cache' / 'sources'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.panel_path = self.data_dir / config.PANEL_FILE
        self.chars = pd.read_csv(self.data_dir / config.FACTOR_LIST_FILE).iloc[:, 0].tolist()
        descriptions = json.loads(Path(__file__).with_name('feature_descriptions.json').read_text())
        self.features = {
            'panel': {c: descriptions[c] for c in self.chars},
            'filings': {c: c.replace('_', ' ') + ' count; strictly next trading day availability' for c in edgar.FEATURE_COLUMNS},
            'daily': DAILY,
            'owned': {c: c.replace('_', ' ') + '; month end + one month (unknown tear-sheet lag)' for c in OWNED},
        }
        self._frames = {}

    def panel(self, columns, start, end):
        table = ds.dataset(self.panel_path).to_table(
            columns=list(dict.fromkeys([*KEYS, *columns])),
            filter=(ds.field('eom') >= pd.Timestamp(start).date()) &
                   (ds.field('eom') <= pd.Timestamp(end).date()), use_threads=False)
        frame = table.to_pandas()
        frame['eom'] = pd.to_datetime(frame.eom)
        if frame.duplicated(KEYS).any():
            raise ValueError('Duplicate panel formation keys')
        return frame

    def _source(self, source):
        if source in self._frames:
            return self._frames[source]
        if source == 'daily':
            path = self.data_dir / 'dsf_mini_from_2014.csv'
            output = self.cache_dir / ('daily_' + fingerprint([path]) + '.parquet')
            if not output.exists():
                build_daily(path, output)
            frame = pd.read_parquet(output)
        elif source == 'filings':
            path = self.data_dir / config.FILINGS_FILE
            folder = self.cache_dir / ('filings_' + fingerprint([path, self.data_dir / 'dsf_mini_from_2014.csv']) + '_' + str(config.LAST_TARGET.date()))
            frame = pd.read_parquet(edgar.build(self.panel_path, path, folder))
            frame['eom'] = (pd.to_datetime(frame.pop('return_month')).dt.to_period('M') - 1).dt.to_timestamp('M')
        elif source == 'owned':
            labels = self.panel(['ticker', 'ticker_name_reference_date', 'ticker_name_status'], '2014-01-01', config.LAST_TARGET)
            parts = []
            for filename, features in [('card_panel_unified.parquet', OWNED[:3]),
                                       ('foot_traffic_panel.parquet', OWNED[3:])]:
                path = self.data_dir / 'carbon/raw/framework' / filename
                raw = pd.read_parquet(path, columns=['ticker', 'date', *features])
                parts.append(map_owned(raw, labels, features))
            frame = parts[0].merge(parts[1], on=KEYS, how='outer', validate='one_to_one')
        else:
            raise ValueError('Not a loadable source: ' + source)
        self._frames[source] = frame
        return frame

    def load(self, source, features, start, end):
        if source not in self.features or not set(features) <= self.features[source].keys():
            raise ValueError(f'Unknown source/features: {source}/{features}')
        if source == 'panel':
            return self.panel(features, start, end)
        frame = self._source(source)
        return frame.loc[frame.eom.between(pd.Timestamp(start), pd.Timestamp(end)), [*KEYS, *features]].copy()

    def catalogue(self, end):
        """Coverage ends at the evidence cutoff, including for agent prompts."""
        result = {}
        labels = self.panel(['ticker', 'ticker_name_reference_date', 'ticker_name_status'], '2015-01-01', end)
        reference = pd.to_datetime(labels.ticker_name_reference_date)
        safe = reference.notna() & reference.le(labels.eom) & labels.ticker_name_status.ne(
            'no_verified_label_for_observation_period')
        labels.loc[~safe, 'ticker'] = None
        for source, features in self.features.items():
            frame = self.load(source, list(features) if source != 'panel' else [], '2015-01-01', end)
            tickers = frame[KEYS].merge(labels[[*KEYS, 'ticker']], on=KEYS, how='left', validate='one_to_one')
            result[source] = dict(features=features, tickers=int(tickers.ticker.nunique()),
                                  securities=int(frame.permno.nunique()),
                                  months=int(frame.eom.nunique()),
                                  first=str(frame.eom.min()), last=str(frame.eom.max()))
        result['panel']['availability'] = 'Official characteristics available at formation eom.'
        result['filings']['availability'] = 'Strictly next supplied trading day after filing_date; 1/3/12 calendar-month counts.'
        result['daily']['availability'] = 'Observed trading returns through formation only; no later daily rows.'
        result['owned']['availability'] = 'Month end plus one calendar month, unknown tear-sheet lag assumed; dated panel ticker links only. Current extracts do not reconstruct revisions.'
        result['carbon_catalogue'] = dict(features={}, loadable=False,
            rule='Free search/quotes only; shopping list never enters scores. Purchases prohibited.')
        return result

    def frame(self, proposal, start, end, labels=False):
        """Load a three-month warmup for exact calendar changes."""
        warmup = pd.Timestamp(start) - pd.offsets.MonthEnd(3)
        fields = ['size_grp', 'gics', 'prc'] + (['ret_exc_lead1m'] if labels else [])
        frame = self.panel(fields, warmup, end)
        for item in proposal['inputs']:
            source, feature = item['source'], item['feature']
            name = source + ':' + feature
            if name not in frame:
                data = self.load(source, [feature], warmup, end).rename(columns={feature: name})
                frame = frame.merge(data, on=KEYS, how='left', validate='one_to_one')
                if source == 'filings':
                    frame[name] = frame[name].fillna(0)
            transform = item.get('transform', 'level')
            transformed = name + ':' + transform
            if transform == 'chg3':
                lag = frame[[*KEYS, name]].copy()
                lag['eom'] = (lag.eom.dt.to_period('M') + 3).dt.to_timestamp('M')
                lag = lag.rename(columns={name: '_lag'})
                frame = frame.merge(lag, on=KEYS, how='left', validate='one_to_one')
                frame[transformed] = frame[name] - frame.pop('_lag')
            elif transform == 'sector_rank':
                sector = frame.gics.astype('string').str.replace(r'\.0$', '', regex=True).str[:2]
                frame[transformed] = frame[name].groupby([frame.eom, sector]).rank(pct=True)
            else:
                frame[transformed] = frame[name]
        frame = frame.loc[frame.eom.ge(pd.Timestamp(start))].reset_index(drop=True)
        frame['return_month'] = frame.eom.dt.to_period('M') + 1
        return frame
