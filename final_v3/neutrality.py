"""Formation-date market beta used consistently by sizing and validation."""

import numpy as np
import pandas as pd
import json
from pathlib import Path

SOURCE = 'Causal S&P 500 excess-return regressions: 252 daily sessions and 36 months; explicit formation-sector median priors for short histories'
LEGACY_SOURCE = 'Unclipped formation-date panel beta_60m; missing estimates use beta 1.0'
COLUMNS = ('beta_sp500_daily', 'beta_sp500_monthly')
ARCANA_COLUMN = 'beta_arcana_spy'
HYBRID_COLUMN = 'beta_spy_guardrail'
DAILY_WINDOW, DAILY_MIN = 252, 126
MONTHLY_WINDOW, MONTHLY_MIN = 36, 12


def rolling_slopes(returns, market, window, minimum):
    """OLS slopes with intercept, using pairwise-complete trailing observations.

    Both axes must represent the complete calendar, so missing stock returns
    never lengthen the lookback or become invented zero returns.
    """
    market = market.reindex(returns.index)
    valid = returns.notna().mul(market.notna(), axis=0)
    n = valid.rolling(window, min_periods=1).sum()
    x = valid.mul(market, axis=0)
    y = returns.where(valid)
    sx = x.rolling(window, min_periods=1).sum()
    sy = y.rolling(window, min_periods=1).sum()
    sxx = valid.mul(market**2, axis=0).rolling(window, min_periods=1).sum()
    sxy = y.mul(market, axis=0).rolling(window, min_periods=1).sum()
    denominator = sxx - sx**2/n
    beta = (sxy-sx*sy/n)/denominator
    return beta.where((n >= minimum) & (denominator > 1e-12)), n


def attach(frame, panel_path, data_dir, output_dir):
    """Estimate risk exposures before each holding month, never from its outcome."""
    import official
    frame = frame.copy()
    months = sorted(pd.to_datetime(frame.return_month).dt.to_period('M').unique())
    formations = {m:(m-1).to_timestamp('M') for m in months}
    first, last = formations[months[0]], formations[months[-1]]
    ids = pd.Index(sorted(frame.permno.unique()))
    source = official.reporting_sources()
    prices = source.loc[source.series.eq('SP500')].set_index('date').value.dropna().sort_index()
    rates = source.loc[source.series.eq('TB3MS')].set_index('date').value.sort_index()
    rates.index = rates.index.to_period('M')
    # Same S&P 500 benchmark as reporting. Lag TB3MS for risk estimation:
    # contemporaneous monthly average rates were not yet published at formation.
    available_rates = rates.reindex(pd.period_range(rates.index.min(), rates.index.max(),freq='M')).shift(2)
    monthly_prices = prices.resample('ME').last()
    market_monthly = monthly_prices.pct_change(fill_method=None)
    market_monthly.index = market_monthly.index.to_period('M')
    rf_monthly = available_rates.reindex(market_monthly.index)/1200
    market_monthly = market_monthly-rf_monthly
    panel = pd.read_parquet(panel_path, columns=['permno','eom','ret'])
    panel['month'] = pd.to_datetime(panel.eom).dt.to_period('M')
    panel = panel.loc[panel.permno.isin(ids)]
    stock_monthly = panel.pivot(index='month',columns='permno',values='ret').reindex(index=market_monthly.index,columns=ids)
    stock_monthly = stock_monthly.sub(rf_monthly,axis=0)
    bm, nm = rolling_slopes(stock_monthly,market_monthly,MONTHLY_WINDOW,MONTHLY_MIN)
    dates = official.trading_days(data_dir)
    end_position = dates.searchsorted(last, side='right')
    start_position = max(0, dates.searchsorted(first,side='right')-DAILY_WINDOW)
    fit_dates = dates[start_position:end_position]
    market_daily = prices.pct_change(fill_method=None).reindex(fit_dates)
    rf_daily = pd.Series(available_rates.reindex(fit_dates.to_period('M')).to_numpy()/25200,index=fit_dates)
    market_daily = market_daily-rf_daily
    parts = []
    for chunk in pd.read_csv(Path(data_dir)/'dsf_mini_from_2014.csv',
                             usecols=['permno','date','daily_return'],chunksize=250000):
        chunk = chunk.loc[chunk.permno.isin(ids)].copy()
        chunk['day'] = pd.to_datetime(chunk.date,format='%Y%m%d')
        chunk = chunk.loc[chunk.day.between(fit_dates.min(),last)]
        if len(chunk): parts.append(chunk[['day','permno','daily_return']])
    daily = pd.concat(parts,ignore_index=True)
    if daily.duplicated(['day','permno']).any():
        raise ValueError('Duplicate stock/day returns in beta history')
    stock_daily = daily.pivot(index='day',columns='permno',values='daily_return').reindex(index=fit_dates,columns=ids)
    stock_daily = stock_daily.sub(rf_daily,axis=0)
    bd, nd = rolling_slopes(stock_daily,market_daily,DAILY_WINDOW,DAILY_MIN)
    estimates = []
    frame_month = pd.to_datetime(frame.return_month).dt.to_period('M')
    for month in months:
        cutoff = formations[month]
        day = fit_dates[fit_dates<=cutoff].max()
        if pd.isna(day): raise ValueError(f'{month}: no pre-formation market history')
        meta = frame.loc[frame_month.eq(month),['permno','sector']].set_index('permno')
        row = meta.copy()
        for label, values, counts, stamp in [('daily',bd,nd,day),('monthly',bm,nm,month-1)]:
            column = 'beta_sp500_'+label
            raw = values.loc[stamp].reindex(meta.index)
            observed = counts.loc[stamp].reindex(meta.index).fillna(0)
            # Priors use only observed peers at this same formation cutoff.
            peers = pd.DataFrame({'beta':raw,'sector':meta.sector})
            sector_prior = peers.groupby('sector').beta.median()
            global_prior = raw.median()
            if not np.isfinite(global_prior): raise ValueError(f'{month}: insufficient {label} beta history')
            row[column] = raw.fillna(meta.sector.map(sector_prior)).fillna(global_prior)
            row[column+'_observations'] = observed.astype(int)
            row[column+'_source'] = np.where(raw.notna(),'stock_regression',
                np.where(meta.sector.map(sector_prior).notna(),'formation_sector_median','formation_cross_section_median'))
        row['return_month'] = month
        row['beta_information_cutoff'] = cutoff
        estimates.append(row.reset_index().drop(columns='sector'))
    estimates = pd.concat(estimates,ignore_index=True)
    frame['return_month'] = frame_month
    frame = frame.merge(estimates,on=['return_month','permno'],how='left',validate='one_to_one')
    if not np.isfinite(frame[list(COLUMNS)].to_numpy()).all():
        raise ValueError('Missing causal S&P 500 beta estimates')
    if not (frame.beta_information_cutoff < frame.return_month.dt.to_timestamp()).all():
        raise ValueError('Beta risk estimate uses holding-period information')
    output = Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    stored = estimates.copy();stored['return_month'] = stored.return_month.astype(str)
    stored.to_parquet(output/'beta_estimates.parquet',index=False)
    (output/'beta_estimation.json').write_text(json.dumps(dict(source=SOURCE,
        daily_window=DAILY_WINDOW,daily_minimum=DAILY_MIN,monthly_window=MONTHLY_WINDOW,
        monthly_minimum=MONTHLY_MIN,benchmark='S&P 500 price index from supplied reporting file',
        risk_free='TB3MS lagged two calendar months; daily annual percent / 25200, monthly / 1200',
        fallback='Formation-sector median of observed stock regressions, then formation cross-sectional median; no fixed beta-1 fill',
        period=str(months[0])+'..'+str(months[-1]),rows=len(stored)),indent=2)+'\n')
    print(f'Causal S&P 500 daily/monthly beta estimates prepared for {len(months)} formations',flush=True)
    return frame


def matrix(frame, ids):
    meta = frame.set_index('permno')
    if not meta.index.is_unique or not pd.Index(ids).isin(meta.index).all():
        raise ValueError('Beta constraint requires unique formation metadata for every holding')
    if all(c in meta for c in COLUMNS):
        columns = list(COLUMNS) + ([HYBRID_COLUMN] if HYBRID_COLUMN in meta else [])
        result = meta.loc[:,columns].reindex(ids).astype(float)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError('Missing or nonfinite beta for a selected security')
        return result
    # Archived allocations retain their historical estimator when replayed.
    return coefficients(frame,ids).to_frame('beta_60m')


def coefficients(frame, ids):
    """Return historical betas without substituting today's risk-model values."""
    meta = frame.set_index('permno')
    if not meta.index.is_unique or not pd.Index(ids).isin(meta.index).all():
        raise ValueError('Beta constraint requires unique formation metadata for every holding')
    if all(c in meta for c in COLUMNS):
        return matrix(frame,ids).iloc[:,0]
    beta = meta.beta_60m.reindex(ids).astype(float).fillna(1.0)
    if not np.isfinite(beta).all():
        raise ValueError('Beta constraint contains nonfinite formation estimates')
    return beta


def exposure(weights, frame):
    book = pd.Series(weights, dtype=float)
    return float(book.dot(coefficients(frame, book.index)))


def exposures(weights, frame):
    book = pd.Series(weights,dtype=float)
    return matrix(frame,book.index).mul(book,axis=0).sum()


def enforce(weights, frame, limit):
    values = exposures(weights, frame)
    beta = float(values.iloc[0])
    if limit is not None:
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError('Final beta limit must be positive and finite')
        if values.abs().gt(limit+1e-9).any():
            raise ValueError(f'Final formation beta {values.to_dict()} exceeds +/-{limit:.4f}')
    return beta


def attach_arcana(frame, path):
    """Attach available Arcana-US stock betas vs SPY, with local daily-beta fallback.

    The guardrail column records which source was used for each security/month.
    This supports partial vendor exports without mislabeling fallback estimates.
    """
    data = pd.read_csv(Path(path), dtype={'return_month':str, 'holding_month':str, 'permno':str, 'asset_id':str,
                                    'arcana_asset_id':str,
                                    'risk_model_id':str, 'benchmark_asset_id':str,
                                    'as_of_date':str, 'available_at':str})
    # Normalize the fetched workbook export (which reflects the user-supplied
    # Arcana-US / SPY metric) to the explicit input schema used by this workflow.
    if {'holding_month','arcana_asset_id','observation_timestamp','arcana_spy_beta'}.issubset(data.columns):
        data = data.rename(columns={'holding_month':'return_month','arcana_asset_id':'asset_id',
                                    'observation_timestamp':'available_at','arcana_spy_beta':ARCANA_COLUMN})
        data['return_month'] = pd.to_datetime(data.return_month,errors='raise').dt.to_period('M').astype(str)
        data['as_of_date'] = (pd.to_datetime(data.available_at,errors='coerce',utc=True)
            .dt.tz_convert('America/New_York').dt.date.astype('string'))
        data['risk_model_id'] = '3'
        data['benchmark_asset_id'] = '21'
    required = {'return_month','permno','asset_id','risk_model_id','benchmark_asset_id',
                'as_of_date','available_at',ARCANA_COLUMN}
    if not required.issubset(data.columns):
        raise ValueError(f'Arcana beta input missing columns: {sorted(required-set(data.columns))}')
    # Drop incomplete observations; they will use the disclosed local fallback.
    data = data.dropna(subset=list(required)).copy()
    if not data.risk_model_id.eq('3').all() or not data.benchmark_asset_id.eq('21').all():
        raise ValueError('Arcana beta input must use Arcana-US model 3 and SPY asset 21')
    if not data.permno.str.fullmatch(r'[1-9]\d*').all() or not data.asset_id.str.fullmatch(r'[1-9]\d*').all():
        raise ValueError('Arcana beta identifiers must be positive integer strings')
    if not data.return_month.str.fullmatch(r'\d{4}-\d{2}').all():
        raise ValueError('Arcana beta return_month must be YYYY-MM')
    if not data.as_of_date.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
        raise ValueError('Arcana beta as_of_date must be YYYY-MM-DD')
    if not data.available_at.str.contains(r'(?:Z|[+-]\d\d:\d\d)$', regex=True).all():
        raise ValueError('Arcana beta available_at requires an explicit UTC offset')
    data['return_month'] = pd.PeriodIndex(data.return_month, freq='M')
    data['permno'] = data.permno.astype(int)
    data[ARCANA_COLUMN] = pd.to_numeric(data[ARCANA_COLUMN],errors='raise')
    data['as_of_date'] = pd.to_datetime(data.as_of_date,errors='raise')
    available = pd.to_datetime(data.available_at,utc=True,errors='raise')
    cutoff = data.return_month.dt.to_timestamp().dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    earliest = (data.return_month - 1).dt.to_timestamp()
    available_date = available.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    if ((data.as_of_date < earliest) | (data.as_of_date >= data.return_month.dt.to_timestamp())
        | (available_date < data.as_of_date) | (available >= cutoff)).any():
        raise ValueError('Arcana beta must be observed and available before the holding month')
    if not np.isfinite(data[ARCANA_COLUMN]).all():
        raise ValueError('Arcana beta contains nonfinite values')
    if data.duplicated(['return_month','permno']).any() or data.duplicated(['return_month','asset_id']).any():
        raise ValueError('Arcana beta has duplicate month/security or month/asset mappings')
    frame = frame.copy()
    if not isinstance(frame.return_month.dtype, pd.PeriodDtype):
        frame['return_month'] = pd.to_datetime(frame.return_month).dt.to_period('M')
    frame = frame.merge(data[['return_month','permno',ARCANA_COLUMN,'asset_id','as_of_date','available_at']],
                        on=['return_month','permno'],how='left',validate='one_to_one')
    frame[HYBRID_COLUMN] = frame[ARCANA_COLUMN].where(frame[ARCANA_COLUMN].notna(),frame['beta_sp500_daily'])
    frame['beta_spy_guardrail_source'] = np.where(frame[ARCANA_COLUMN].notna(),
        'arcana_us_model3_spy21','local_252d_regression_fallback')
    if not np.isfinite(frame[HYBRID_COLUMN]).all():
        raise ValueError('Hybrid SPY beta guardrail has missing or nonfinite exposures')
    return frame
